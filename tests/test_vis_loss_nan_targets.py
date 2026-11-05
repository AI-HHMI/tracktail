"""A non-finite visibility target must not destroy the backward pass.

`vis_true` / `vis_true_cams` are three-state in any annotation format that distinguishes
"occluded" from "never assessed". Before this was guarded, a NaN target produced a loss curve
that fell perfectly normally while every parameter received NaN gradients -- measured downstream
as 36 of 40 training steps skipped, with nothing in the printed loss to suggest a problem.

The mechanism is worth stating because it is not obvious: `TotalLoss` drops non-finite sub-losses
from its forward sum, but that filter is a `masked_select`, whose backward hands the dropped
element a real 0.0 gradient. `BCEWithLogitsBackward` then computes
`0.0 * (sigmoid(x) - NaN) = NaN`. Filtering is not detaching.
"""
import torch
import torch.nn.functional as F

from tracktail.model.losses import BCELossVis


def _grads(target, weight=1.0):
    pred = torch.zeros(2, 4, 3, 1, requires_grad=True)
    loss = BCELossVis(weight=weight)(pred, target, device='cpu')
    if not torch.isfinite(loss):
        return loss, None
    loss.backward()
    return loss, pred.grad


def test_nan_target_does_not_poison_the_gradient():
    target = torch.randint(0, 2, (2, 4, 3, 1)).float()
    target[0, 1, 2, 0] = float('nan')
    loss, grad = _grads(target)
    assert torch.isfinite(loss), 'a partially-assessed target must still give a finite loss'
    assert torch.isfinite(grad).all(), 'NaN target poisoned the backward pass'


def test_masked_loss_equals_the_assessed_subset():
    target = torch.randint(0, 2, (2, 4, 3, 1)).float()
    target[0, 1, 2, 0] = float('nan')
    target[1, 0, 0, 0] = float('nan')
    pred = torch.zeros(2, 4, 3, 1)
    ok = torch.isfinite(target)
    want = F.binary_cross_entropy_with_logits(pred[ok], target[ok], reduction='mean')
    got, _ = _grads(target)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=0)


def test_fully_assessed_target_is_unchanged():
    """The guard must be a no-op when nothing is missing, or it silently rescales the term."""
    target = torch.randint(0, 2, (2, 4, 3, 1)).float()
    pred = torch.zeros(2, 4, 3, 1)
    want = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')
    got, _ = _grads(target)
    torch.testing.assert_close(got, want, atol=1e-6, rtol=0)


def test_nothing_assessed_is_inert():
    """All-NaN means "no supervision here", which is the same signal as a zero weight."""
    loss, grad = _grads(torch.full((2, 4, 3, 1), float('nan')))
    assert not torch.isfinite(loss) and grad is None


def test_zero_weight_is_still_inert():
    loss, grad = _grads(torch.randint(0, 2, (2, 4, 3, 1)).float(), weight=0.0)
    assert not torch.isfinite(loss) and grad is None
