"""Loss-side guards: the vis pair validation (item 12) and SmoothnessLoss degradation (item 18)."""
import pytest
import torch

from tracktail.model.losses import TotalLoss, SmoothnessLoss


class _Stub:
    training = False
    S = 24
    stride_overlap = 4


def _outputs():
    return {
        'coords_pred': torch.zeros(1, 4, 2, 3),
        'vis_pred': torch.zeros(1, 4, 2, 1),
        'conf_pred': torch.zeros(1, 4, 2, 1),
        '2d_pred': torch.zeros(1, 1, 4, 2, 2),
        'vis_pred_2d': torch.zeros(1, 1, 4, 2),
        'conf_pred_2d': torch.zeros(1, 1, 4, 2, 1),
        'depth_pred': torch.zeros(1, 1, 4, 2, 1),
    }


def test_vis_true_alone_raises():
    loss = TotalLoss()
    with pytest.raises(ValueError, match='both-or-neither'):
        loss.forward(_Stub(), _outputs(), torch.zeros(1, 4, 2, 3),
                     vis_true=torch.ones(1, 4, 2, 1), vis_true_cams=None)


def test_vis_true_cams_alone_raises():
    loss = TotalLoss()
    with pytest.raises(ValueError, match='both-or-neither'):
        loss.forward(_Stub(), _outputs(), torch.zeros(1, 4, 2, 3),
                     vis_true=None, vis_true_cams=torch.ones(1, 1, 4, 2, 1))


def test_vis_pair_both_none_does_not_raise():
    loss = TotalLoss()
    # Both None is the legitimate "derive both from geometry" path; it must reach PAST the pair
    # guard. The forward will fail later on these minimal fake inputs (no camera group), but that
    # failure must NOT be the pair guard.
    try:
        loss.forward(_Stub(), _outputs(), torch.zeros(1, 4, 2, 3),
                     vis_true=None, vis_true_cams=None, device='cpu')
    except Exception as e:
        assert 'both-or-neither' not in str(e), f'pair guard fired on a both-None call: {e}'


# --------------------------------------------------------------------------------------
# item 18: SmoothnessLoss degrades instead of raising on short windows
# --------------------------------------------------------------------------------------

def test_smoothness_short_window_degrades_to_first_difference():
    s = SmoothnessLoss(order=4)
    pred = torch.tensor([[[0.0], [1.0]]])          # T=2
    true = torch.tensor([[[0.0], [1.0]]])
    vis = torch.ones(1, 2, 1, 1)
    # order=4 at T=2 must clamp to a first difference and return a finite scalar, not raise.
    out = s.forward(pred, true, vis, time_dim=1, device='cpu')
    assert torch.isfinite(out)


def test_smoothness_single_frame_returns_zero():
    s = SmoothnessLoss(order=4)
    pred = torch.tensor([[[0.0]]])                 # T=1
    true = torch.tensor([[[0.0]]])
    vis = torch.ones(1, 1, 1, 1)
    out = s.forward(pred, true, vis, time_dim=1, device='cpu')
    assert out.item() == 0.0
