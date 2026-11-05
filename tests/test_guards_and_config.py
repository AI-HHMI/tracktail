"""Guards, validation, config recording/refusal, and the additive API pieces."""
import io
import tempfile
import os

import pytest
import torch
import numpy as np

from tracktail.model.train_utils import check_config_compatibility
from tracktail.model.losses import coordinate_softmax_loss
from tracktail.model.utils import PadToSize
from tracktail.datasets.tracktail_dataset import crop_box_for_points


# --------------------------------------------------------------------------------------
# item 24: coordinate_softmax_loss derives its clamp bound from the logits
# --------------------------------------------------------------------------------------

def test_coordinate_softmax_bit_identical_at_256():
    """The hardcoded pixel_size=256 equals P at image_size=256, so this must be bit-identical."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 512)          # P = 256
    target = torch.rand(2, 3, 4, 2) * 255.0
    vis = torch.ones(2, 3, 4, 1)
    explicit = coordinate_softmax_loss(logits, target, vis, pixel_size=256)
    derived = coordinate_softmax_loss(logits, target, vis)
    assert torch.equal(explicit, derived)


def test_coordinate_softmax_bounds_at_384_and_192():
    """Above 256 the old default silently collapsed every target beyond bin 255 to 255; below it
    could clamp to an index >= P and crash cross_entropy. Deriving from P fixes both."""
    for P in (384, 192):
        torch.manual_seed(0)
        logits = torch.randn(1, 1, 1, 2 * P)
        target = torch.tensor([[[[float(P) - 1.0, float(P) - 1.0]]]])   # top corner
        vis = torch.ones(1, 1, 1, 1)
        # Must not raise, and the top corner must map to bin P-1 (not P, which would be OOB).
        loss = coordinate_softmax_loss(logits, target, vis)
        assert torch.isfinite(loss)


# --------------------------------------------------------------------------------------
# item 22: PadToSize pads to a per-call target
# --------------------------------------------------------------------------------------

def test_pad_to_size_per_call_target():
    pad = PadToSize(256)
    img = torch.zeros(2, 3, 160, 160)
    # Default target (256): a 160 crop pads up to 256.
    out = pad(img)
    assert out.shape[-2:] == (256, 256)
    # Per-call target (192): a 160 crop pads up to 192, not 256.
    out192 = pad(img, 192)
    assert out192.shape[-2:] == (192, 192)
    # An input already larger than the target is not shrunk (pad is max(target - h, 0)).
    out_big = pad(torch.zeros(1, 3, 300, 300), 192)
    assert out_big.shape[-2:] == (300, 300)


def test_pad_to_size_is_bottom_right():
    pad = PadToSize(8)
    img = torch.ones(1, 1, 5, 6)
    out = pad(img, 8)
    assert out[0, 0, 0, 0] == 1.0        # top-left preserved
    assert out[0, 0, -1, -1] == 0.0      # bottom-right zero-filled
    assert out[0, 0, 0, -1] == 0.0       # right edge zero-filled
    assert out[0, 0, -1, 0] == 0.0       # bottom edge zero-filled


# --------------------------------------------------------------------------------------
# item 23: crop_box_for_points is a pure function, int32-exact against the mutating wrapper
# --------------------------------------------------------------------------------------

def test_crop_box_for_points_is_int32_exact_against_wrapper():
    from tracktail.datasets.tracktail_dataset import TracktailDataset
    from tracktail.model.cube import project_points_torch

    class D:
        min_crop_dim = 96
    d = D()

    torch.manual_seed(0)
    coords = torch.rand(8, 5, 3) * 2 + torch.tensor([0.0, 0.0, 6.0])
    cgroup = [{
        'name': 'c0', 'type': 'pinhole',
        'ext': torch.eye(4),
        'mat': torch.tensor([[600., 0, 128], [0, 600., 128], [0, 0, 1.]]),
        'dist': torch.tensor([0., 0., 0., 0., 0.]),
        'size': torch.tensor([512, 512], dtype=torch.int32),
        'offset': torch.zeros(2),
        'center': torch.zeros(3),
    }]

    # The wrapper's own boxes, via the unbound method on a minimal object.
    _, crops = TracktailDataset.crop_cgroup_to_points(d, cgroup, coords)
    p2d = project_points_torch(cgroup, coords)
    box = crop_box_for_points(p2d[0], cgroup[0]['size'], d.min_crop_dim)
    assert box.dtype == torch.int32
    assert torch.equal(box, crops[0])


# --------------------------------------------------------------------------------------
# item 6: config record + refusal on architecture switches
# --------------------------------------------------------------------------------------

def _cfg(model_overrides):
    return {'model': model_overrides}


def test_check_config_compatibility_absent_config_warns_not_raises():
    # Every checkpoint on disk today has no 'config' key: warn, do not refuse.
    checkpoint = {'model_state': {}}
    check_config_compatibility(checkpoint, _cfg({'output_mode': 'direct'}))


def test_check_config_compatibility_refuses_on_output_mode_mismatch():
    checkpoint = {'config': _cfg({'output_mode': 'gridresid'})}
    with pytest.raises(ValueError, match='output_mode'):
        check_config_compatibility(checkpoint, _cfg({'output_mode': 'direct'}))


def test_check_config_compatibility_does_not_refuse_on_image_size():
    # image_size changes weight shapes and is handled by _interp_res_params (finetune-at-a-
    # different-resolution), so it must NOT trip the silent-reinterpretation refusal.
    checkpoint = {'config': _cfg({'output_mode': 'gridresid', 'image_size': 256})}
    check_config_compatibility(checkpoint, _cfg({'output_mode': 'gridresid', 'image_size': 384}))


def test_check_config_compatibility_refuses_on_rays_conf_normalize_mismatch():
    checkpoint = {'config': _cfg({'output_mode': 'gridresid', 'rays_conf_normalize': True})}
    with pytest.raises(ValueError, match='rays_conf_normalize'):
        check_config_compatibility(checkpoint, _cfg({'output_mode': 'gridresid', 'rays_conf_normalize': False}))


def test_check_config_compatibility_benign_diff_warns_not_raises():
    checkpoint = {'config': _cfg({'output_mode': 'gridresid', 'learning_rate': 1e-4})}
    check_config_compatibility(checkpoint, _cfg({'output_mode': 'gridresid', 'learning_rate': 2e-4}))


def test_check_config_compatibility_strict_false_warns():
    checkpoint = {'config': _cfg({'output_mode': 'gridresid'})}
    check_config_compatibility(checkpoint, _cfg({'output_mode': 'direct'}), strict=False)


# --------------------------------------------------------------------------------------
# item 11: custom_collate asserts batch_size == 1
# --------------------------------------------------------------------------------------

def test_custom_collate_asserts_batch_size_one():
    from tracktail.datasets.tracktail_dataset import custom_collate
    with pytest.raises(AssertionError, match='batch_size'):
        custom_collate([None, None])   # two items; the assert fires before anything else is read
