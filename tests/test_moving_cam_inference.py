"""Item 10: moving-camera inference loads per-frame extrinsics instead of silently going static."""
import os
import tempfile

import pytest
import torch
import numpy as np
import yaml

from tracktail.inference.inference_utils import load_camera_group_from_metadata


def _identity_extrinsics(T):
    E = np.eye(4, dtype=np.float64)
    return np.stack([E.copy() for _ in range(T)], axis=0)   # (T,4,4)


def _write_metadata(path, moving, T, malformed=False):
    # A single camera whose frame-t extrinsic translates the rig by t along x.
    ext = []
    for t in range(T):
        E = np.eye(4, dtype=np.float64)
        E[0, 3] = float(t) if moving else 0.0
        ext.append(E)
    ext = np.stack(ext, axis=0) if moving else ext[0]
    if malformed:
        ext = np.eye(3, dtype=np.float64)                     # wrong rank -> must raise

    meta = {
        'cam_type': 'pinhole',
        'camera_heights': {'cam0': 512},
        'camera_widths': {'cam0': 512},
        'intrinsic_matrices': {'cam0': [[600.0, 0.0, 256.0],
                                        [0.0, 600.0, 256.0],
                                        [0.0, 0.0, 1.0]]},
        'extrinsic_matrices': {'cam0': ext.tolist()},
        'distortion_matrices': {'cam0': [0.0, 0.0, 0.0, 0.0, 0.0]},
        'offset_dict': {'cam0': [0.0, 0.0]},
    }
    if moving:
        meta['moving_cams'] = True
    with open(path, 'w') as f:
        yaml.safe_dump(meta, f)


def test_static_metadata_loads_static_extrinsic():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'metadata.yaml')
        _write_metadata(p, moving=False, T=4)
        cg = load_camera_group_from_metadata(p, device='cpu')
        assert len(cg) == 1
        assert cg[0]['ext'].ndim == 2, 'a static rig must stay (4,4)'


def test_moving_metadata_loads_per_frame_extrinsics():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'metadata.yaml')
        _write_metadata(p, moving=True, T=4)
        cg = load_camera_group_from_metadata(p, device='cpu')
        assert len(cg) == 1
        # The fix: per-frame (T,4,4) extrinsics instead of the static fallback.
        assert cg[0]['ext'].ndim == 3 and cg[0]['ext'].shape[0] == 4
        # And the frame-t translation actually landed (0..3 along x), not frame 0 for all.
        assert torch.allclose(cg[0]['ext'][:, 0, 3], torch.arange(4, dtype=torch.float))


def test_moving_metadata_with_malformed_extrinsics_raises():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'metadata.yaml')
        _write_metadata(p, moving=True, T=4, malformed=True)
        # The old code silently fell back to a static extrinsic; it must refuse instead.
        with pytest.raises(ValueError, match='moving_cams=true'):
            load_camera_group_from_metadata(p, device='cpu')
