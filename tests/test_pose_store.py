"""pose3d.pq is an exact, frame-sliceable stand-in for pose3d.npz (pose + vis)."""
import numpy as np
import pytest

pytest.importorskip('pyarrow')
from tracktail.datasets import pose_store as ps


def _make(tmp_path, T=75, N=13, C=4, S=2):
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(S, T, N, 3)).astype(np.float32)
    pose[0, 5, 2] = np.nan
    vis = rng.random((S, T, N, C)) > 0.3
    d = tmp_path / 'trial'
    d.mkdir()
    np.savez_compressed(d / 'pose3d.npz', pose=pose, vis=vis, keypoints=np.arange(N))
    return str(d / 'pose3d.npz'), pose, vis


def test_fallback_without_pq(tmp_path):
    npz, pose, vis = _make(tmp_path)
    assert ps.pq_path_for(npz) is None
    p, v = ps.load_pose_clip(npz, 3, 3 + 10 * 2, 2)
    np.testing.assert_array_equal(p, pose[:, 3:23:2])
    np.testing.assert_array_equal(v, vis[:, 3:23:2])


@pytest.mark.parametrize('start,n,interval', [(0, 24, 1), (30, 10, 3), (60, 15, 1), (7, 1, 1)])
def test_pq_matches_npz_slices(tmp_path, start, n, interval):
    npz, pose, vis = _make(tmp_path)
    ps.convert_npz_to_pq(npz, row_group=8)
    assert ps.pq_path_for(npz) is not None
    end = start + n * interval
    p, v = ps.load_pose_clip(npz, start, end, interval)
    np.testing.assert_array_equal(p, pose[:, start:end:interval])
    np.testing.assert_array_equal(v, vis[:, start:end:interval])
    assert p.dtype == np.float32 and v.dtype == bool
    np.testing.assert_array_equal(ps.load_pose_full(npz), pose)
    p2, v2 = ps.load_pose_clip(npz, start, end, interval, with_vis=False)
    assert v2 is None
