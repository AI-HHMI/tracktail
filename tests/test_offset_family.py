"""Per-frame camera offset: the (T,2) case, at every rank the library actually passes.

The library gave `ext` a time axis and never gave `offset` one. A moving CROP -- one that follows
the subject per frame instead of standing still over the window -- is expressible as a per-frame
camera offset and nothing else, so these five call sites are one bug with five faces.

The bar for every test here is: **the static path must be bit-identical**, because every run that
does not use a moving crop takes it.
"""
import pytest
import torch

from tracktail.model.cube import (project_cam, undistort_points, points_to_rays,
                                    get_camera_scale, project_points_torch)


def make_cam(offset=None, T=None, moving_ext=False):
    ext = torch.eye(4, dtype=torch.float32)
    ext[:3, 3] = torch.tensor([0.1, -0.2, 5.0])
    if moving_ext:
        ext = ext[None].repeat(T, 1, 1)
    cam = {
        'name': 'c0',
        'type': 'pinhole',
        'ext': ext,
        'mat': torch.tensor([[600.0, 0.0, 128.0],
                             [0.0, 600.0, 128.0],
                             [0.0, 0.0, 1.0]]),
        'dist': torch.tensor([0.01, -0.005, 0.001, 0.002, 0.0]),
        'size': torch.tensor([256, 256], dtype=torch.int32),
    }
    if offset is not None:
        cam['offset'] = offset
    R = ext[..., :3, :3]
    t = ext[..., :3, 3]
    cam['center'] = -torch.einsum('...ji,...j->...i', R, t)
    return cam


def rand_p3d(*shape):
    g = torch.Generator().manual_seed(0)
    p = torch.rand(*shape, 3, generator=g) * 2 - 1
    p[..., 2] += 6.0          # in front of the camera
    return p


# --------------------------------------------------------------------------------------
# 1. project_cam
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(7, 3), (4, 7, 3), (2, 4, 7, 3)])
@pytest.mark.parametrize('downsample_factor', [1, 2])
def test_a_static_offset_takes_the_original_code_path_exactly(shape, downsample_factor):
    """A 1-D offset must be bit-identical to the original code path. Every current run is here.

    The original subtracts the offset BEFORE the downsample division, all in float64, and rounds
    to float32 only at the end. Passing a float64 input makes project_cam return that exact value
    (no final rounding), so the float32 result must equal it rounded once. A reimplementation
    that instead subtracted offset/downsample_factor from the ALREADY-divided float32 result
    (the detail the report warns a wrapper gets wrong) differs here in the last ulp.
    """
    p3d = rand_p3d(*shape[:-1])
    off = torch.tensor([12.0, -5.0])
    got = project_cam(make_cam(offset=off), p3d, downsample_factor)
    exact = project_cam(make_cam(offset=off), p3d.double(), downsample_factor)
    assert torch.equal(got, exact.to(torch.float32))


def test_a_static_offset_of_zero_equals_no_offset_key():
    p3d = rand_p3d(4, 7)
    a = project_cam(make_cam(offset=torch.zeros(2)), p3d)
    b = project_cam(make_cam(), p3d)
    assert torch.equal(a, b)


@pytest.mark.parametrize('lead', [(), (2,), (3, 2)])
def test_a_per_frame_offset_is_right_aligned_at_every_rank(lead):
    """THE BUG. offset[None,:] prepends exactly one axis, so it is correct for a (2,) offset and
    expressible for nothing else. Right-alignment must put time at axis -3 -- where this
    function's own docstring already puts it for a (T,4,4) ext -- at every rank."""
    T, N = 4, 7
    p3d = rand_p3d(*lead, T, N)
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 3.0,
                       torch.arange(T, dtype=torch.float32) * -2.0], dim=1)   # (T,2)

    got = project_cam(make_cam(offset=off), p3d)
    assert got.shape == (*lead, T, N, 2), 'rank must be preserved, not grown'

    # Ground truth: each frame projected on its own with that frame's static offset.
    for t in range(T):
        want_t = project_cam(make_cam(offset=off[t]), p3d[..., t, :, :])
        assert torch.allclose(got[..., t, :, :], want_t, atol=1e-9)


def test_a_constant_per_frame_offset_equals_the_static_one():
    """The two paths must agree exactly where they describe the same camera."""
    T, N = 5, 6
    p3d = rand_p3d(T, N)
    off1 = torch.tensor([9.0, 4.0])
    offT = off1[None].repeat(T, 1)
    assert torch.allclose(project_cam(make_cam(offset=offT), p3d),
                          project_cam(make_cam(offset=off1), p3d), atol=1e-9)


def test_a_timeless_point_set_through_a_moving_camera_raises():
    """Without the guard the subtraction GROWS a time axis -- (1,n,2) - (T,1,2) -> (T,n,2) --
    and hands back a plausible tensor of the wrong rank that fails frames away from its cause."""
    off = torch.zeros(4, 2)
    with pytest.raises(ValueError, match='time axis'):
        project_cam(make_cam(offset=off), rand_p3d(7))          # (N,3): no time axis at all


def test_a_mismatched_frame_count_raises():
    off = torch.zeros(4, 2)
    with pytest.raises(ValueError, match='time axis'):
        project_cam(make_cam(offset=off), rand_p3d(9, 7))       # T=9 against a 4-frame offset


def test_a_3d_offset_is_rejected():
    with pytest.raises(ValueError, match=r'\(2,\) or \(T,2\)'):
        project_cam(make_cam(offset=torch.zeros(4, 3, 2)), rand_p3d(4, 3, 7))


def test_project_points_torch_carries_the_per_frame_offset():
    T, N = 4, 6
    p3d = rand_p3d(T, N)
    off = torch.stack([torch.arange(T, dtype=torch.float32),
                       torch.arange(T, dtype=torch.float32) * 2], dim=1)
    got = project_points_torch([make_cam(offset=off), make_cam(offset=off)], p3d)
    assert got.shape == (2, T, N, 2)
    assert torch.allclose(got[0], project_cam(make_cam(offset=off), p3d))


# --------------------------------------------------------------------------------------
# 2. undistort_points  (+ item 20: the unguarded cam['offset'] read)
# --------------------------------------------------------------------------------------

def test_undistort_without_an_offset_key_does_not_raise():
    """project_cam guards the offset ('if offset in cam'); this one used to KeyError."""
    pts = torch.rand(10, 2) * 200
    got = undistort_points(make_cam(), pts)
    assert torch.allclose(got, undistort_points(make_cam(offset=torch.zeros(2)), pts))


def test_undistort_on_a_per_frame_offset_equals_per_frame_static_calls():
    """(T,2)[0] is a VALID index returning frame 0's (x,y), so the whole clip used to be
    undistorted against the first frame's crop -- silently."""
    T, N = 4, 5
    pts = torch.rand(T, N, 2) * 200
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 7.0,
                       torch.arange(T, dtype=torch.float32) * -3.0], dim=1)

    got = undistort_points(make_cam(offset=off), pts)
    assert got.shape == (T, N, 2)
    for t in range(T):
        want = undistort_points(make_cam(offset=off[t]), pts[t])
        assert torch.allclose(got[t], want, atol=1e-6)


def test_undistort_accepts_the_flat_tn_ray_order():
    """points_to_rays flattens as 't n r -> (t n) r'; the offset must expand over the SAME
    order. Getting this wrong displaces each ray by up to the crop's travel."""
    T, N = 4, 5
    pts_tn = torch.rand(T, N, 2) * 200
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 7.0,
                       torch.arange(T, dtype=torch.float32) * -3.0], dim=1)

    flat = undistort_points(make_cam(offset=off), pts_tn.reshape(T * N, 2))
    nested = undistort_points(make_cam(offset=off), pts_tn)
    assert torch.allclose(flat.reshape(T, N, 2), nested, atol=1e-6)


def test_undistort_raises_on_an_unresolvable_layout():
    off = torch.zeros(4, 2)
    with pytest.raises(ValueError, match='does not line up'):
        undistort_points(make_cam(offset=off), torch.rand(7, 2))   # 7 rays, 4 frames


# --------------------------------------------------------------------------------------
# 3. points_to_rays -- the gauge frame
# --------------------------------------------------------------------------------------

def test_the_gauge_frame_anchors_at_frame_zero_for_a_shared_ray():
    """A single crop-centre point against a T-frame offset is the ray-local GAUGE: it must be
    one stable frame for the clip, and frame 0 is the anchor the library already picks."""
    T = 4
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 11.0,
                       torch.arange(T, dtype=torch.float32) * 5.0], dim=1)
    center = torch.tensor([[128.0, 128.0]])

    cam_moving = make_cam(offset=off, T=T, moving_ext=True)
    got = points_to_rays(cam_moving, center, normalize_t=False, ext=cam_moving['ext'][0])
    want = points_to_rays(make_cam(offset=off[0]), center, normalize_t=False)
    assert torch.allclose(got, want, atol=1e-6)


def test_the_gauge_frame_resolves_with_ext_none():
    """A moving CROP leaves the rig STATIC, so the caller passes ext=None -- a check on `ext`
    alone misses it entirely. The arithmetic sign (rays not divisible by frames) must catch it."""
    T = 4
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 11.0,
                       torch.arange(T, dtype=torch.float32) * 5.0], dim=1)
    center = torch.tensor([[128.0, 128.0]])

    got = points_to_rays(make_cam(offset=off), center, normalize_t=False, ext=None)
    want = points_to_rays(make_cam(offset=off[0]), center, normalize_t=False)
    assert torch.allclose(got, want, atol=1e-6)


def test_per_ray_offsets_are_kept_when_rays_divide_by_frames():
    T, N = 4, 6
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 11.0,
                       torch.arange(T, dtype=torch.float32) * 5.0], dim=1)
    pts = torch.rand(T * N, 2) * 200
    got = points_to_rays(make_cam(offset=off), pts, normalize_t=False)
    # Frame 0's rays must match a static frame-0 camera; later frames must NOT.
    want0 = points_to_rays(make_cam(offset=off[0]), pts, normalize_t=False)
    assert torch.allclose(got[:N], want0[:N], atol=1e-6)
    assert not torch.allclose(got[N:2 * N], want0[N:2 * N], atol=1e-3)


# --------------------------------------------------------------------------------------
# 4. get_camera_scale -- offset-invariant
# --------------------------------------------------------------------------------------

def test_get_camera_scale_is_unchanged_by_a_per_frame_offset():
    """It returns a projection SENSITIVITY -- a Jacobian -- and a constant image-plane
    translation has zero derivative, so collapsing the offset is EXACT. The one caller that
    matters passes a query anchor with no time axis, which would otherwise hit project_cam's
    guard and stop the run."""
    T = 4
    p = rand_p3d(2, 9)                                  # (B,N,3): no time axis
    off = torch.stack([torch.arange(T, dtype=torch.float32) * 11.0,
                       torch.arange(T, dtype=torch.float32) * 5.0], dim=1)

    moving = get_camera_scale([make_cam(offset=off)], p)
    static = get_camera_scale([make_cam(offset=off[0])], p)
    assert torch.allclose(moving, static, atol=1e-6, equal_nan=True)


def test_get_camera_scale_static_path_is_unchanged():
    p = rand_p3d(2, 9)
    a = get_camera_scale([make_cam(offset=torch.tensor([3.0, 4.0]))], p)
    b = get_camera_scale([make_cam(offset=torch.tensor([3.0, 4.0]))], p)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------------------
# 5. End to end: the moving crop projects where the pixels were cut
# --------------------------------------------------------------------------------------

def test_the_moving_crop_projects_where_the_pixels_were_cut():
    """The whole point of the family: a crop that follows the subject per frame is a per-frame
    offset and nothing else, so projecting into the cropped camera must equal projecting into
    the full camera and subtracting that frame's crop origin."""
    T, N = 6, 8
    p3d = rand_p3d(T, N)

    full = make_cam()
    p2d_full = project_cam(full, p3d)                       # (T,N,2) in full-frame pixels

    # A crop box that tracks the points, one per frame.
    origins = torch.stack([p2d_full[t].min(dim=0).values.floor() - 10 for t in range(T)])
    cropped = make_cam(offset=origins)
    cropped['size'] = torch.tensor([96, 96], dtype=torch.int32)

    p2d_crop = project_cam(cropped, p3d)
    assert torch.allclose(p2d_crop, p2d_full - origins[:, None, :], atol=1e-9)
