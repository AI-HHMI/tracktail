"""Figure 1 (cover) qualitative demo on rat7m: dense rat point cloud -> Tracktail -> trajectory panels.

Stages (all run by default; `--stages` picks a subset, later stages reuse earlier outputs in --out-dir):
  cloud   initial 3D point cloud of the rat at the query frame, from VGGT depth + our calibration
            1. masks:  median-background difference -> SAM ViT-L (box + 8 points) per camera
            2. hull:   voxel visual hull of the masks; its surface points are the initial anchors
            3. depth:  VGGT-1B on square crops around each mask (all cameras jointly)
            4. align:  each depth map -> camera z by a robust affine+plane fit to the visible anchors
            5. fuse:   back-project with OUR K/dist/extrinsics; keep points inside all masks, not in front
                       of any other view's depth (25 mm) and within 12 mm of >=2 other cameras' depth
            6. repeat 4-5 three times with the fused cloud as anchors; voxel-downsample to <= 6000 points
  track   track the cloud with Tracktail (anonymous/tracktail-static @ 2026-10-12) as 3D queries
  figure  cover panels: camera1 + 3 small views with fading trails, 5-view row, camera frustums, input grid,
          3D renders of the cloud at several times (row + time-lapse scenes; `_glow` = dark hero version
          with tracks between poses (white bg by default; bg='#05060a' for the dark look), display-smoothed with a 2-frame Gaussian, and the query pose highlighted)

Extra dependencies (not in the pixi env): VGGT and segment-anything, see scripts/cover/README.md, e.g.
    pip install --no-deps --target <dir> \
        "vggt @ git+https://github.com/facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce" \
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything@dca509fe793f601edb92606367a655c15ac00fdf"
and pass `--extra-path <dir>`. The SAM ViT-L checkpoint is downloaded to the torch hub cache if
--sam-checkpoint is not given.

Example (the paper figure; <rat7m> = your converted Rat7M test split):
    pixi run python scripts/cover/cover_rat7m_demo.py \
        --trial <rat7m>/test/s5-d1/101500 \
        --frame 670 --out-dir out/cover_rat7m --extra-path <dir>
"""
import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42      # TrueType fonts -> text stays editable in Illustrator/Inkscape
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.ndimage import binary_erosion, map_coordinates, minimum_filter  # noqa: E402

from tracktail.inference.inference_utils import load_camera_group_from_metadata  # noqa: E402

SAM_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth'
RV = 518                                                    # VGGT crop resolution
sig = lambda x: 1 / (1 + np.exp(-x))


# ============================================================================ geometry / io
class Geom:
    """OpenCV projection with the trial's calibration (world -> camera extrinsics, mm)."""
    def __init__(self, metadata_path):
        cams = load_camera_group_from_metadata(metadata_path)
        self.names = [c['name'] for c in cams]
        self.K = [c['mat'].double().numpy() for c in cams]
        self.D = [c['dist'].double().numpy() for c in cams]
        self.E = [c['ext'].double().numpy() for c in cams]
        self.wh = [(int(c['size'][0]), int(c['size'][1])) for c in cams]
        self.C = len(cams)

    def project(self, X):
        out = []
        for K, D, E in zip(self.K, self.D, self.E):
            p, _ = cv2.projectPoints(X.reshape(-1, 1, 3).astype(np.float64), cv2.Rodrigues(E[:3, :3])[0], E[:3, 3], K, D)
            out.append(p[:, 0])
        return np.stack(out)

    def depth(self, X):
        return np.stack([(X @ E[:3, :3].T + E[:3, 3])[:, 2] for E in self.E])

    def undistort(self, i, pts):
        p = cv2.undistortPointsIter(pts.reshape(-1, 1, 2).astype(np.float64), self.K[i], self.D[i], None, None,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9))
        return p[:, 0]

    def triangulate(self, pts):
        """pts (C,N,2) -> (N,3), linear DLT on undistorted points."""
        N = pts.shape[1]; A = np.zeros((N, 2 * self.C, 4))
        for i in range(self.C):
            u = self.undistort(i, pts[i]); P = self.E[i][:3]
            A[:, 2 * i] = u[:, :1] * P[2] - P[0]; A[:, 2 * i + 1] = u[:, 1:] * P[2] - P[1]
        X = np.linalg.svd(A)[2][:, -1]
        return X[:, :3] / X[:, 3:]


def read_frames(path, f0, n=1):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, f0); out = []
    for _ in range(n):
        ok, im = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def voxel_down(X, *attrs, voxel=2.0):
    k = np.floor(X / voxel).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
    out = []
    for a in (X,) + attrs:
        m = np.zeros((len(cnt),) + a.shape[1:]); np.add.at(m, inv, a.astype(np.float64))
        out.append((m / cnt.reshape((-1,) + (1,) * (a.ndim - 1))).astype(a.dtype if a is not X else np.float64))
    return tuple(out)


def in_mask(G, X, masks):
    """(C,N) True where X projects inside the camera's mask (False if out of image)."""
    P = np.round(G.project(X)).astype(int); out = np.zeros(P.shape[:2], bool)
    for c, m in enumerate(masks):
        H, W = m.shape; ok = (P[c, :, 0] >= 0) & (P[c, :, 0] < W) & (P[c, :, 1] >= 0) & (P[c, :, 1] < H)
        out[c, ok] = m[P[c, ok, 1], P[c, ok, 0]]
    return out


def in_image(P, H, W):
    u, v = np.round(P[..., 0]), np.round(P[..., 1])
    return (u >= 0) & (u < W) & (v >= 0) & (v < H)


# ============================================================================ stage 1: cloud
def fg_mask(img, bg, thr=30, dilate=15):
    m = (np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(-1) > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        a = st[1:, cv2.CC_STAT_AREA]; m = np.isin(lab, 1 + np.nonzero(a >= max(500, 0.1 * a.max()))[0]).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.dilate(m, np.ones((dilate, dilate), np.uint8)).astype(bool)


def median_background(vid, n=30):
    cap = cv2.VideoCapture(vid); T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for t in np.linspace(0, T - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t)); ok, im = cap.read()
        if ok: fr.append(im[..., ::-1])
    return np.median(np.stack(fr), 0).astype(np.uint8)


def sam_masks(imgs, vids, ckpt):
    from segment_anything import sam_model_registry, SamPredictor
    pred = SamPredictor(sam_model_registry['vit_l'](checkpoint=ckpt).cuda().eval())
    out = []
    for img, vid in zip(imgs, vids):
        m = cv2.erode(fg_mask(img, median_background(vid)).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        H, W = m.shape; ys, xs = np.nonzero(m)
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        box = np.array([max(0, xs.min() - w / 4), max(0, ys.min() - h / 4), min(W - 1, xs.max() + w / 4), min(H - 1, ys.max() + h / 4)])
        m2 = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        ys, xs = np.nonzero(m2 if m2.sum() > 100 else m); pts = np.stack([xs, ys], 1)
        sel = [len(pts) // 2]; d = np.linalg.norm(pts - pts[sel[0]], axis=1)
        for _ in range(7):                                         # 8 farthest-point prompts
            sel.append(int(d.argmax())); d = np.minimum(d, np.linalg.norm(pts - pts[sel[-1]], axis=1))
        pred.set_image(img)
        ms, _, _ = pred.predict(point_coords=pts[sel].astype(float), point_labels=np.ones(len(sel)), box=box, multimask_output=True)
        cover = np.array([(k & m).sum() / m.sum() for k in ms]); area = np.array([k.sum() / m.sum() for k in ms])
        k = ms[int(np.argmin(np.abs(np.log(area / 1.35)) + 5 * (cover < 0.65)))].astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(k)           # drop small blobs
        if n > 1: k = np.isin(lab, 1 + np.nonzero(st[1:, cv2.CC_STAT_AREA] >= 0.05 * st[1:, cv2.CC_STAT_AREA].max())[0])
        out.append(k.astype(bool))
    del pred; torch.cuda.empty_cache()
    return out


def crop_boxes(masks, pad=1.2, min_side=256):
    boxes = []
    for m in masks:
        H, W = m.shape; ys, xs = np.nonzero(m)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        s = int(min(max(xs.max() - xs.min(), ys.max() - ys.min(), min_side / pad) * pad, H))
        boxes.append((int(np.clip(cx - s / 2, 0, W - s)), int(np.clip(cy - s / 2, 0, H - s)), s))
    return boxes


def to_crop(p, box, R): x0, y0, s = box; return (p - [x0, y0]) * R / s + 0.5 * R / s - 0.5
def to_full(q, box, R): x0, y0, s = box; return (q + 0.5) * s / R - 0.5 + np.array([x0, y0])


def hull_surface(G, masks, half=250.0, vox=3.0):
    cen = np.array([[np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()] for m in masks])
    ctr = G.triangulate(cen[:, None])[0]
    ax = np.arange(-half, half, vox)
    V = np.stack(np.meshgrid(ctr[0] + ax, ctr[1] + ax, ctr[2] + ax, indexing='ij'), -1).reshape(-1, 3)
    occ = np.all(in_mask(G, V, masks), 0).reshape((len(ax),) * 3)
    return V[(occ & ~binary_erosion(occ)).ravel()]


def vggt_depth(imgs, boxes):
    from vggt.models.vggt import VGGT
    net = VGGT.from_pretrained('facebook/VGGT-1B').cuda().eval()
    crops = [cv2.resize(im[y:y + s, x:x + s], (RV, RV), interpolation=cv2.INTER_AREA) for im, (x, y, s) in zip(imgs, boxes)]
    x = torch.from_numpy(np.stack(crops)).float().permute(0, 3, 1, 2)[None].cuda() / 255
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        d = net(x)['depth'][0, ..., 0].float().cpu().numpy()
    del net; torch.cuda.empty_cache()
    return d


def zbuf_visible(G, X, H, W, splat=6, tol=10.0):
    P, Z = G.project(X), G.depth(X); vis = np.zeros(Z.shape, bool)
    for c in range(G.C):
        u, v = np.round(P[c, :, 0]).astype(int), np.round(P[c, :, 1]).astype(int)
        idx = np.nonzero((u >= 0) & (u < W) & (v >= 0) & (v < H) & (Z[c] > 0))[0]
        zb = np.full((H, W), np.inf); np.minimum.at(zb, (v[idx], u[idx]), Z[c, idx])
        zb = cv2.erode(zb.astype(np.float32), np.ones((2 * splat + 1,) * 2, np.uint8))
        vis[c, idx] = Z[c, idx] <= zb[v[idx], u[idx]] + tol
    return vis, P, Z


def align(D, R, q, z, thr=8.0, iters=500, seed=0):
    """Robust fit z ~ a*D + b + c*u + d*v of depth map D (R,R) to anchors (crop px q, camera depth z)."""
    d = map_coordinates(D, [q[:, 1], q[:, 0]], order=1)
    rng = np.random.default_rng(seed); best = (0, None)
    for _ in range(iters):                                          # RANSAC on the affine part
        i = rng.choice(len(d), 2, replace=False)
        if abs(d[i[0]] - d[i[1]]) < 1e-6: continue
        s = (z[i[0]] - z[i[1]]) / (d[i[0]] - d[i[1]]); b = z[i[0]] - s * d[i[0]]
        if s > 0 and (n := (np.abs(s * d + b - z) < thr).sum()) > best[0]: best = (n, (s, b))
    s, b = best[1]; inl = np.abs(s * d + b - z) < thr
    A = np.stack([d, np.ones_like(d), q[:, 0] / R, q[:, 1] / R], 1)
    for _ in range(3):
        coef = np.linalg.lstsq(A[inl], z[inl], rcond=None)[0]; inl = np.abs(A @ coef - z) < thr
    vv, uu = np.mgrid[0:R, 0:R]
    return coef[0] * D + coef[1] + coef[2] * uu / R + coef[3] * vv / R


def build_views(G, anchors, D, boxes, core):
    H, W = core[0].shape; vis, P, Z = zbuf_visible(G, anchors, H, W); views = []
    for c in range(G.C):
        pi = np.round(P[c]).astype(int)
        sel = vis[c] & (pi[:, 0] >= 0) & (pi[:, 0] < W) & (pi[:, 1] >= 0) & (pi[:, 1] < H)
        sel[sel] &= core[c][pi[sel, 1], pi[sel, 0]]
        q = to_crop(P[c, sel], boxes[c], RV); ok = (q >= 0).all(1) & (q < RV - 1).all(1)
        views.append(dict(c=c, box=boxes[c], Z=align(D[c], RV, q[ok], Z[c, sel][ok])))
    return views


def fuse(G, views, imgs, masks, core, min_agree, tol=12.0, ftol=25.0, stride_px=2.0):
    H, W = core[0].shape; Xs, rgbs, src = [], [], []
    for v in views:
        st = max(1, int(round(stride_px * RV / v['box'][2])))
        vv, uu = np.mgrid[0:RV:st, 0:RV:st]
        p = to_full(np.stack([uu.ravel(), vv.ravel()], 1).astype(float), v['box'], RV); pi = np.round(p).astype(int)
        ok = (pi[:, 0] >= 0) & (pi[:, 0] < W) & (pi[:, 1] >= 0) & (pi[:, 1] < H)
        ok[ok] &= core[v['c']][pi[ok, 1], pi[ok, 0]]
        z = v['Z'][vv.ravel(), uu.ravel()]; ok &= np.isfinite(z) & (z > 0)
        n = G.undistort(v['c'], p[ok]); E = G.E[v['c']]
        Xs.append((np.c_[n * z[ok, None], z[ok]] - E[:3, 3]) @ E[:3, :3])
        rgbs.append(imgs[v['c']][pi[ok, 1], pi[ok, 0]]); src.append(np.full(ok.sum(), v['c']))
    X, rgb, src = np.concatenate(Xs), np.concatenate(rgbs), np.concatenate(src)
    P, Zc = G.project(X), G.depth(X)
    inside = np.all(in_mask(G, X, masks) | ~in_image(P, H, W), 0)
    core_hit = in_mask(G, X, core); front = np.zeros(len(X), int); agree = np.zeros((G.C, len(X)), bool)
    for v in views:
        j = v['c']; q = to_crop(P[j], v['box'], RV)
        ok = (q >= 0).all(1) & (q <= RV - 1).all(1) & core_hit[j] & (src != j)
        dj = np.full(len(X), np.nan); dj[ok] = map_coordinates(v['Z'], [q[ok, 1], q[ok, 0]], order=1)
        agree[j] |= np.abs(Zc[j] - dj) < tol; front += (Zc[j] - dj) < -ftol
    keep = inside & (front == 0) & (agree.sum(0) >= min_agree)
    return X[keep], rgb[keep]


def build_cloud(trial, frame, sam_ckpt, n_iter=3, max_pts=6000):
    G = Geom(os.path.join(trial, 'metadata.yaml'))
    vids = [os.path.join(trial, 'vid', n + '.mp4') for n in G.names]
    imgs = [read_frames(v, frame)[0] for v in vids]
    sm = sam_masks(imgs, vids, sam_ckpt)
    masks = [cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for m in sm]
    core = [cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for m in sm]
    boxes = crop_boxes(sm)
    D = vggt_depth(imgs, boxes)
    anchors = hull_surface(G, masks)
    for _ in range(n_iter):
        views = build_views(G, anchors, D, boxes, core)
        anchors = voxel_down(fuse(G, views, imgs, masks, core, min_agree=1)[0], voxel=2.0)[0]
        X, rgb = fuse(G, views, imgs, masks, core, min_agree=2)
    vox = 2.0
    while len(voxel_down(X, voxel=vox)[0]) > max_pts: vox *= 1.1
    return voxel_down(X, rgb, voxel=vox)


# ============================================================================ stage 2: track
def track_cloud(trial, frame, xyz, n_frames, repo='anonymous/tracktail-static', revision='2026-10-12'):
    from tracktail.model.tracker_encoder import TrackerEncoder
    from tracktail.inference.inference_utils import run_tracker_encoder_on_videos
    model = TrackerEncoder.from_pretrained(repo, revision=revision).cuda().eval()
    cams = load_camera_group_from_metadata(os.path.join(trial, 'metadata.yaml'))
    vids = [os.path.join(trial, 'vid', c['name'] + '.mp4') for c in cams]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        res = run_tracker_encoder_on_videos(model, vids, cams, torch.as_tensor(xyz, dtype=torch.float32),
                                            start_frame=frame, n_frames=n_frames, max_kpts=1000)
    f = lambda k: res[k].float().cpu().numpy()
    return dict(coords=f('coords_pred')[0], vis=f('vis_pred')[0, ..., 0], vis_cams=f('vis_pred_2d')[:, 0],
                frames=res['frame_numbers'].numpy())


# ============================================================================ stage 3: figure
def rainbow(X0):
    """HSV rainbow along the cloud's principal axis (head -> tail)."""
    c = X0 - X0.mean(0); s = c @ np.linalg.svd(c, full_matrices=False)[2][0]
    r = np.argsort(np.argsort(s)) / max(len(s) - 1, 1)
    hsv = np.stack([(r * 170).astype(np.uint8), np.full(len(r), 255, np.uint8), np.full(len(r), 255, np.uint8)], -1)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0] / 255.0


def fps(X, k, seed=0):
    idx = [int(np.random.default_rng(seed).integers(len(X)))]; d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(k - 1):
        idx.append(int(np.argmax(d))); d = np.minimum(d, np.linalg.norm(X - X[idx[-1]], axis=1))
    return np.array(idx)


def zbuf_front(p2d, z, wh, cell=5, tol=12.0):
    """(N,) True if the point is within tol of the front-most depth around its pixel."""
    W, H = wh; ok = np.isfinite(p2d).all(-1) & (z > 0)
    gw, gh = W // cell + 2, H // cell + 2; buf = np.full((gh, gw), np.inf)
    ij = np.clip((np.nan_to_num(p2d) // cell).astype(int), 0, [gw - 1, gh - 1])
    np.minimum.at(buf, (ij[ok, 1], ij[ok, 0]), z[ok])
    buf = minimum_filter(buf, size=3, mode='nearest')
    return ok & (z <= buf[ij[:, 1], ij[:, 0]] + tol)


def crop_box(pts, w, h, pad=60, ar=4 / 3):
    pts = pts[np.isfinite(pts).all(1)]
    (x0, y0), (x1, y1) = pts.min(0) - pad, pts.max(0) + pad
    if (x1 - x0) / (y1 - y0) < ar: cx = (x0 + x1) / 2; bw = (y1 - y0) * ar; x0, x1 = cx - bw / 2, cx + bw / 2
    else: cy = (y0 + y1) / 2; bh = (x1 - x0) / ar; y0, y1 = cy - bh / 2, cy + bh / 2
    bw, bh = min(x1 - x0, w), min(y1 - y0, h)
    x0, y0 = np.clip(x0, 0, w - bw), np.clip(y0, 0, h - bh)
    return x0, y0, x0 + bw, y0 + bh


class Panels:
    def __init__(self, trial, tr, n_trails=800, trail_px=1.2, trail_len=30, vis_thr=0.5):
        self.G = G = Geom(os.path.join(trial, 'metadata.yaml'))
        self.meta = yaml.safe_load(open(os.path.join(trial, 'metadata.yaml')))
        self.vids = [os.path.join(trial, 'vid', n + '.mp4') for n in G.names]
        X, vis, vc, self.frames = tr['coords'], sig(tr['vis']), sig(tr['vis_cams']), tr['frames']
        med = np.nanmedian(np.linalg.norm(X[:60] - X[:1], axis=2), 0)     # drop static low bedding points
        keep = ~((X[0, :, 2] < 20.0) & (med < 3.0))
        self.X, self.vis, vc = X[:, keep], vis[:, keep], vc[:, :, keep]
        T, N = self.X.shape[:2]
        self.colors, self.trail_idx = rainbow(self.X[0]), fps(self.X[0], min(n_trails, N))
        self.trail_px, self.trail_len, self.vis_thr = trail_px, trail_len, vis_thr
        flat = self.X.reshape(-1, 3)
        self.p2d = G.project(flat).reshape(G.C, T, N, 2); self.z = G.depth(flat).reshape(G.C, T, N)
        self.good = [(self.vis >= vis_thr) & (vc[c] >= vis_thr) &
                     np.stack([zbuf_front(self.p2d[c, t], self.z[c, t], G.wh[c]) for t in range(T)]) for c in range(G.C)]
        self._img = {}

    def img(self, c, t):
        if (c, t) not in self._img: self._img[c, t] = read_frames(self.vids[c], int(self.frames[t]))[0]
        return self._img[c, t]

    def draw(self, ax, c, t, dense=False, ar=4 / 3):
        im = self.img(c, t); h, w = im.shape[:2]
        x0, y0, x1, y1 = crop_box(self.p2d[c, t], w, h, ar=ar)
        xi0, yi0, xi1, yi1 = int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))   # embed only the crop
        ax.imshow(im[yi0:yi1, xi0:xi1], extent=(xi0 - 0.5, xi1 - 0.5, yi1 - 0.5, yi0 - 0.5), interpolation='lanczos')
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ppx = ax.get_position().width * ax.figure.get_figwidth() * 72 / (x1 - x0)
        good, pts = self.good[c], self.p2d[c]
        if dense:
            ks = np.where(good[t])[0]; ks = ks[np.argsort(-self.z[c, t, ks])]
            ax.scatter(pts[t, ks, 0], pts[t, ks, 1], s=(1.6 * ppx) ** 2, c=self.colors[ks], edgecolors='none', zorder=3)
        else:
            t0 = max(0, t - self.trail_len)
            for k in self.trail_idx:
                segs, al = [], []
                for s in range(t0, t):
                    if good[s, k] and good[s + 1, k]:
                        segs.append(pts[s:s + 2, k]); al.append((s - t0 + 1) / max(t - t0, 1))
                if segs:
                    ax.add_collection(LineCollection(segs, colors=[(*self.colors[k], a) for a in al],
                                                     linewidths=self.trail_px * ppx, capstyle='round', zorder=2))
        ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]

    def tracks_main(self, out, t, main='camera1', small=('camera2', 'camera3', 'camera4'), dense=False, W=9, g=0.01, ar=4 / 3):
        sw = (1 - 2 * g) / len(small); H = 1 / ar + g + sw / ar
        fig = plt.figure(figsize=(W, W * H))
        self.draw(fig.add_axes([0, (g + sw / ar) / H, 1, (1 / ar) / H]), self.G.names.index(main), t, dense)
        for i, cam in enumerate(small):
            self.draw(fig.add_axes([i * (sw + g), 0, sw, (sw / ar) / H]), self.G.names.index(cam), t, dense)
        save(fig, out)

    def row(self, out, t, cams=('camera1', 'camera2', 'camera4', 'camera5', 'camera3'), dense=False, W=12, g=0.005, ar=4 / 3):
        pw = (1 - (len(cams) - 1) * g) / len(cams)
        fig = plt.figure(figsize=(W, W * pw / ar))
        for i, cam in enumerate(cams):
            self.draw(fig.add_axes([i * (pw + g), 0, pw, 1]), self.G.names.index(cam), t, dense)
        save(fig, out)

    def input_grid(self, out, views=('camera1', 'camera2', 'camera3', 'camera4'), dt=(0, 5, 10, 15, 20, 25), fs=18):
        fig, axs = plt.subplots(len(views), len(dt), figsize=(12, 6), gridspec_kw={'hspace': 0, 'wspace': 0})
        for i, v in enumerate(views):
            c = self.G.names.index(v)
            for j, t in enumerate(dt):
                ax = axs[i, j]; ax.imshow(self.img(c, t), aspect='auto')
                if j == 0:
                    ks = np.where(self.good[c][0])[0]
                    ax.scatter(self.p2d[c, 0, ks, 0], self.p2d[c, 0, ks, 1], s=0.3, c=self.colors[ks], edgecolors='none')
                    ax.set_xlim(0, self.G.wh[c][0]); ax.set_ylim(self.G.wh[c][1], 0)
                ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
                if i == len(views) - 1: ax.set_xlabel(f't={t}', fontsize=fs)
                if j == 0: ax.set_ylabel(f'View {i + 1}', fontsize=fs)
        fig.subplots_adjust(hspace=0, wspace=0, top=1, bottom=0, left=0, right=1)
        save(fig, out)

    def frustums(self, out, t=0, azim=-60, elev=25, depth=600.0, canvas_w=2400, margin=0.04, ss=2, up=(0, 0, 1)):
        m, planes = self.meta, []
        for c, n in enumerate(self.G.names):
            E = np.asarray(m['extrinsic_matrices'][n], float); R, tt = E[:3, :3], E[:3, 3]
            K = np.asarray(m['intrinsic_matrices'][n], float)
            im = cv2.undistort(self.img(c, t), K, np.asarray(m['distortion_matrices'][n], float)); h, w = im.shape[:2]
            cpx = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
            Xc = np.c_[(cpx - K[:2, 2]) / np.diag(K)[:2], np.ones(4)] * depth
            planes.append(dict(img=im, cpx=cpx, cw=(Xc - tt) @ R, center=-R.T @ tt))
        ok = np.isfinite(self.X[t]).all(1) & (self.vis[t] >= self.vis_thr); X = self.X[t, ok]
        allp = np.concatenate([p['cw'] for p in planes] + [np.array([p['center'] for p in planes]), X])
        target = allp.mean(0); rad = np.linalg.norm(allp - target, axis=1).max()
        a, e = np.deg2rad(azim), np.deg2rad(elev)
        eye = target + 3 * rad * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        zc = (target - eye) / np.linalg.norm(target - eye); xc = np.cross(zc, up); xc /= np.linalg.norm(xc)
        Rv = np.stack([xc, np.cross(zc, xc), zc])
        nrm = lambda P: ((P - eye) @ Rv.T)[:, :2] / ((P - eye) @ Rv.T)[:, 2:3]
        q = nrm(allp); lo, hi = q.min(0), q.max(0)
        f = canvas_w * ss * (1 - 2 * margin) / (hi - lo)[0]; Wc = canvas_w * ss; Hc = int(round((hi - lo)[1] * f + 2 * margin * Wc))
        c0 = np.array([Wc / 2, Hc / 2]) - f * (lo + hi) / 2; proj = lambda P: nrm(P) * f + c0
        canvas = np.ones((Hc, Wc, 3), np.float32)
        for i in np.argsort([(p['cw'].mean(0) - eye) @ zc for p in planes])[::-1]:
            p = planes[i]; Hm = cv2.getPerspectiveTransform(p['cpx'].astype(np.float32), proj(p['cw']).astype(np.float32))
            wp = cv2.warpPerspective(p['img'].astype(np.float32) / 255, Hm, (Wc, Hc))
            mk = cv2.warpPerspective(np.ones(p['img'].shape[:2], np.float32), Hm, (Wc, Hc))[..., None]
            canvas = canvas * (1 - mk) + wp * mk
        canvas = cv2.resize(canvas, (Wc // ss, Hc // ss), interpolation=cv2.INTER_AREA)
        fig = plt.figure(figsize=(canvas.shape[1] / 200, canvas.shape[0] / 200))
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(canvas, interpolation='lanczos'); ax.set_axis_off()
        for p in planes:
            C = proj(p['center'][None])[0] / ss
            for cr in proj(p['cw']) / ss: ax.plot([C[0], cr[0]], [C[1], cr[1]], color='0.5', lw=0.8)
            ax.scatter(*C, color='0.5', s=25, zorder=4)
        o = np.argsort(-((X - eye) @ zc)); P = proj(X) / ss
        ax.scatter(P[o, 0], P[o, 1], s=1.5, c=self.colors[ok][o], edgecolors='none', zorder=5)
        ax.set_xlim(0, canvas.shape[1]); ax.set_ylim(canvas.shape[0], 0)
        save(fig, out, dpi=200)

    # -- 3D renderings of the tracked cloud at several times (virtual perspective camera, floor + shadows)
    def _view(self, target, dist, azim, elev, up=(0, 0, 1)):
        a, e = np.deg2rad(azim), np.deg2rad(elev)
        eye = target + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        zc = (target - eye) / np.linalg.norm(target - eye); xc = np.cross(zc, up); xc /= np.linalg.norm(xc)
        Rv = np.stack([xc, np.cross(zc, xc), zc])
        return lambda P: (((P - eye) @ Rv.T)[:, :2] / ((P - eye) @ Rv.T)[:, 2:3], ((P - eye) @ Rv.T)[:, 2])

    def _render3d(self, ax, layers, view, lim, grid_ctr, grid_r=300, step=50, pt=6.0):
        """layers: list of (X (N,3), colors (N,3), alpha). Draws floor grid, soft shadows, then points far->near."""
        for g in np.arange(-grid_r, grid_r + 1, step):                     # floor grid at z=0
            for P in (np.array([[grid_ctr[0] + g, grid_ctr[1] - grid_r, 0], [grid_ctr[0] + g, grid_ctr[1] + grid_r, 0]]),
                      np.array([[grid_ctr[0] - grid_r, grid_ctr[1] + g, 0], [grid_ctr[0] + grid_r, grid_ctr[1] + g, 0]])):
                q = view(np.linspace(P[0], P[1], 50))[0]; ax.plot(q[:, 0], -q[:, 1], color='0.85', lw=0.6, zorder=0)
        for X, col, al in layers:                                          # shadows (vertical projection)
            S = X.copy(); S[:, 2] = 0; q = view(S)[0]
            ax.scatter(q[:, 0], -q[:, 1], s=pt * 1.5, c='k', alpha=0.012 * al, edgecolors='none', zorder=1)
        allq, allc, alld = [], [], []
        for X, col, al in layers:
            q, d = view(X)
            allq.append(q); alld.append(d); allc.append(np.c_[col, np.full(len(col), al)])
        q, d, c = np.concatenate(allq), np.concatenate(alld), np.concatenate(allc)
        shade = 1 - 0.35 * (d - d.min()) / max(np.ptp(d), 1e-6)               # darker = farther
        c[:, :3] *= shade[:, None]
        o = np.argsort(-d)
        ax.scatter(q[o, 0], -q[o, 1], s=pt, c=c[o], edgecolors='none', zorder=2)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect('equal'); ax.set_axis_off()

    def clouds3d(self, out, ts=(0, 60, 120, 180, 240, 300), azim=-70, elev=22, W=14, pt=5.0):
        """Row of 3D renders of the cloud at several times, one fixed camera for all panels."""
        ts = [t for t in ts if t < len(self.frames)]
        ok = [self.vis[t] >= self.vis_thr for t in ts]
        cen = np.nanmean(np.concatenate([self.X[t][k] for t, k in zip(ts, ok)]), 0)
        span = np.nanmax(np.ptp(np.concatenate([self.X[t][k] for t, k in zip(ts, ok)]), 0))
        view = self._view(np.r_[cen[:2], 60.0], 4 * span, azim, elev)
        qs = [view(self.X[t][k])[0] for t, k in zip(ts, ok)]
        fig, axs = plt.subplots(1, len(ts), figsize=(W, W / len(ts) * 1.1), gridspec_kw={'wspace': 0})
        for ax, t, k, q in zip(np.atleast_1d(axs), ts, ok, qs):
            vt = self._view(np.r_[np.nanmean(self.X[t][k][:, :2], 0), 60.0], 4 * span, azim, elev)   # follow the rat
            qq = vt(self.X[t][k])[0]; cc = qq.mean(0); rr = 0.75 * np.ptp(np.concatenate(qs), 0).max() / 2.2
            self._render3d(ax, [(self.X[t][k], self.colors[k], 1.0)], vt, (cc[0] - rr, cc[0] + rr, -cc[1] - rr * 1.1, -cc[1] + rr * 1.1),
                           np.nanmean(self.X[t][k][:, :2], 0), grid_r=200, pt=pt)
            ax.set_title(f't = {t / 120:.2f} s', fontsize=12, family='serif')
        save(fig, out)

    def cloud_timelapse(self, out, ts=(0, 110, 200, 300), azim=-70, elev=35, W=9, pt=3.0):
        """All times in one 3D scene; earlier poses fade out."""
        ts = [t for t in ts if t < len(self.frames)]
        layers = [(self.X[t][self.vis[t] >= self.vis_thr], self.colors[self.vis[t] >= self.vis_thr],
                   0.15 + 0.85 * i / max(len(ts) - 1, 1)) for i, t in enumerate(ts)]
        allX = np.concatenate([l[0] for l in layers]); cen = allX.mean(0); span = np.ptp(allX, 0).max()
        view = self._view(np.r_[cen[:2], 40.0], 2.5 * span, azim, elev)
        q = view(allX)[0]; lo, hi = q.min(0), q.max(0); pad = 0.08 * (hi - lo).max()
        fig = plt.figure(figsize=(W, W * (hi - lo)[1] / (hi - lo)[0] + 0.3)); ax = fig.add_axes([0, 0, 1, 1])
        self._render3d(ax, layers, view, (lo[0] - pad, hi[0] + pad, -hi[1] - pad, -lo[1] + pad), cen[:2],
                       grid_r=int(span * 0.8), pt=pt)
        save(fig, out)

    def cloud_timelapse_tracks(self, out, ts=(0, 110, 200, 300), n_tracks=250, azim=-70, elev=35, W=9, pt=3.0,
                               later='color', later_alpha=0.45, track_alpha=0.55, track_lw=0.7, label=True):
        """Time-lapse of poses connected by 3D point tracks; the initial query pose (ts[0]) is highlighted.
        later: 'color' (faded rainbow) or 'gray' for the non-query poses."""
        ts = [t for t in ts if t < len(self.frames)]
        okv = [self.vis[t] >= self.vis_thr for t in ts]
        allX = np.concatenate([self.X[t][k] for t, k in zip(ts, okv)]); cen = allX.mean(0); span = np.ptp(allX, 0).max()
        view = self._view(np.r_[cen[:2], 40.0], 2.5 * span, azim, elev)
        q = view(allX)[0]; lo, hi = q.min(0), q.max(0); pad = 0.08 * (hi - lo).max()
        fig = plt.figure(figsize=(W, W * (hi - lo)[1] / (hi - lo)[0] + 0.3)); ax = fig.add_axes([0, 0, 1, 1])
        # floor + shadows of every pose (reuse renderer with no points)
        self._render3d(ax, [(self.X[t][k], self.colors[k], 1.0) for t, k in zip(ts, okv)], view,
                       (lo[0] - pad, hi[0] + pad, -hi[1] - pad, -lo[1] + pad), cen[:2], grid_r=int(span * 0.8), pt=0)
        # tracks: evenly spread points, full path from the query frame to the last pose
        sel = self.trail_idx[fps(self.X[ts[0]][self.trail_idx], min(n_tracks, len(self.trail_idx)))]
        P = self.X[ts[0]:ts[-1] + 1][:, sel]                                   # (T,K,3)
        good = self.vis[ts[0]:ts[-1] + 1][:, sel] >= self.vis_thr
        qs, ds = view(P.reshape(-1, 3)); qs = qs.reshape(P.shape[0], -1, 2); ds = ds.reshape(P.shape[0], -1)
        segs = np.stack([qs[:-1], qs[1:]], 2) * [1, -1]                         # (T-1,K,2,2)
        m = good[:-1] & good[1:]
        col = np.broadcast_to(self.colors[sel][None], m.shape + (3,))
        o = np.argsort(-ds[:-1][m])
        ax.add_collection(LineCollection(segs[m][o], colors=np.c_[col[m][o], np.full(m.sum(), track_alpha)],
                                         linewidths=track_lw, zorder=1.5, capstyle='round'))
        # poses: later ones faded (or gray), query pose on top with a dark outline
        for t, k in list(zip(ts, okv))[1:]:
            qq, dd = view(self.X[t][k]); o = np.argsort(-dd)
            c = np.c_[self.colors[k], np.full(k.sum(), later_alpha)] if later == 'color' else (0.55, 0.55, 0.55, later_alpha)
            ax.scatter(qq[o, 0], -qq[o, 1], s=pt, c=c[o] if later == 'color' else [c], edgecolors='none', zorder=2)
        k = okv[0]; qq, dd = view(self.X[ts[0]][k]); o = np.argsort(-dd)
        ax.scatter(qq[o, 0], -qq[o, 1], s=pt * 4.0, c='k', edgecolors='none', zorder=3)          # outline
        ax.scatter(qq[o, 0], -qq[o, 1], s=pt * 1.6, c=self.colors[k][o], edgecolors='none', zorder=4)
        if label:
            top = qq[np.argmin(qq[:, 1])]
            ax.annotate('query pose (t = 0)', (top[0], -top[1]), xytext=(0, 18), textcoords='offset points',
                        ha='center', fontsize=13, family='serif')
            for t, kk in list(zip(ts, okv))[1:]:
                qq2 = view(self.X[t][kk])[0]; tp = qq2[np.argmin(qq2[:, 1])]
                ax.annotate(f't = {t / 120:.1f} s', (tp[0], -tp[1]), xytext=(0, 12), textcoords='offset points',
                            ha='center', fontsize=11, family='serif', color='0.35')
        save(fig, out)

    def cloud_timelapse_glow(self, out, ts=(0, 114, 252), n_tracks=200, azim=-24, elev=42, W=12, pt=2.6,
                             bg='white', smooth=2.0, ghost_alpha=1.0, reflect=False, label=False, track_alpha=0.4, ghost_size=1.3,
                             floor=False, roll=-6.0, qlabel_offset=(0, 30), mirror=False):
        """'Hero' time-lapse: dark background, glowing point tracks (temporally smoothed for display,
        sigma=`smooth` frames) connecting the poses, a glossy floor with reflections, and the initial
        query pose highlighted with a white glow."""
        from scipy.ndimage import gaussian_filter1d
        light = np.mean(matplotlib.colors.to_rgb(bg)) > 0.5               # white-background theme
        grid_rgb, fg, fg2 = ((0.55, 0.55, 0.55), 'black', '0.35') if light else ((0.75, 0.75, 0.75), 'white', '0.65')
        ghost_alpha = (0.6 if light else 0.75) if ghost_alpha is None else ghost_alpha
        ts = [t for t in ts if t < len(self.frames)]
        okv = [self.vis[t] >= self.vis_thr for t in ts]
        Xs = gaussian_filter1d(self.X, smooth, axis=0, mode='nearest') if smooth else self.X
        allX = np.concatenate([self.X[t][k] for t, k in zip(ts, okv)]); cen = allX.mean(0); span = np.ptp(allX, 0).max()
        view = self._view(np.r_[cen[:2], 40.0], 2.4 * span, azim, elev)
        if mirror:                                                    # left-right flip of the rendered view
            _m = view; view = lambda P: (lambda qd: (qd[0] * [-1, 1], qd[1]))(_m(P))
        if roll:                                                      # in-plane rotation of the rendered view
            th = np.deg2rad(roll); Rr = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); _v = view
            view = lambda P: (lambda qd: (qd[0] @ Rr.T, qd[1]))(_v(P))
        low = allX[allX[:, 2] < 40] * [1, 1, -1]                      # keep only the near-floor part of the reflection in frame
        sel0 = self.trail_idx[fps(self.X[ts[0]][self.trail_idx], min(n_tracks, len(self.trail_idx)))]
        trk = Xs[ts[0]:ts[-1] + 1][:, sel0][self.vis[ts[0]:ts[-1] + 1][:, sel0] >= self.vis_thr]   # tracks stay in frame
        bnd = np.r_[allX, trk] if not (reflect and len(low)) else np.r_[allX, trk, low]
        q = view(bnd)[0]; lo, hi = np.percentile(q, 0.2, 0), np.percentile(q, 99.8, 0)
        pad = 0.05 * (hi - lo).max()
        figh = W * ((hi - lo)[1] + 2 * pad) / ((hi - lo)[0] + 2 * pad)
        fig = plt.figure(figsize=(W, figh), facecolor=bg); ax = fig.add_axes([0, 0, 1, 1], facecolor=bg)
        ppx = 72 * W / ((hi - lo)[0] + 2 * pad)                     # points per view unit (for line widths)
        # glossy floor: grid fading with distance from the action
        R = int(span * 1.1); step = 40
        for g in (np.arange(-R, R + 1, step) if floor else []):
            for P in (np.array([[cen[0] + g, cen[1] - R, 0], [cen[0] + g, cen[1] + R, 0]]),
                      np.array([[cen[0] - R, cen[1] + g, 0], [cen[0] + R, cen[1] + g, 0]])):
                L = np.linspace(P[0], P[1], 80); qq = view(L)[0] * [1, -1]
                a = np.clip(1 - np.linalg.norm(L[:, :2] - cen[:2], axis=1) / R, 0, 1) ** 1.5 * 0.22
                seg = np.stack([qq[:-1], qq[1:]], 1)
                ax.add_collection(LineCollection(seg, colors=np.c_[np.tile(grid_rgb, (len(seg), 1)), a[:-1] * (1.6 if light else 1)], linewidths=0.5, zorder=0))
        # tracks from the query frame to the last pose, glow = wide faint + thin bright strokes
        sel = self.trail_idx[fps(self.X[ts[0]][self.trail_idx], min(n_tracks, len(self.trail_idx)))]
        T0, T1 = ts[0], ts[-1]
        P = Xs[T0:T1 + 1][:, sel]; good = self.vis[T0:T1 + 1][:, sel] >= self.vis_thr
        qs, ds = view(P.reshape(-1, 3)); qs = qs.reshape(P.shape[0], -1, 2) * [1, -1]; ds = ds.reshape(P.shape[0], -1)
        segs = np.stack([qs[:-1], qs[1:]], 2); m = good[:-1] & good[1:]
        tt = (np.arange(T1 - T0) / max(T1 - T0 - 1, 1))[:, None].repeat(len(sel), 1)
        col = np.broadcast_to(self.colors[sel][None], m.shape + (3,))[m]; ramp = (0.35 + 0.65 * tt[m])
        if reflect:                                                   # faint mirror image in the floor
            Pr = P * [1, 1, -1]; qr = view(Pr.reshape(-1, 3))[0].reshape(P.shape[0], -1, 2) * [1, -1]
            sr = np.stack([qr[:-1], qr[1:]], 2)[m]
            hf = np.exp(-np.clip(P[:-1, :, 2], 0, None) / 50.0)[m]            # fade with height
            ax.add_collection(LineCollection(sr, colors=np.c_[col, 0.10 * ramp * hf], linewidths=0.5, zorder=0.5))
        for lw, a in (((8.0, 0.015), (3.0, 0.04), (0.8, 0.6)) if light else ((8.0, 0.02), (3.0, 0.05), (0.8, 0.5))):
            ax.add_collection(LineCollection(segs[m], colors=np.c_[col, a * ramp * track_alpha], linewidths=lw * 0.65, zorder=1, capstyle='round'))
        # poses
        def pose(t, k, alpha, size, glow):
            Xp = self.X[t][k]; qq, dd = view(Xp); o = np.argsort(-dd); qq = qq * [1, -1]
            shade = (1 - 0.3 * (dd - dd.min()) / max(np.ptp(dd), 1e-6))[o, None]
            c = self.colors[k][o] * shade
            if reflect:
                qr, dr = view(Xp * [1, 1, -1]); qr = qr * [1, -1]
                hf = np.exp(-np.clip(Xp[o, 2], 0, None) / 50.0)
                ax.scatter(qr[o, 0], qr[o, 1], s=size, c=np.c_[c, 0.10 * alpha * hf], edgecolors='none', zorder=0.6)
            g = (1.0 if glow else 0.35)
            for s_, a_ in ((size * 40, 0.010), (size * 14, 0.025), (size * 5, 0.07)):
                ax.scatter(qq[o, 0], qq[o, 1], s=s_, c=np.c_[c if light else 0.5 + 0.5 * c, np.full(len(o), a_ * g * (0.8 if light else 1))],
                           edgecolors='none', zorder=2.5)
            ax.scatter(qq[o, 0], qq[o, 1], s=size * 3, c=np.c_[c, np.full(len(o), 0.12 * alpha)], edgecolors='none', zorder=2.6)
            ax.scatter(qq[o, 0], qq[o, 1], s=size, c=np.c_[c, np.full(len(o), alpha)], edgecolors='none', zorder=2.7)
            return qq
        tops = []
        for t, k in list(zip(ts, okv))[1:]:
            tops.append((t, pose(t, k, ghost_alpha, pt * ghost_size, False)))
        q0 = pose(ts[0], okv[0], 1.0, pt * 1.6, True)
        if label:
            kw = dict(textcoords='offset points', ha='center', family='serif',
                      bbox=dict(boxstyle='round,pad=0.25', fc=bg, ec='none', alpha=0.75))
            anchor = lambda qq: (np.median(qq[:, 0]), np.percentile(qq[:, 1], 85))   # robust top-centre of a pose
            top = anchor(q0)
            ax.annotate('query point cloud  (t = 0)', (top[0], top[1]), xytext=qlabel_offset, fontsize=15, color=fg, **kw)
            for t, qq in tops:
                tp = anchor(qq)
                ax.annotate(f't = {t / 120:.1f} s', (tp[0], tp[1]), xytext=(0, 14), fontsize=12, color=fg2, **kw)
        ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(-hi[1] - pad, -lo[1] + pad); ax.set_aspect('equal'); ax.set_axis_off()
        for coll in ax.collections:          # points / tracks / grid -> raster in the PDF; text stays vector
            coll.set_rasterized(True)
        for ext in ('png', 'pdf'):
            fig.savefig(f'{out}.{ext}', dpi=300, facecolor=bg, pad_inches=0)
        plt.close(fig); print('wrote', out + '.png')


def save(fig, out, dpi=300):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig); print('wrote', out + '.png')


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trial', required=True, help='trial dir with metadata.yaml and vid/<cam>.mp4')
    ap.add_argument('--frame', type=int, default=670, help='query frame for the initial cloud')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--stages', default='cloud,track,figure')
    ap.add_argument('--sam-checkpoint', default=None)
    ap.add_argument('--extra-path', default=None, help='dir with vggt / segment_anything installs')
    ap.add_argument('--n-frames', type=int, default=360, help='frames to track (3D time panels use t up to 300)')
    ap.add_argument('--t', type=int, default=30, help='frame (relative to --frame) shown in the panels')
    ap.add_argument('--n-trails', type=int, default=800)
    ap.add_argument('--trail-px', type=float, default=1.2)
    a = ap.parse_args()
    if a.extra_path: sys.path.insert(0, os.path.abspath(a.extra_path))
    os.makedirs(a.out_dir, exist_ok=True)
    stages = a.stages.split(',')
    f_cloud, f_track = os.path.join(a.out_dir, 'cloud.npz'), os.path.join(a.out_dir, 'track.npz')
    if 'cloud' in stages:
        ckpt = a.sam_checkpoint
        if ckpt is None:
            ckpt = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(SAM_URL))
            if not os.path.exists(ckpt): torch.hub.download_url_to_file(SAM_URL, ckpt)
        xyz, rgb = build_cloud(a.trial, a.frame, ckpt)
        np.savez_compressed(f_cloud, xyz=xyz.astype(np.float32), rgb=rgb.astype(np.uint8), trial=a.trial, frame=a.frame)
        print(f'cloud: {len(xyz)} points -> {f_cloud}')
    if 'track' in stages:
        d = np.load(f_cloud)
        np.savez_compressed(f_track, **track_cloud(a.trial, a.frame, d['xyz'], a.n_frames))
        print('track ->', f_track)
    if 'figure' in stages:
        p = Panels(a.trial, dict(np.load(f_track)), n_trails=a.n_trails, trail_px=a.trail_px)
        o = lambda s: os.path.join(a.out_dir, s)
        p.tracks_main(o('cover_tracks_main'), a.t)
        p.tracks_main(o('cover_tracks_main_dense'), a.t, dense=True)
        p.row(o('cover_row'), a.t)
        p.input_grid(o('cover_input_grid'))
        p.frustums(o('cover_frustums'))
        p.clouds3d(o('cover_clouds3d_times'))
        p.cloud_timelapse(o('cover_cloud_timelapse'))
        p.cloud_timelapse_tracks(o('cover_cloud_timelapse_tracks'))
        p.cloud_timelapse_glow(o('cover_cloud_timelapse_glow'))


if __name__ == '__main__':
    main()
