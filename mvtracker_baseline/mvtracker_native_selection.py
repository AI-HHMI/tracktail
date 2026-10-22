"""Ground-truth visibility and track/query selection for the MVTracker-native datasets
(kubric-multiview, dex-ycb-multiview, panoptic-multiview), following MVTracker's own dataloaders
(mvtracker/datasets/{kubric_multiview,dexycb_multiview,panoptic_studio_multiview}_dataset.py).

Used by mvtracker_inference_native.py, mvtracker_inference_native_official.py and
mvtracker_metrics_native.py so inference, query selection and scoring all see the same labels.

Track selection by default uses MVTracker's own cached track files (<trial>/cache/cachedtracks--*.npz,
shipped with the released datasets) so the evaluated tracks and query frames are exactly the paper's.
If no cache exists for the requested cameras, if a different number of queries is requested, or if a seed
is given, the selection is re-derived with the official policy and seeds instead (this reproduces the
caches exactly for panoptic and dex-ycb, and the tracks, and ~97-99% of the query times, for kubric).

What the official loaders do, which is also what the re-derivation replicates:

  visibility (per selected view, then "visible" = any selected view):
    kubric   : ~occlusion from view_<v>/tracks_2d.npz
    dex-ycb  : tracks_2d_visibilities[views]; for every view, a track invisible at frame 0 is also set
               invisible for frames 0-4 ("probably on the hand")
    panoptic : per_view_visibilities[views], set invisible before the TAPVid-3D query time
               query_points_3d[:, 0]

  track selection:
    panoptic : visible in >= 2 frames; random sample (torch.randperm, seed 72); n = 384
    dex-ycb  : visible in >= 2 frames and not "hectic" (>= 3 views losing visibility in one step);
               random sample (seed 72); n = 384
    kubric   : tracks first stratified by GT movement (50% dynamic, 25% very dynamic, rest static),
               candidates = visible at the first or the middle frame and in >= 2 frames; random
               sample (seed 72 + dataset index); n = 512

  queries: each track is queried at its first visible frame (the last visible frame is excluded);
    for dex-ycb and kubric a quarter of the tracks are instead queried at a random visible frame.
"""
from pathlib import Path

import numpy as np
import torch

DEFAULT_N_QUERIES = {"kubric": 512, "dex-ycb": 384, "panoptic": 384}
OFFICIAL_SEED = 72


# ── ground-truth visibility ───────────────────────────────────────────────────

def load_gt(dataset_type: str, trial_dir: Path, views):
    """Return (traj3d (T,N,3) float, vis_views (C,T,N) bool), with the official visibility rules applied."""
    trial_dir = Path(trial_dir)
    views = list(views)
    if dataset_type == "kubric":
        traj3d = np.load(trial_dir / "tracks_3d.npz")["tracks_3d"].astype(np.float64)
        vis = np.stack([~np.load(trial_dir / f"view_{v}" / "tracks_2d.npz")["occlusion"] for v in views])
    elif dataset_type == "dex-ycb":
        d = np.load(trial_dir / "tracks_3d.npz", allow_pickle=True)
        traj3d = np.asarray(d["tracks_3d"], dtype=np.float64)
        vis = np.asarray(d["tracks_2d_visibilities"])[views].astype(bool).copy()
        inv0 = ~vis[:, 0, :]                                 # (C, N): invisible at frame 0 in that view
        vis[:, 0:5, :] &= ~inv0[:, None, :]
    elif dataset_type == "panoptic":
        a = np.load(trial_dir / "tapvid3d_annotations.npz")
        traj3d = a["trajectories"].astype(np.float64)
        vis = a["per_view_visibilities"][views].astype(bool).copy()
        qt = a["query_points_3d"][:, 0].round().astype(int)
        vis &= (np.arange(vis.shape[1])[None, :, None] >= qt[None, None, :])
    else:
        raise ValueError(dataset_type)
    return traj3d, vis


def gt_visibility(dataset_type: str, trial_dir: Path, views):
    """(T, N_full) bool: point visible in any of the selected views (official rules applied)."""
    return load_gt(dataset_type, trial_dir, views)[1].any(0)


# ── track + query selection ───────────────────────────────────────────────────

def _queries_from_visibility(gt_vis, rnd_np, n_non_first):
    """gt_vis (T,N) bool for the sampled tracks. Returns query times (N,) int."""
    T, N = gt_vis.shape
    gv = torch.from_numpy(gt_vis.copy())
    last_visible = (torch.arange(T).unsqueeze(-1) * gv).max(0).values
    gv[last_visible[None, :], torch.arange(N)] = False
    n_first = N - n_non_first
    first = torch.argmax(gv[..., -n_first:].float(), dim=0)
    non_first = first.new_zeros((n_non_first,))
    for i in range(n_non_first):
        nz = torch.nonzero(gv[:, i] == 1)
        non_first[i] = nz[rnd_np.randint(len(nz))].item()
    return torch.cat([non_first, first]).numpy().astype(int)


def _select_derived(dataset_type: str, trial_dir: Path, views, n_queries=None, seed=None, trial_index=0):
    """Re-derive the tracks to evaluate and their query times with the official policy.

    Returns dict with, all aligned and sorted by original track index:
      track_ids (N,) int   original track index in the dataset's full track array
      query_t   (N,) int   query frame
      query_xyz (N,3) f32  GT position at the query frame (raw world frame)
      valid_mask (N_full,) bool
      vis_any   (T,N_full) bool  official GT visibility (any selected view)
    """
    if n_queries is None:
        n_queries = DEFAULT_N_QUERIES[dataset_type]
    traj3d, vis = load_gt(dataset_type, trial_dir, views)
    T, N_full, _ = traj3d.shape
    if seed is None:
        seed = OFFICIAL_SEED + (trial_index if dataset_type == "kubric" else 0)
    rnd_torch = torch.Generator().manual_seed(seed)
    rnd_np = np.random.RandomState(seed=seed)
    orig_ids = np.arange(N_full)

    if dataset_type == "kubric":
        # stratify by movement (visible at t and t+1 in any selected view)
        vis2 = (vis[:, :-1] & vis[:, 1:]).any(0)
        movement = np.linalg.norm(traj3d[1:] - traj3d[:-1], axis=-1)
        movement[~vis2] = 0
        movement = movement.sum(0)
        static_p = movement < 0.01
        dyn_p = movement > 0.1
        vdyn_p = movement > 2.0
        r_dyn, r_vdyn, max_preload_cfg = 0.5, 0.25, 18000
        mtp = min(N_full, int(dyn_p.sum() / r_dyn), int(vdyn_p.sum() // r_vdyn),
                  int(static_p.sum() / (1 - r_dyn - r_vdyn)), max_preload_cfg)
        n_dyn = min(int(mtp * r_dyn), dyn_p.sum())
        n_vdyn = min(int(mtp * r_vdyn), vdyn_p.sum())
        n_static = mtp - n_dyn - n_vdyn
        sel = np.concatenate([rnd_np.choice(np.where(dyn_p)[0], n_dyn, replace=False),
                              rnd_np.choice(np.where(vdyn_p)[0], n_vdyn, replace=False),
                              rnd_np.choice(np.where(static_p)[0], n_static, replace=False)])
        rnd_np.shuffle(sel)
        sel = sel[:mtp]
        vis_s = vis[:, :, sel]
        va = vis_s.any(0)
        cand = np.union1d(np.nonzero(va[0])[0], np.nonzero(va[T // 2])[0])
        cand = cand[(va.sum(0) >= 2)[cand]]
        perm = torch.randperm(len(cand), generator=rnd_torch).numpy()
        n = min(len(cand), n_queries)
        pick = cand[perm[:n]]                       # indices into the preloaded (selected) array
        track_ids = sel[pick]
        gt_vis = va[:, pick]
        n_non_first = n // 4
    else:
        va_full = vis.any(0)
        if dataset_type == "dex-ycb":
            hectic = ((vis[:, :-1] & ~vis[:, 1:]).sum(0) >= 3).any(0)
            valid = (va_full.sum(0) >= 2) & ~hectic
            n_non_first_frac = True
        else:  # panoptic
            valid = va_full.sum(0) >= 2
            n_non_first_frac = False
        vt = np.nonzero(valid)[0]
        perm = torch.randperm(len(vt), generator=rnd_torch).numpy()
        n = min(len(vt), n_queries)
        track_ids = vt[perm[:n]]
        gt_vis = va_full[:, track_ids]
        n_non_first = n // 4 if n_non_first_frac else 0

    query_t = _queries_from_visibility(gt_vis, rnd_np, n_non_first)
    # the same original track can be drawn twice by the kubric stratification; keep the first
    _, first_idx = np.unique(track_ids, return_index=True)
    first_idx.sort()
    track_ids, query_t = track_ids[first_idx], query_t[first_idx]
    order = np.argsort(track_ids, kind="stable")
    track_ids, query_t = track_ids[order], query_t[order]
    valid_mask = np.zeros(N_full, dtype=bool)
    valid_mask[track_ids] = True
    return dict(track_ids=track_ids, query_t=query_t,
                query_xyz=traj3d[query_t, track_ids].astype(np.float32),
                valid_mask=valid_mask, vis_any=vis.any(0), source="derived")


# ── cached tracks (default) ───────────────────────────────────────────────────

def cache_path(dataset_type: str, trial_dir: Path, views):
    """Where the authors' cached track file for these views lives (it may not exist for other cameras)."""
    v = "_".join(str(x) for x in views)
    name = {
        "kubric": f"cachedtracks--seed{OFFICIAL_SEED}-dynamic0.5-verydynamic-0.25-views{v}-n512-numviews-1-t24--v1.npz",
        "dex-ycb": f"cachedtracks--seed{OFFICIAL_SEED}-views{v}-n384--v1.npz",
        "panoptic": f"cachedtracks--seed{OFFICIAL_SEED}-views{v}-n384--v1.npz",
    }[dataset_type]
    return Path(trial_dir) / "cache" / name


def _select_from_cache(dataset_type, trial_dir, views, path, n_queries=None):
    """Tracks and query frames from the authors' cache file. Returns None if n_queries disagrees with it."""
    c = np.load(path)
    n_cached = int(c["query_points"].shape[0])
    if n_queries is not None and n_queries != n_cached:
        return None
    traj3d, vis = load_gt(dataset_type, trial_dir, views)
    query_t = c["query_points"][:, 0].round().astype(int)
    if dataset_type == "kubric":
        # track_indices refer to the movement-stratified pool, not the dataset's track array: map back
        # to the dataset's track ids by exact trajectory content
        cached = c["traj3d_world"].astype(np.float32)               # (T, N, 3)
        full = traj3d.astype(np.float32)
        lookup = {}
        for j in range(full.shape[1]):
            lookup.setdefault(full[:, j].tobytes(), j)
        track_ids = np.array([lookup.get(cached[:, k].tobytes(), -1) for k in range(cached.shape[1])])
        if (track_ids < 0).any():
            raise ValueError(f"{path}: {(track_ids < 0).sum()} cached tracks not found in the dataset's tracks")
    else:
        track_ids = c["track_indices"].astype(int)
    # the same track can appear twice in the kubric pool; keep the first occurrence
    _, first_idx = np.unique(track_ids, return_index=True)
    first_idx.sort()
    track_ids, query_t = track_ids[first_idx], query_t[first_idx]
    cache_vis = c["visibility"].astype(bool)[:, :, first_idx]       # (V, T, N)
    vis_mismatch = float((cache_vis != vis[:, :, track_ids]).mean())
    order = np.argsort(track_ids, kind="stable")
    track_ids, query_t = track_ids[order], query_t[order]
    valid_mask = np.zeros(traj3d.shape[1], dtype=bool)
    valid_mask[track_ids] = True
    return dict(track_ids=track_ids, query_t=query_t,
                query_xyz=traj3d[query_t, track_ids].astype(np.float32),
                valid_mask=valid_mask, vis_any=vis.any(0), source="cache",
                cache_file=str(path), cache_vis_mismatch=vis_mismatch)


def select_tracks(dataset_type: str, trial_dir: Path, views, n_queries=None, seed=None, trial_index=0,
                  use_cache: bool = True):
    """Tracks and query frames for one trial. Uses the authors' cached track file when available
    (use_cache=True, no custom seed, and n_queries is None or equal to the cache size); otherwise re-derives
    the selection with the official policy. The result's "source" says which one was used.

    Returns dict, all aligned and sorted by original track index:
      track_ids (N,) int, query_t (N,) int, query_xyz (N,3) float32 (GT position at the query frame),
      valid_mask (N_full,) bool, vis_any (T,N_full) bool (official GT visibility, any selected view), source.
    """
    if use_cache and seed is None:
        path = cache_path(dataset_type, trial_dir, views)
        if path.exists():
            r = _select_from_cache(dataset_type, trial_dir, views, path, n_queries)
            if r is not None:
                return r
    return _select_derived(dataset_type, trial_dir, views, n_queries=n_queries, seed=seed,
                           trial_index=trial_index)
