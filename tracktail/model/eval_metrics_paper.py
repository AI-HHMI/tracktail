import warnings
import numpy as np


# (static, dynamic, very dynamic) thresholds on a track's movement, used to split the tracks of a dataset.
# movement is the path length of the ground truth track over its visible frames. datasets that are
# None (or not listed) are scored over all of their tracks
PAPER_MOVEMENT_THRESHOLDS = {
    'kubric-multiview': (0.01, 0.1, 2.0),
    'dex_ycb':          (0.01, 0.1, 0.5),
    'cmupanoptic_3dgs': None,
}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_lower_median(x, axis = 1):
    '''
    median that ignores nans and, for an even number of values, returns the
    lower of the two middle values (the convention of torch.nanmedian, which is
    what the paper's mte numbers use) instead of their average

    parameters:
        x: array with nans for the entries to ignore
        axis: axis to take the median over

    returns:
        median with `axis` removed, nan where there were no values
    '''

    x = np.moveaxis(x, axis, 0)
    n_valid = np.sum(~np.isnan(x), axis = 0)
    x_sorted = np.sort(x, axis = 0)                      # nans go last
    idx = np.clip((n_valid - 1) // 2, 0, None)
    median = np.take_along_axis(x_sorted, idx[None], axis = 0)[0]
    median[n_valid == 0] = np.nan

    return median


def get_query_times(vis_true, query_times = None):
    '''
    broadcasts the query frame of each track to B, N. if none are given a
    track is queried at the first frame it is visible

    parameters:
        vis_true: B, T, N  bool
        query_times: (N,) or (B, N) int, or None
    '''

    B, T, N = vis_true.shape
    if query_times is None:
        return np.argmax(vis_true, axis = 1)

    query_times = np.asarray(query_times).astype(int)
    if query_times.ndim == 1:
        query_times = query_times[None]

    return np.broadcast_to(query_times, (B, N))


def get_track_movement(coords_true, vis_true):
    '''
    path length of each ground truth track over its visible frames

    parameters:
        coords_true: B, T, N, 3
        vis_true: B, T, N  bool (already False before the query frame)

    returns:
        movement: B, N
    '''

    B, T, N, _ = coords_true.shape

    movement = np.zeros((B, N))
    for b in range(B):
        for n in range(N):
            path = coords_true[b, vis_true[b, :, n], n]
            movement[b, n] = np.sum(np.linalg.norm(path[1:] - path[:-1], axis = -1))

    return movement


def get_track_metrics(coords_pred, coords_true, vis_pred, vis_true, query_times, thresholds, survival_threshold,
                      has_pred = None):
    '''
    metrics of every track on its own. the query frame and all frames before it are
    never evaluated, so predictions before the query do not count as false positives.
    frames where has_pred is False (no prediction, so predicted occluded) still count as misses in occlusion
    accuracy, the jaccard and delta_x (a visible frame without a position is not within any threshold). the
    metrics that are distances (ate, mte, fde, survival) are computed on the frames that have a prediction

    parameters:
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_pred: B, T, N  bool
        vis_true: B, T, N  bool (already False before the query frame)
        query_times: B, N  int
        thresholds: distances for delta_x and jaccard
        survival_threshold: failure distance for the survival rate
        has_pred: B, T, N  bool, frames with a prediction (None: all); vis_pred is False where it is False

    returns:
        dict of B, N arrays
    '''

    B, T, N = vis_true.shape
    t = np.arange(T)[None, :, None]
    qt = query_times[:, None, :]

    # tracks without a frame to evaluate give nan / inf here and are left out of the averages later
    with np.errstate(divide = 'ignore', invalid = 'ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category = RuntimeWarning)

        evaluated = t > qt                                           # B, T, N
        occ_true, occ_pred = ~vis_true, ~vis_pred
        dist = np.linalg.norm(coords_pred - coords_true, axis = -1)  # B, T, N

        # frames with a position, for the metrics that are distances
        has_pos = np.ones_like(vis_true) if has_pred is None else has_pred
        vis_dist = vis_true & has_pos
        n_visible = np.sum(vis_true & evaluated, axis = 1)
        n_visible_dist = np.sum(vis_dist & evaluated, axis = 1)

        metrics = {}
        metrics['occlusion_acc'] = np.sum((occ_pred == occ_true) & evaluated, axis = 1) / np.sum(evaluated, axis = 1)

        for thresh in thresholds:
            within = dist < thresh
            tp = np.sum(within & vis_pred & vis_true & evaluated, axis = 1)
            fp = np.sum(((~within & vis_pred) | (vis_pred & occ_true)) & evaluated, axis = 1)
            metrics[f'delta_x_{thresh:.3g}'] = np.sum(within & vis_dist & evaluated, axis = 1) / n_visible
            metrics[f'jaccard_{thresh:.3g}'] = tp / (n_visible + fp)

        metrics['delta_x_avg'] = np.mean([metrics[f'delta_x_{thresh:.3g}'] for thresh in thresholds], axis = 0)
        metrics['avg_jaccard'] = np.mean([metrics[f'jaccard_{thresh:.3g}'] for thresh in thresholds], axis = 0)

        # distances on the visible frames from the query on (the query frame itself counts here)
        dist_vis = np.where(vis_dist, dist, np.nan)
        dist_vis[np.broadcast_to(t < qt, dist_vis.shape)] = np.nan
        metrics['mte'] = get_lower_median(dist_vis, axis = 1)
        metrics['ate'] = np.nanmean(dist_vis, axis = 1)

        last_visible = np.argmax(vis_dist * np.arange(T)[None, :, None], axis = 1)
        metrics['fde'] = np.take_along_axis(dist_vis, last_visible[:, None, :], axis = 1)[:, 0]

        failed = (dist_vis > survival_threshold) & vis_dist
        first_failure = np.where(np.any(failed, axis = 1), np.argmax(failed, axis = 1), T)
        metrics['survival_rate'] = (first_failure - query_times) / (T - query_times)
        metrics['survival_rate'][n_visible_dist == 0] = np.nan       # no distance to judge it by


    return metrics


def get_eval_metrics_paper(coords_pred, coords_true, vis_pred, vis_true, query_times = None,
                           thresholds = None, survival_threshold = 50, movement_thresholds = None,
                           prefix = ''):
    '''
    evaluation metrics computed the way the paper reports them: per track, averaged over
    tracks, and (when movement_thresholds are given) as the mean of the static and the
    dynamic tracks. the same keys as get_eval_metrics (fractions, mte/ate/fde in the units
    of the coordinates) are returned for the headline score, and the scores of each group
    are added with a suffix (_any, _static, _dynamic, _very_dynamic)

    parameters:
        coords_pred: B, T, N, 3
        coords_true: B, T, N, 3
        vis_pred: B, T, N, 1  raw logit
        vis_true: B, T, N, 1  bool
        query_times: (N,) or (B, N) frame each track is queried at, None for its first visible frame
        coords_pred may be nan where a method gives no prediction (e.g. a failed triangulation): the point counts as
            predicted occluded there and a miss (occlusion accuracy, jaccard, delta_x), and has no distance (ate, mte, fde,
            survival)
        thresholds: distances for delta_x and jaccard
        survival_threshold: failure distance for the survival rate
        movement_thresholds: (static, dynamic, very dynamic) or None to score all tracks together
        prefix: added in front of every key

    returns:
        dict of floats
    '''

    if thresholds is None:
        thresholds = [1, 2, 4, 8, 16]

    vis_pred = _sigmoid(np.squeeze(vis_pred, axis = -1)) > 0.5               # B, T, N
    vis_true = np.squeeze(vis_true, axis = -1).astype(bool)                  # B, T, N
    B, T, N = vis_true.shape

    has_pred = np.isfinite(coords_pred).all(axis = -1)                       # B, T, N
    coords_pred = np.where(has_pred[..., None], coords_pred, 0)
    vis_pred = vis_pred & has_pred

    query_times = get_query_times(vis_true, query_times)
    vis_true = vis_true & (np.arange(T)[None, :, None] >= query_times[:, None, :])

    # tracks need the query frame and at least one more visible frame
    enough = np.sum(vis_true, axis = 1) >= 2                                 # B, N
    metrics = get_track_metrics(coords_pred, coords_true, vis_pred, vis_true,
                                query_times, thresholds, survival_threshold, has_pred)

    groups = {'any': np.ones((B, N), dtype = bool)}
    if movement_thresholds is not None:
        static, dynamic, very_dynamic = movement_thresholds
        movement = get_track_movement(coords_true, vis_true)
        groups['static'] = movement < static
        groups['dynamic'] = movement > dynamic
        groups['very_dynamic'] = movement > very_dynamic

    # score of each group: mean over its tracks, then over the batch
    scores = {}
    for group, mask in groups.items():
        mask = mask & enough
        if not np.any(mask):
            continue
        # a track without a predicted frame on its visible frames has no distance metrics: it is left out of those
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category = RuntimeWarning)
            per_batch = [{k: np.nanmean(v[b][mask[b]]) for k, v in metrics.items()}
                         for b in range(B) if np.any(mask[b])]
            scores[group] = {k: np.nanmean([s[k] for s in per_batch]) for k in metrics}
        scores[group]['n_tracks'] = int(np.sum(mask))

    headline = {k: scores['any'][k] for k in metrics}
    if 'static' in scores and 'dynamic' in scores:
        headline = {k: (scores['static'][k] + scores['dynamic'][k]) / 2 for k in metrics}

    results = {f'{prefix}{k}': float(v) for k, v in headline.items()}
    for group, score in scores.items():
        for k, v in score.items():
            results[f'{prefix}{k}_{group}'] = float(v)

    return results
