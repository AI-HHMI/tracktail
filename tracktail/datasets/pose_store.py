"""Optional frame-chunked pose store: ``pose3d.pq`` next to ``pose3d.npz``.

``pose3d.npz`` is zlib-compressed, so reading a 24-frame clip decompresses the whole trial
(point-odyssey-human: up to ~1.2 GB, ~4.5 s per sample, which stalls the GPU). ``pose3d.pq`` is a
parquet file with one row per frame (zstd, row groups of ``ROW_GROUP`` frames), so a clip only
decompresses the few row groups it touches (~25 ms, and ~3x smaller than the npz on disk).

The npz stays the source of truth; the dataset uses the .pq only when it exists and falls back
to the npz otherwise. Only ``pose`` (S, T, N, 3) and ``vis`` (S, T, N, C) are stored.

Convert with ``scripts/convert_pose_to_pq.py``.
"""

import ast
import os

import numpy as np

PQ_NAME = 'pose3d.pq'
ROW_GROUP = 32          # frames per row group
ZSTD_LEVEL = 3
_KEYS = ('pose', 'vis')

_pq_exists = {}         # per-process cache: npz path -> sibling .pq path or None


def pq_path_for(npz_path):
    """Sibling ``pose3d.pq`` of a ``pose3d.npz`` if it exists, else None (cached)."""
    if npz_path not in _pq_exists:
        p = None
        if os.path.basename(npz_path) == 'pose3d.npz':
            cand = os.path.join(os.path.dirname(npz_path), PQ_NAME)
            p = cand if os.path.exists(cand) else None
        _pq_exists[npz_path] = p
    return _pq_exists[npz_path]


def _meta(pf):
    md = pf.schema_arrow.metadata or {}
    out = {}
    for k in _KEYS:
        if f'{k}_shape'.encode() in md:
            out[k] = (tuple(ast.literal_eval(md[f'{k}_shape'.encode()].decode())),
                      np.dtype(md[f'{k}_dtype'.encode()].decode()))
    return out


def _column_to_array(col, n_rows, frame_shape, dtype):
    """Fixed-size-binary column (one frame per row) -> (n_rows, *frame_shape) array (zero-copy view)."""
    arr = col.combine_chunks() if hasattr(col, 'combine_chunks') else col
    width = int(np.prod(frame_shape)) * dtype.itemsize
    buf = np.frombuffer(arr.buffers()[1], dtype=np.uint8)
    buf = buf[arr.offset * width:(arr.offset + n_rows) * width]
    return buf.view(dtype).reshape((n_rows,) + tuple(frame_shape))


def read_pq_frames(pq_path, frames, keys=_KEYS):
    """Read ``frames`` (sorted int array of frame indices) -> {key: (S, len(frames), ...) array}."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(pq_path)
    meta = _meta(pf)
    keys = [k for k in keys if k in meta]
    md = pf.metadata
    rg_rows = md.row_group(0).num_rows
    frames = np.asarray(frames, dtype=np.int64)
    groups = sorted(set((frames // rg_rows).tolist()))
    tab = pf.read_row_groups(groups, columns=list(keys), use_threads=False)
    # row index within the concatenated groups (all groups but the last are rg_rows long)
    first = np.array(groups) * rg_rows
    starts = np.concatenate([[0], np.cumsum([md.row_group(g).num_rows for g in groups])[:-1]])
    pos = starts[np.searchsorted(groups, frames // rg_rows)] + (frames - first[np.searchsorted(groups, frames // rg_rows)])
    out = {}
    for k in keys:
        shape, dtype = meta[k]                      # full (S, T, ...) shape
        frame_shape = (shape[0],) + shape[2:]
        a = _column_to_array(tab.column(k), tab.num_rows, frame_shape, dtype)[pos]   # (n, S, ...)
        out[k] = np.ascontiguousarray(np.moveaxis(a, 0, 1))                          # (S, n, ...)
    return out


def load_pose_clip(npz_path, start_ix, end_ix, interval, with_vis=True):
    """``data['pose'][:, start:end:interval]`` (and ``vis``, None if absent) from .pq if present, else npz."""
    keys = ('pose', 'vis') if with_vis else ('pose',)
    pq_path = pq_path_for(npz_path)
    if pq_path is not None:
        import pyarrow.parquet as pq
        n_total = pq.ParquetFile(pq_path).metadata.num_rows
        frames = np.arange(start_ix, min(end_ix, n_total), interval)
        d = read_pq_frames(pq_path, frames, keys)
        return d['pose'], d.get('vis')
    data = np.load(npz_path)
    pose = data['pose'][:, start_ix:end_ix:interval]
    vis = data['vis'][:, start_ix:end_ix:interval] if (with_vis and 'vis' in data) else None
    return pose, vis


def load_pose_full(npz_path):
    """Whole ``pose`` array (S, T, N, 3), from .pq if present (faster zstd decode), else npz."""
    pq_path = pq_path_for(npz_path)
    if pq_path is not None:
        import pyarrow.parquet as pq
        n = pq.ParquetFile(pq_path).metadata.num_rows
        return read_pq_frames(pq_path, np.arange(n), ('pose',))['pose']
    return np.load(npz_path)['pose']


def convert_npz_to_pq(npz_path, out_path=None, row_group=ROW_GROUP, level=ZSTD_LEVEL, verify=True):
    """Write ``pose3d.pq`` next to ``npz_path`` (atomic: tmp file + rename). Returns (out_path, bytes)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = out_path or os.path.join(os.path.dirname(npz_path), PQ_NAME)
    data = np.load(npz_path)
    arrays = {k: data[k] for k in _KEYS if k in data}
    T = arrays['pose'].shape[1]
    cols, meta = {}, {}
    for k, a in arrays.items():
        assert a.shape[1] == T, (k, a.shape)
        fr = np.ascontiguousarray(np.moveaxis(a, 1, 0))            # (T, S, ...)
        width = fr[0].nbytes
        cols[k] = pa.Array.from_buffers(
            pa.binary(width), T, [None, pa.py_buffer(fr.reshape(-1).view(np.uint8))])
        meta[f'{k}_shape'.encode()] = str(tuple(a.shape)).encode()
        meta[f'{k}_dtype'.encode()] = str(a.dtype).encode()
    tab = pa.table(cols).replace_schema_metadata(meta)
    tmp = out_path + '.tmp'
    pq.write_table(tab, tmp, row_group_size=row_group, compression='zstd',
                   compression_level=level, use_dictionary=False, write_statistics=False)
    if verify:
        rng = np.random.default_rng(0)
        idx = np.unique(np.concatenate([[0, T - 1], rng.integers(0, T, 16)]))
        back = read_pq_frames(tmp, idx, tuple(arrays))
        for k, a in arrays.items():
            if not np.array_equal(back[k], a[:, idx], equal_nan=True):
                os.remove(tmp)
                raise ValueError(f'round-trip mismatch for {k} in {npz_path}')
    os.replace(tmp, out_path)
    _pq_exists.pop(npz_path, None)
    return out_path, os.path.getsize(out_path)
