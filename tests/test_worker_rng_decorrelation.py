"""Confirm DDP-rank RNG decorrelation for the train.py dataloader.

TracktailDataset drives nearly all of its sampling + augmentation through `np.random.*`.

Modern PyTorch (>=1.9, verified on 2.11) already reseeds numpy PER WORKER, so workers
within a rank are decorrelated out of the box -- the classic "numpy correlated across
workers" bug does NOT apply here. The remaining, real problem is cross-GPU:

  * train.py does `seed = fabric.broadcast(resolve_seed(...), src=0)` then `set_seeds(seed)`
    on EVERY rank -> identical `torch.initial_seed()` on every rank.
  * PyTorch's per-worker seed is `f(base_seed, worker_id)` -- rank-independent.
  * => worker `w` on rank 0 and worker `w` on rank 1 draw the SAME numpy/torch stream,
    so every GPU applies correlated augmentation/subsampling to its (different) clips.

`train_utils.make_worker_init_fn(seed, rank)` folds `rank` into the per-worker seed,
decorrelating the ranks. These tests reproduce the cross-rank correlation with a toy
dataset (the exact fork+numpy mechanism the real dataset relies on) and confirm the fix.

Two ranks are simulated as two subprocesses, each running `set_seeds`-equivalent seeding
with the SAME seed -- exactly what happens across GPUs under DDP.

Run:  pixi run python tests/test_worker_rng_decorrelation.py
"""

import os
import sys
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracktail.model.train_utils import make_worker_init_fn  # noqa: E402

SEED = 1234


class _NumpyDrawDataset(Dataset):
    """__getitem__ returns (worker_id, a numpy random draw) -- index-independent,

    exactly like TracktailDataset's augmentation/sampling decisions.
    """

    def __init__(self, n=64):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else -1
        return wid, np.random.randint(0, 2**31 - 1)


def _rank_worker_stream(rank, use_fix):
    """Body of a simulated DDP rank: seed identically to every other rank (as train.py

    does via the broadcast seed), build a train.py-style loader, return worker 0's draws.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # -> identical torch.initial_seed() on every rank
    wif = make_worker_init_fn(SEED, rank=rank) if use_fix else None
    loader = DataLoader(
        _NumpyDrawDataset(n=64),
        batch_size=1,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=wif,
    )
    by_worker = {}
    for wid, val in loader:
        by_worker.setdefault(int(wid), []).append(int(val))
    return by_worker.get(0, [])[:8]


def _stream_via_subprocess(rank, use_fix):
    """Run one simulated rank in a fresh process (faithful to per-GPU processes)."""
    cmd = [sys.executable, os.path.abspath(__file__), "--rank", str(rank)]
    if use_fix:
        cmd.append("--fix")
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    for line in out.stdout.splitlines():
        if line.startswith("STREAM "):
            return [int(x) for x in line[len("STREAM "):].split()]
    raise RuntimeError(f"no STREAM line from rank {rank}:\n{out.stdout}\n{out.stderr}")


def test_repro_ranks_correlated_without_fix():
    """BUG REPRO: two ranks with the same broadcast seed emit IDENTICAL worker streams."""
    r0 = _stream_via_subprocess(rank=0, use_fix=False)
    r1 = _stream_via_subprocess(rank=1, use_fix=False)
    assert len(r0) >= 4 and len(r1) >= 4
    assert r0 == r1, f"expected correlated cross-rank streams without the fix; {r0} vs {r1}"


def test_ranks_decorrelated_with_fix():
    """FIX: with make_worker_init_fn(seed, rank), the two ranks emit different streams."""
    r0 = _stream_via_subprocess(rank=0, use_fix=True)
    r1 = _stream_via_subprocess(rank=1, use_fix=True)
    assert len(r0) >= 4 and len(r1) >= 4
    assert r0 != r1, f"ranks still correlated after applying make_worker_init_fn; {r0} vs {r1}"


def test_workers_still_decorrelated_within_rank():
    """Sanity: the fix must not RE-correlate workers within a rank (torch already splits them)."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    loader = DataLoader(
        _NumpyDrawDataset(n=64), batch_size=1, num_workers=2, prefetch_factor=2,
        persistent_workers=True, worker_init_fn=make_worker_init_fn(SEED, rank=0))
    by_worker = {}
    for wid, val in loader:
        by_worker.setdefault(int(wid), []).append(int(val))
    n = min(len(by_worker[0]), len(by_worker[1]))
    assert n >= 4
    assert by_worker[0][:n] != by_worker[1][:n], "workers within a rank got re-correlated"


if __name__ == "__main__":
    # subprocess entry point: emit worker-0's stream for one simulated rank
    if "--rank" in sys.argv:
        rank = int(sys.argv[sys.argv.index("--rank") + 1])
        stream = _rank_worker_stream(rank, use_fix=("--fix" in sys.argv))
        print("STREAM " + " ".join(str(x) for x in stream))
        sys.exit(0)

    test_repro_ranks_correlated_without_fix()
    print("test_repro_ranks_correlated_without_fix: PASS (cross-rank correlation confirmed)")
    test_ranks_decorrelated_with_fix()
    print("test_ranks_decorrelated_with_fix: PASS (fix decorrelates ranks)")
    test_workers_still_decorrelated_within_rank()
    print("test_workers_still_decorrelated_within_rank: PASS")
    print("ALL PASS")
