"""Write pose3d.pq (frame-chunked zstd parquet) next to every pose3d.npz under the given dataset dirs.

The npz is left untouched; TracktailDataset reads the .pq instead when it exists (only the clip's
frames are decompressed). Each file is verified on a random subset of frames before it is renamed
into place. Existing .pq files are skipped unless --overwrite.

    python scripts/convert_pose_to_pq.py /path/posetail-pretraining-v6/point-odyssey-human \
        /path/posetail-pretraining-v6/po-animal-sub256-motion --workers 8
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tracktail.datasets.pose_store import PQ_NAME, convert_npz_to_pq  # noqa: E402


def find_npz(root):
    """<dataset>/<split>/<session>/<trial>/pose3d.npz (no recursive walk into img/ folders)."""
    out = []
    for split in sorted(os.listdir(root)):
        sp = os.path.join(root, split)
        if not os.path.isdir(sp):
            continue
        for sess in sorted(os.listdir(sp)):
            ss = os.path.join(sp, sess)
            if not os.path.isdir(ss):
                continue
            for trial in sorted(os.listdir(ss)):
                p = os.path.join(ss, trial, 'pose3d.npz')
                if os.path.exists(p):
                    out.append(p)
    return out


def _one(npz):
    t = time.time()
    path, nbytes = convert_npz_to_pq(npz)
    return npz, nbytes, os.path.getsize(npz), time.time() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset_dirs', nargs='+')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    todo = []
    for d in args.dataset_dirs:
        found = find_npz(d)
        skip = [p for p in found if not args.overwrite
                and os.path.exists(os.path.join(os.path.dirname(p), PQ_NAME))]
        print(f'{d}: {len(found)} trials, {len(skip)} already converted')
        todo += [p for p in found if p not in set(skip)]
    if args.dry_run or not todo:
        print(f'{len(todo)} to convert' + (' (dry run)' if args.dry_run else ''))
        return

    tot_pq = tot_npz = 0
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_one, p): p for p in todo}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                npz, nb, nz, dt = f.result()
            except Exception as e:  # keep going; report at the end
                print(f'FAILED {futs[f]}: {e}', flush=True)
                continue
            tot_pq += nb
            tot_npz += nz
            print(f'[{k}/{len(todo)}] {npz}  npz {nz / 1e6:.0f} MB -> pq {nb / 1e6:.0f} MB  {dt:.1f}s', flush=True)
    print(f'done: {tot_npz / 1e9:.2f} GB npz -> {tot_pq / 1e9:.2f} GB pq in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
