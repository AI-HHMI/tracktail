#!/usr/bin/env bash
#
# ARROW (RGB-only, https://github.com/ifradlin/arrow) on the TAPVid-3D datasets (tapvid3d_adt, tapvid3d_drivetrack,
# tapvid3d_pstudio), the counterpart of scripts/tracktail_tapvid3d.sh. Per dataset: arrow_inference.py, then
# scripts/eval_testset_metrics.py on the saved predictions (the same paper-protocol and depth-relative *_cam / *_world
# metrics as tracktail; it never loads a tracktail model here), merge_eval_summaries.py + summary_to_csv.py
# (summary.json, metrics.md, summary.csv over the finished datasets), and scripts/tapvid3d_official_metrics.py: the
# official TAPVid-3D AJ / APD / OA (depth-adaptive pixel thresholds, per-video median scale), next to the numbers of the
# ARROW paper (arXiv 2610.01314, Table 2), in tapvid3d_official.md.
#
# Protocol (ARROW paper, Sec. 4, TAPVid-3D): monocular, no intrinsics given to the model, every target queried against
# the full encoded sequence (WINDOW=300 >= the longest TAPVid-3D clip: one encoding per video), each target predicted in
# its own camera's coordinates (--reference target; the ground truth is per-frame camera coordinates) and one
# per-video median scale (Eq. 11, --scale-mode gt-tracks).
#
# Submit all datasets:        bash arrow_tapvid3d.sh
# Submit a subset by name:    bash arrow_tapvid3d.sh tapvid3d_pstudio tapvid3d_adt
# Run here instead of bsub:   LOCAL=1 DEVICE=cuda:0 bash arrow_tapvid3d.sh tapvid3d_pstudio
# Worker (run by bsub):       bash arrow_tapvid3d.sh --worker <dataset> [dataset ...]
#
# Env overrides:
#   SPLIT    minival (default: the TAPVid-3D mini-val videos, 50 per dataset as in the paper, from tapvid3d_minival.json)
#            | full (every test trial: ~1900 adt, ~2400 drivetrack, ~150 pstudio trials)
#   QUERIES  xyt (default: TAPVid-3D's own query pixels and frames, tracked both ways, as in the paper)
#            | first (each track queried at its first visible frame, as tracktail_tapvid3d.sh)
#   REFERENCE=first: every target in the first camera's frame (the world-coordinate task; output folder -world);
#            the camera-coordinate metrics of that folder are meaningless, score it with tapvid3d_world_metrics.py.
#   NO_DT_CROP=1 disables the DriveTrack crop + recentering (the authors' protocol, issue #3 of ifradlin/arrow).
#   WINDOW (300), DEVICE (cuda:0), QUEUE (gpu_h100), WALL (24:00), OUT_BASE / OUT, ARROW_PYTHON, CKPT, LOGS.
# One bsub job per dataset. A full-context ADT video (300 frames, 512x512) takes ~80 s on an RTX PRO 6000, so SPLIT=full
# adt (~1900 videos) does not fit one 24 h job: predictions already in <dataset>/ are reused, so just submit again.
#
# Results go to $OUT (default $OUT_BASE/tapvid3d-<split>-q<queries>/). The ARROW numbers to compare to the paper are
# tapvid3d_official.md with SPLIT=minival QUERIES=xyt; summary.csv / metrics.md with QUERIES=first are the ones to put
# next to tracktail's.

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"   # physical path: the nodes do not see ~/ghome
REPO="$SCRIPTS/.."
LOGS=${LOGS:-/groups/karashchuk/home/karashchukl/logs/arrow}
DATASET_ROOT=/groups/karashchuk/karashchuklab/animal-datasets-processed/posetail-pretraining-v6
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/karashchukl/results/paper-final-v2/arrow-outputs}
SPLIT=${SPLIT:-minival}
QUERIES=${QUERIES:-xyt}
REFERENCE=${REFERENCE:-target}   # target: camera coordinates (default); first: world frame = the first camera
REFSUF=$([[ "$REFERENCE" == "first" ]] && echo "-world")
OUT=${OUT:-$OUT_BASE/tapvid3d-${SPLIT}-q${QUERIES}${REFSUF}}
WINDOW=${WINDOW:-300}
CKPT=${CKPT:-/groups/karashchuk/home/karashchukl/weights/arrow}   # local copy of the ifradlin/arrow hub weights

DATASETS="tapvid3d_pstudio tapvid3d_drivetrack tapvid3d_adt"
QUEUE=${QUEUE:-gpu_h100}
WALL=${WALL:-24:00}

if [[ "$SPLIT" != "minival" && "$SPLIT" != "full" ]]; then echo "SPLIT must be minival or full, got $SPLIT"; exit 1; fi
if [[ "$QUERIES" != "xyt" && "$QUERIES" != "first" ]]; then echo "QUERIES must be xyt or first, got $QUERIES"; exit 1; fi

if [[ "$1" != "--worker" ]]; then
    SELF="$SCRIPTS/$(basename "${BASH_SOURCE[0]}")"
    if [[ -n "$LOCAL" ]]; then
        exec bash "$SELF" --worker ${@:-$DATASETS}
    fi
    mkdir -p "$LOGS"
    ENVS="SPLIT=$SPLIT QUERIES=$QUERIES REFERENCE=$REFERENCE OUT=$OUT WINDOW=$WINDOW CKPT=$CKPT"
    [[ -n "$ARROW_PYTHON" ]] && ENVS="$ENVS ARROW_PYTHON=$ARROW_PYTHON"
    for DS in ${@:-$DATASETS}; do
        NAMEJ="arrow-${DS}-${SPLIT}-q${QUERIES}${REFSUF}"
        CMD="env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $ENVS bash $SELF --worker $DS"
        if command -v bsub >/dev/null 2>&1; then
            bsub -J "$NAMEJ" -W "$WALL" -n 8 -q "$QUEUE" -gpu "num=1" -e "$LOGS/$NAMEJ-%J.err" -o "$LOGS/$NAMEJ-%J.out" "$CMD"
        else
            ssh -n -o BatchMode=yes login2 "bsub -J '$NAMEJ' -W '$WALL' -n 8 -q '$QUEUE' -gpu 'num=1' -e '$LOGS/$NAMEJ-%J.err' -o '$LOGS/$NAMEJ-%J.out' \"$CMD\""
        fi
    done
    echo "logs: $LOGS   results: $OUT"
    exit 0
fi

# ---------------- worker ----------------
set -e
shift
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ARROW's uv environment: .venv-cluster (built on login2, python on /groups, see submit_arrow.sh) when it exists
if [[ -z "$ARROW_PYTHON" ]]; then
    if [[ -x "$REPO/../arrow/.venv-cluster/bin/python" ]]; then ARROW_PYTHON="$REPO/../arrow/.venv-cluster/bin/python"
    else ARROW_PYTHON="$REPO/../arrow/.venv/bin/python"; fi   # cd ../arrow && uv sync
fi
# the pixi env needs its own libstdc++ on the nodes (as scripts/tracktail_tapvid3d.sh); only for its python, not ARROW's
if [[ -x "$REPO/.pixi/envs/default/bin/python" ]]; then
    PYTHON="env LD_LIBRARY_PATH=$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH $REPO/.pixi/envs/default/bin/python"
else PYTHON="pixi run python"; fi

QUERY_ARGS=$([[ "$QUERIES" == "xyt" ]] && echo "--queries-xyt" || echo "--query-first")
TRIALS_JSON="$SCRIPTS/tapvid3d_minival.json"
mkdir -p "$OUT"

for DS in "$@"; do
    # dataset root: the real one, or (minival) a tree of symlinks to the mini-val trials, so that arrow_inference.py
    # and eval_testset_metrics.py see only those ("_root" is skipped by merge_eval_summaries.py)
    ROOT=$DATASET_ROOT
    if [[ "$SPLIT" == "minival" ]]; then
        ROOT="$OUT/_root"
        $PYTHON - "$DATASET_ROOT" "$ROOT" "$DS" "$TRIALS_JSON" <<'EOF'
import json, os, sys
src, dst, ds, js = sys.argv[1:]
missing = 0
for tid in json.load(open(js))[ds]:
    seq, trial = tid.rsplit('_', 1)
    s, d = os.path.join(src, ds, 'test', seq, trial), os.path.join(dst, ds, 'test', seq, trial)
    if not os.path.isdir(s):
        missing += 1
        continue
    os.makedirs(os.path.dirname(d), exist_ok=True)
    if not os.path.lexists(d):
        os.symlink(s, d)
print(f'{ds}: mini-val root {dst} ({missing} mini-val videos not in the dataset)')
EOF
    fi

    # DriveTrack: ARROW's Waymo preprocessing (authors, github.com/ifradlin/arrow/issues/3): drop the top 394 rows
    # (1920x1280 -> 1920x886, resized to 504x238) and recenter the camera; no query falls in the removed strip
    EXTRA=""
    [[ "$DS" == "tapvid3d_drivetrack" && -z "$NO_DT_CROP" ]] && EXTRA="--crop-top 394 --recenter"

    echo "starting arrow tapvid3d eval: $DS ($SPLIT, queries $QUERIES) -> $OUT"
    (cd "$SCRIPTS" && $ARROW_PYTHON arrow_inference.py \
        --dataset-root "$ROOT" \
        --output-root  "$OUT" \
        --datasets     "$DS" \
        --split        test \
        --device       "${DEVICE:-cuda:0}" \
        --checkpoint   "$CKPT" \
        $QUERY_ARGS \
        --reference    "$REFERENCE" \
        --scale-mode   gt-tracks \
        --window       "$WINDOW" $EXTRA)

    if [[ "$REFERENCE" == "first" ]]; then continue; fi   # world frame: scored by scripts/tapvid3d_world_metrics.py

    # metrics of the saved predictions (all trials are cached: a missing one fails instead of running tracktail)
    $PYTHON "$REPO/scripts/eval_testset_metrics.py" \
        --root          "$ROOT" \
        --out           "$OUT" \
        --datasets      "$DS" \
        --wandb-folder  /nonexistent-no-tracktail-model

    # every call above rewrites summary.json with its own dataset only: rebuild it over the datasets that are finished,
    # and write summary.csv (one row per dataset)
    $PYTHON "$REPO/scripts/merge_eval_summaries.py" "$OUT"
    $PYTHON "$REPO/scripts/summary_to_csv.py" "$OUT"

    # official TAPVid-3D metrics over the datasets that have predictions, with the paper's numbers
    $PYTHON "$REPO/scripts/tapvid3d_official_metrics.py" --out "$OUT" --root "$ROOT"
done

echo "done!"
