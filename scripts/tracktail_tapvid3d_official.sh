#!/usr/bin/env bash
#
# Tracktail on TAPVid-3D with the official benchmark protocol (the counterpart of arrow_baseline/arrow_tapvid3d.sh, for a
# like-for-like comparison with ARROW / D4RT): the benchmark's own queries (queries_xyt), tracked forward and backward
# (time-reversed video), every frame scored with the official TAPVid-3D metric (scripts/tapvid3d_official_metrics.py:
# depth-adaptive 1-16 px thresholds at 256 px, per-video median scaling; also without scaling, tracktail being metric).
# scripts/tracktail_tapvid3d.sh (first-visible queries, forward only, depth-relative thresholds) is NOT comparable to
# the paper numbers.
#
# Submit (one job per dataset): bash tracktail_tapvid3d_official.sh [dataset ...]
# Run here:               LOCAL=1 DEVICE=cuda:0 bash tracktail_tapvid3d_official.sh
# Env: VIS_THRESH (default 0.1) visibility threshold for inference and scoring (folder suffix _vis<thr>);
#      GT_POSES=1 gives tracktail the ground-truth camera poses (DriveTrack/ADT only; folder _gtposes);
#      MAX_FRAMES=48 for the D4RT short-clip setting (separate output folder _first48);
#      SPLIT=minival (default, 50 videos per dataset as in the papers) | full; NAME, WANDB_RUN, OUT_BASE, REPO, QUEUE, WALL.
# Predictions in $OUT/<dataset>/ are reused (resubmit a job that hit its time limit).

LOGS=${LOGS:-/groups/karashchuk/home/karashchukl/logs/tracktail}
WANDB_BASE=/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/karashchukl/results/paper-final-v2/posetail-outputs}
REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"}

NAME=${NAME:-main_finetune_vzkjqrys}
WANDB_RUN=${WANDB_RUN:-$WANDB_BASE/run-20260912_115732-vzkjqrys}
SPLIT=${SPLIT:-minival}
MAX_FRAMES=${MAX_FRAMES:-}   # e.g. 48: the first 48 frames only (D4RT setting)
GT_POSES=${GT_POSES:-}       # 1: give tracktail the ground-truth camera poses (DriveTrack, ADT)
VIS_THRESH=${VIS_THRESH:-0.1}   # visibility threshold in inference (occlusion carried between windows) and scoring
SUFFIX=$([[ -n "$MAX_FRAMES" ]] && echo "_first${MAX_FRAMES}")$([[ -n "$GT_POSES" ]] && echo "_gtposes")_vis${VIS_THRESH}
OUT=${OUT:-$OUT_BASE/${NAME}_tapvid3d_official_${SPLIT}${SUFFIX}}
DATASETS="tapvid3d_pstudio tapvid3d_drivetrack tapvid3d_adt"
QUEUE=${QUEUE:-gpu_h100}
WALL=${WALL:-24:00}

if [[ "$1" != "--worker" ]]; then
    SELF="$REPO/scripts/$(basename "${BASH_SOURCE[0]}")"
    if [[ -n "$LOCAL" ]]; then exec bash "$SELF" --worker ${@:-$DATASETS}; fi
    mkdir -p "$LOGS"
    for DS in ${@:-$DATASETS}; do   # one job per dataset
        J="${NAME}_${DS}_official_${SPLIT}${SUFFIX}"
        CMD="env NAME=$NAME WANDB_RUN=$WANDB_RUN SPLIT=$SPLIT MAX_FRAMES=$MAX_FRAMES GT_POSES=$GT_POSES VIS_THRESH=$VIS_THRESH OUT=$OUT REPO=$REPO bash $SELF --worker $DS"
        if command -v bsub >/dev/null 2>&1; then
            bsub -J "$J" -W "$WALL" -e "$LOGS/$J-%J.err" -o "$LOGS/$J-%J.out" -n 8 -q "$QUEUE" -gpu "num=1" $CMD
        else   # not on a submit host: go through login2
            ssh -n -o BatchMode=yes login2 "bsub -J '$J' -W '$WALL' -e '$LOGS/$J-%J.err' -o '$LOGS/$J-%J.out' -n 8 -q '$QUEUE' -gpu 'num=1' $CMD"
        fi
    done
    echo "logs: $LOGS   results: $OUT"
    exit 0
fi

# ---------------- worker ----------------
set -e
shift
[[ -f ~/.setup_env_vars.sh ]] && source ~/.setup_env_vars.sh
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH"
cd "$REPO"
if [[ -x .pixi/envs/default/bin/python ]]; then PYTHON=.pixi/envs/default/bin/python; else PYTHON="pixi run python"; fi

SPLIT_ARGS=$([[ "$SPLIT" == "full" ]] && echo "--all-trials" || echo "--trials-json arrow_baseline/tapvid3d_minival.json")
mkdir -p "$OUT"
for DS in "$@"; do
    echo "starting tracktail tapvid3d (official protocol): $NAME / $DS ($SPLIT) -> $OUT"
    $PYTHON scripts/tracktail_tapvid3d_official.py --wandb-folder "$WANDB_RUN" --out "$OUT" --datasets "$DS" \
        $SPLIT_ARGS --device "${DEVICE:-cuda:0}" ${MAX_FRAMES:+--max-frames $MAX_FRAMES} ${GT_POSES:+--gt-poses} --vis-thresh $VIS_THRESH
    # official metrics over the datasets that have predictions: median scaling (official default) and none (metric)
    $PYTHON scripts/tapvid3d_official_metrics.py --out "$OUT" --scaling median --tag _median --vis-thresh $VIS_THRESH
    $PYTHON scripts/tapvid3d_official_metrics.py --out "$OUT" --scaling none --tag _none --vis-thresh $VIS_THRESH
done
echo "done!"
