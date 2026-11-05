#!/usr/bin/env bash
#
# Tracktail on TAPVid-3D with the official benchmark protocol (for a like-for-like comparison with the ARROW / D4RT
# numbers): the benchmark's own queries (queries_xyt), tracked forward and backward
# (time-reversed video), every frame scored with the official TAPVid-3D metric (scripts/tapvid3d_official_metrics.py:
# depth-adaptive 1-16 px thresholds at 256 px, per-video median scaling; also without scaling, tracktail being metric).
#
# Run here (default):      DEVICE=cuda:0 bash scripts/tracktail_tapvid3d_official.sh [dataset ...]
# Submit to LSF instead:   SUBMIT=1 bash scripts/tracktail_tapvid3d_official.sh [dataset ...]   (one job per dataset)
# Paper protocol: QUERY_OCC=unknown (default), VIS_THRESH=0.1 (default), SPLIT=minival (default).
# Env: VIS_THRESH (default 0.1) visibility threshold for inference and scoring (folder suffix _vis<thr>);
#      GT_POSES=1 gives tracktail the ground-truth camera poses (DriveTrack/ADT only; folder _gtposes);
#      MAX_FRAMES=48 for the D4RT short-clip setting (separate output folder _first48);
#      SPLIT=minival (default, 50 videos per dataset as in the papers) | full; NAME (a network of scripts/paper_common.sh),
#      TRACKTAIL_OUT_BASE, QUEUE, WALL.
# Predictions in $OUT/<dataset>/ are reused (resubmit a job that hit its time limit).

REPO=${REPO:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"}
source "$REPO/scripts/paper_common.sh"
LOGS=${LOGS:-$TRACKTAIL_LOGS}

NAME=${NAME:-main_finetune_vzkjqrys}
SPLIT=${SPLIT:-minival}
MAX_FRAMES=${MAX_FRAMES:-}   # e.g. 48: the first 48 frames only (D4RT setting)
GT_POSES=${GT_POSES:-}       # 1: give tracktail the ground-truth camera poses (DriveTrack, ADT)
QUERY_OCC=${QUERY_OCC:-unknown}  # unknown: no visibility input at the query frame (folder _occunk); gt: visible
VIS_THRESH=${VIS_THRESH:-0.1}   # visibility threshold in inference (occlusion carried between windows) and scoring
SUFFIX=$([[ -n "$MAX_FRAMES" ]] && echo "_first${MAX_FRAMES}")$([[ -n "$GT_POSES" ]] && echo "_gtposes")_vis${VIS_THRESH}$([[ "$QUERY_OCC" == "unknown" ]] && echo "_occunk")
OUT=${OUT:-$TRACKTAIL_OUT_BASE/${NAME}_tapvid3d_official_${SPLIT}${SUFFIX}}
DATASETS="tapvid3d_pstudio tapvid3d_drivetrack tapvid3d_adt"
QUEUE=${QUEUE:-gpu_h100}
WALL=${WALL:-24:00}

if [[ "$1" != "--worker" ]]; then
    SELF="$REPO/scripts/$(basename "${BASH_SOURCE[0]}")"
    if [[ -z "${SUBMIT:-}" ]]; then exec bash "$SELF" --worker ${@:-$DATASETS}; fi
    mkdir -p "$LOGS"
    for DS in ${@:-$DATASETS}; do   # one job per dataset
        J="${NAME}_${DS}_official_${SPLIT}${SUFFIX}"
        CMD="env NAME=$NAME SPLIT=$SPLIT MAX_FRAMES=$MAX_FRAMES GT_POSES=$GT_POSES VIS_THRESH=$VIS_THRESH QUERY_OCC=$QUERY_OCC OUT=$OUT REPO=$REPO bash $SELF --worker $DS"
        command -v bsub >/dev/null 2>&1 || { echo "SUBMIT=1 needs LSF (bsub)"; exit 1; }
        bsub -J "$J" -W "$WALL" -e "$LOGS/$J-%J.err" -o "$LOGS/$J-%J.out" -n 8 -q "$QUEUE" -gpu "num=1" $CMD
    done
    echo "logs: $LOGS   results: $OUT"
    exit 0
fi

# ---------------- worker ----------------
set -e
shift
cd "$REPO"
PYTHON=$PIXI_PYTHON

MODEL_ARGS=$(model_args "$NAME")
SPLIT_ARGS=$([[ "$SPLIT" == "full" ]] && echo "--all-trials" || echo "--trials-json arrow_baseline/tapvid3d_minival.json")
mkdir -p "$OUT"
for DS in "$@"; do
    echo "starting tracktail tapvid3d (official protocol): $NAME / $DS ($SPLIT) -> $OUT"
    $PYTHON scripts/tracktail_tapvid3d_official.py $MODEL_ARGS --root "$TRACKTAIL_DATA" --out "$OUT" --datasets "$DS" \
        $SPLIT_ARGS --device "${DEVICE:-cuda:0}" ${MAX_FRAMES:+--max-frames $MAX_FRAMES} ${GT_POSES:+--gt-poses} --vis-thresh $VIS_THRESH --query-occlusion $QUERY_OCC
    # official metrics over the datasets that have predictions: median scaling (official default) and none (metric)
    $PYTHON scripts/tapvid3d_official_metrics.py --out "$OUT" --root "$TRACKTAIL_DATA" --scaling median --tag _median --vis-thresh $VIS_THRESH
    $PYTHON scripts/tapvid3d_official_metrics.py --out "$OUT" --root "$TRACKTAIL_DATA" --scaling none --tag _none --vis-thresh $VIS_THRESH
done
echo "done!"
