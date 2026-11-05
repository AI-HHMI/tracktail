#!/usr/bin/env bash
# MVTracker on DexYCB with Kinect SENSOR depth (MVTracker's dex-ycb-multiview view_XX/depth/*.png, mm -> m),
# same protocol as mvtracker_cached.sh's DUSt3R dex_ycb row: canonical view_00..03, pose3d_cached.npz,
# --official-settings (vis thr 0.01), --query-first, view-seed 0, same metrics code.
# This standalone driver produced the paper's DexYCB GT (†Kinect) MVTracker row (Tab. main_results), from
#   $MVTRACKER_OUT_BASE-dexycb-sensordepth/paper-canonical-cached-sensor-linear
#   (default $TRACKTAIL_RESULTS/mvtracker-outputs-dexycb-sensordepth/paper-canonical-cached-sensor-linear).
# mvtracker_cached.sh runs the same call (sensor depth, linear resize) for its dex_ycb GT row.
# The paper run used the ethz-vlg/mvtracker clone at ceea8ad as torch.hub source (MVT_HUB_DIR).
#
# Usage: bash mvtracker_dexycb_sensordepth.sh [linear|nearest|duster]   (duster = re-run of the paper DUSt3R row, env check)
# Paths come from paper.env (TRACKTAIL_DATA, MVTRACKER_DATA, MVTRACKER_REPO, DUST3R_REPO, MVT_HUB_DIR, MVTRACKER_CKPT,
# MVTRACKER_PYTHON). Env overrides: DATASET_ROOT, OUTPUT_BASE, DEVICE (default cuda:0), PYTHON, CKPT.
set -e
RESIZE=${1:-linear}
DEVICE=${DEVICE:-cuda:0}
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS/.." && pwd)"
source "$REPO/scripts/paper_common.sh"
BASELINE_LD=${LD_LIBRARY_PATH:-}   # LD_LIBRARY_PATH without the pixi libs of paper_common.sh (those are for the pixi python)
while [[ "$BASELINE_LD" == "$PAPER_REPO/.pixi/envs/default/lib:"* ]]; do BASELINE_LD=${BASELINE_LD#"$PAPER_REPO/.pixi/envs/default/lib:"}; done
export PYTHONPATH="$MVTRACKER_REPO:$DUST3R_REPO:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MVT_HUB_DIR                        # torch.hub loads the pinned clone (paper.env)
PYTHON=${PYTHON:-env LD_LIBRARY_PATH=$BASELINE_LD $MVTRACKER_PYTHON}
CKPT=${CKPT:-$MVTRACKER_CKPT}
DATASET_ROOT=${DATASET_ROOT:-$TRACKTAIL_DATA}
NATIVE=${MVTRACKER_DATA:?set MVTRACKER_DATA (paper.env)}/dex-ycb-multiview
BASE=${OUTPUT_BASE:-$MVTRACKER_OUT_BASE-dexycb-sensordepth}
if [[ "$RESIZE" == "duster" ]]; then
    OUT=$BASE/paper-canonical-cached-duster-rerun; DEPTH_ARGS="--depth-type duster"
else
    OUT=$BASE/paper-canonical-cached-sensor-$RESIZE; DEPTH_ARGS="--depth-type gt-native --native-root $NATIVE --depth-resize $RESIZE"
fi
$PYTHON "$SCRIPTS/mvtracker_inference.py" --dataset-root "$DATASET_ROOT" --depth-root "$DATASET_ROOT" \
    --output-root "$OUT" --datasets dex_ycb --split test --checkpoint "$CKPT" --device "$DEVICE" \
    --cam-names view_00 view_01 view_02 view_03 --view-seed 0 --pose-file pose3d_cached.npz \
    --query-first --official-settings $DEPTH_ARGS
$PYTHON "$SCRIPTS/mvtracker_metrics.py" --dataset-root "$DATASET_ROOT" --output-root "$OUT" --datasets dex_ycb --split test --force
cd "$REPO"; $PIXI_PYTHON scripts/combine_metrics.py --prefix "$OUT"
