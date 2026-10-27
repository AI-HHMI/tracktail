#!/usr/bin/env bash
#
# Tracktail on the TAPVid-3D datasets (tapvid3d_adt, tapvid3d_drivetrack, tapvid3d_pstudio): the full test sets, there
# are no cached tracks. One bsub job runs scripts/eval_testset_metrics.py on the datasets one after the other (query_first
# inference, the paper-protocol metrics and the depth-relative TAPVid-3D metrics) and, after each dataset, rebuilds
# summary.json, metrics.md and summary.csv (one row per dataset) over the datasets that are finished.
#
# Submit all datasets:        bash tracktail_tapvid3d.sh
# Submit a subset by name:    bash tracktail_tapvid3d.sh tapvid3d_pstudio tapvid3d_adt
# Another network:            NAME=main_pretrain_e25w9v86 WANDB_RUN=<its wandb folder> bash tracktail_tapvid3d.sh
# Worker (run by bsub):       bash tracktail_tapvid3d.sh --worker <dataset> [dataset ...]
#
# The job runs the checkout in /groups (REPO), also when this is submitted from another copy: run update.sh first so that it
# has this script, summary_to_csv.py and the current evaluator.
#
# Results go to $OUT_BASE/<name>_tapvid3d/ . Predictions are kept in <dataset>/ and reused, so a job that hit its time
# limit is simply submitted again and continues where it stopped. The datasets run smallest first. A trial is one camera of
# one sequence (tapvid3d_pstudio 156 trials, tapvid3d_adt 1931, tapvid3d_drivetrack 2457): tapvid3d_pstudio takes about
# 3 s per trial (measured, 8 minutes in all), the other two are extrapolated from that by frames x tracks (about 2 h for
# drivetrack, about 12 h for adt), so the 24 h limit below should hold the three; raise WALL if it does not.

LOGS=/groups/karashchuk/home/ruppk2/results/posetail-ablations-logs
WANDB_BASE=/groups/karashchuk/home/ruppk2/results/posetail-pretraining-v6/wandb
OUT_BASE=${OUT_BASE:-/groups/karashchuk/home/ruppk2/results/paper-final-v2/posetail-outputs}
REPO=${REPO:-/groups/karashchuk/home/ruppk2/github/posetail}   # the checkout the job runs; update it first (update.sh)

NAME=${NAME:-main_finetune_vzkjqrys}
WANDB_RUN=${WANDB_RUN:-$WANDB_BASE/run-20260912_115732-vzkjqrys}
OUT=$OUT_BASE/${NAME}_tapvid3d

DATASETS="tapvid3d_pstudio tapvid3d_drivetrack tapvid3d_adt"
QUEUE=${QUEUE:-gpu_h100}
WALL=${WALL:-24:00}

if [[ "$1" != "--worker" ]]; then
    # the job runs $REPO: it has to have the paper-protocol evaluator (an older checkout does not)
    if ! grep -q eval_outputs_paper "$REPO/scripts/eval_testset_metrics.py"; then
        echo "$REPO/scripts/eval_testset_metrics.py has no eval_outputs_paper: update that checkout (git pull) or set REPO"
        exit 1
    fi
    mkdir -p "$LOGS"
    bsub -J "${NAME}_tapvid3d" -W "$WALL" -e $LOGS/%J.err -o $LOGS/%J.out \
        -n 4 -q "$QUEUE" -gpu "num=1" \
        bash "$REPO/scripts/$(basename "${BASH_SOURCE[0]}")" --worker ${@:-$DATASETS}
    exit 0
fi

# ---------------- worker ----------------
set -e
shift

source ~/.setup_env_vars.sh
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$PYTHONPATH"
export LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:$LD_LIBRARY_PATH"

cd "$REPO"
if [[ -x .pixi/envs/default/bin/python ]]; then PYTHON=.pixi/envs/default/bin/python; else PYTHON="pixi run python"; fi

mkdir -p "$OUT"
for DS in "$@"; do
    echo "starting tracktail eval: $NAME / $DS -> $OUT"
    $PYTHON scripts/eval_testset_metrics.py \
        --wandb-folder "$WANDB_RUN" \
        --out           "$OUT" \
        --datasets      "$DS" \
        --device        "${DEVICE:-cuda:0}"

    # every call above rewrites summary.json with its own dataset only: rebuild it over the datasets that are finished,
    # and write summary.csv (one row per dataset)
    $PYTHON scripts/merge_eval_summaries.py "$OUT"
    $PYTHON scripts/summary_to_csv.py "$OUT"
done

echo "done!"
