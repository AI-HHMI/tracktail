#!/usr/bin/env bash
# Run bench.py for every method, each in its own environment, one after the other on the same GPU.
#
# Usage:
#   bash benchmark_speed/run_all.sh [OUT_DIR] [METHODS]
#   METHODS default: "tracktail cotracker mvtracker-gt mvtracker-dust3r arrow"
# Extra arguments for bench.py can be passed with BENCH_ARGS, e.g. BENCH_ARGS="--n-frames 48 --repeats 1".

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/.."
OUT_DIR=${1:-"$HERE/results/$(date +%Y%m%d_%H%M%S)"}
METHODS=${2:-"tracktail cotracker mvtracker-gt mvtracker-dust3r arrow"}
BENCH_ARGS=${BENCH_ARGS:-}
mkdir -p "$OUT_DIR"

[[ -f ~/.setup_env_vars.sh ]] && source ~/.setup_env_vars.sh
nvidia-smi | tee "$OUT_DIR/nvidia-smi.txt"

# everything on /groups so it is visible from the cluster nodes (the /home/<user>@hhmi.org dirs are workstation-local)
BUILDS=/groups/karashchuk/home/karashchukl/builds     # co-tracker, mvtracker sources; arrow-venv; arrow-weights
PIXI_PY="$REPO/.pixi/envs/default/bin/python"
MVT_ENV="/groups/karashchuk/home/ruppk2/miniforge3/envs/mvtracker"
MVT_PY="$MVT_ENV/bin/python"   # has roma for DUSt3R
ARROW_PY="$BUILDS/arrow-venv/bin/python"     # python 3.13.9 copy + ../arrow/.venv site-packages

for METHOD in $METHODS; do
    echo "=== $METHOD ==="
    case $METHOD in
        tracktail)
            env PYTHONPATH="/groups/karashchuk/home/karashchukl/builds/vjepa2:$REPO" \
                LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
                "$PIXI_PY" "$HERE/bench.py" --method $METHOD --out "$OUT_DIR/$METHOD.jsonl" $BENCH_ARGS ;;
        cotracker)
            env PYTHONPATH="$BUILDS/co-tracker:$REPO" \
                LD_LIBRARY_PATH="$REPO/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}" \
                "$PIXI_PY" "$HERE/bench.py" --method $METHOD --out "$OUT_DIR/$METHOD.jsonl" $BENCH_ARGS ;;
        mvtracker-gt|mvtracker-dust3r)
            # The env libicui18n needs CXXABI_1.3.15; nodes otherwise resolve the older system libstdc++.
            # This is the runtime-library setup from conda activation, applied directly to avoid libmamba
            # cache parse failures when both benchmark methods are launched at once.
            env PYTHONPATH="$BUILDS/mvtracker:/groups/karashchuk/home/ruppk2/github/dust3r" \
                LD_LIBRARY_PATH="$MVT_ENV/lib:${LD_LIBRARY_PATH:-}" \
                "$MVT_PY" "$HERE/bench.py" --method $METHOD --out "$OUT_DIR/$METHOD.jsonl" $BENCH_ARGS ;;
        arrow)
            "$ARROW_PY" "$HERE/bench.py" --method $METHOD --out "$OUT_DIR/$METHOD.jsonl" $BENCH_ARGS ;;
        *) echo "unknown method $METHOD"; exit 1 ;;
    esac 2>&1 | tee "$OUT_DIR/$METHOD.log"
done

"$PIXI_PY" "$HERE/summarize.py" "$OUT_DIR"
