#!/usr/bin/env bash
# Clone the baseline repos at the commits used for the paper and download their weights.
#
#   bash baselines/setup.sh                  # into ./external
#   BASELINE_ROOT=/data/baselines bash baselines/setup.sh
#
# Needs git, curl and sha256sum. Safe to re-run: existing clones and files are kept (and checked).
# It does not create the Python environments; see baselines/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="${BASELINE_ROOT:-$PWD/external}"
mkdir -p "$BASELINE_ROOT/weights"
BASELINE_ROOT="$(cd "$BASELINE_ROOT" && pwd)"
W="$BASELINE_ROOT/weights"

# name  url  commit
REPOS=(
  "mvtracker  https://github.com/ethz-vlg/mvtracker.git          ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"
  "co-tracker https://github.com/facebookresearch/co-tracker.git 82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
  "dust3r     https://github.com/naver/dust3r.git                4c24a6ebf04809f2cfe59915e51779c8984aaa40"
  "arrow      https://github.com/ifradlin/arrow.git              6fca6f832facf74250ecb64f978f3cb743be327d"
)

for line in "${REPOS[@]}"; do
  read -r name url commit <<<"$line"
  dir="$BASELINE_ROOT/$name"
  if [[ ! -d "$dir/.git" ]]; then
    echo "== cloning $name"
    git clone --quiet "$url" "$dir"
  fi
  if [[ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
  fi
  git -C "$dir" submodule update --init --recursive --quiet   # dust3r: croco @ d7de070
  patch="$HERE/patches/$name.patch"
  if [[ -f "$patch" ]]; then
    if git -C "$dir" apply --check "$patch" 2>/dev/null; then
      git -C "$dir" apply "$patch"; echo "   applied patches/$name.patch"
    elif git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
      :   # already applied
    else
      echo "ERROR: patches/$name.patch does not apply to $dir" >&2; exit 1
    fi
  fi
  echo "$name @ $(git -C "$dir" rev-parse --short HEAD)"
done

# file  url  sha256
WEIGHTS=(
  "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth 9f194a32e34fc124ed10f64115b8e848b7da87c2d1ed131d34b4806721c19dbd"
  "mvtracker_200000_june2025.pth https://huggingface.co/ethz-vlg/mvtracker/resolve/010d5d114e860aae6b2568104927b636cdca01bc/mvtracker_200000_june2025.pth a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6"
  "scaled_offline.pth https://huggingface.co/facebook/cotracker3/resolve/bf55ea50d4390e1820a267f131cd6587240fb2c5/scaled_offline.pth 2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834"
  "arrow/model.safetensors https://huggingface.co/ifradlin/arrow/resolve/fd2b2f74c49990f739264a6aee786029227a93d4/model.safetensors 113b6f83da69fdc3f03dc623e606da0c346a1a13d796bf9ee07dc1abf07aaf25"
  "arrow/config.json https://huggingface.co/ifradlin/arrow/resolve/fd2b2f74c49990f739264a6aee786029227a93d4/config.json 84c0599e077b5971467717c307efa0968ad088b4339249bd4f7bafa23e4f9b5a"
)

for line in "${WEIGHTS[@]}"; do
  read -r file url sum <<<"$line"
  dest="$W/$file"
  mkdir -p "$(dirname "$dest")"
  if [[ ! -f "$dest" ]]; then
    echo "== downloading $file"
    curl -L --fail --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi
  if ! echo "$sum  $dest" | sha256sum --check --status; then
    echo "WARNING: sha256 of $dest differs from the file used for the paper ($sum)" >&2
  fi
done

cat <<EOF

Done. Export these (or put them in paper.env):

export BASELINE_ROOT=$BASELINE_ROOT
export MVT_HUB_DIR=\$BASELINE_ROOT/mvtracker          # torch.hub.load(..., source="local"): pins MVTracker to ceea8ad
export MVTRACKER_REPO=\$BASELINE_ROOT/mvtracker
export COTRACKER_REPO=\$BASELINE_ROOT/co-tracker
export DUST3R_REPO=\$BASELINE_ROOT/dust3r
export ARROW_REPO=\$BASELINE_ROOT/arrow
export MVTRACKER_CKPT=\$BASELINE_ROOT/weights/mvtracker_200000_june2025.pth
export DUST3R_CKPT=\$BASELINE_ROOT/weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
export COTRACKER_CKPT=\$BASELINE_ROOT/weights/scaled_offline.pth
export ARROW_CKPT=\$BASELINE_ROOT/weights/arrow
# Python environments (see baselines/README.md):
export MVTRACKER_PYTHON=<conda env>/bin/python        # MVTracker + DUSt3R
export DUST3R_PYTHON=\$MVTRACKER_PYTHON
export ARROW_PYTHON=\$ARROW_REPO/.venv/bin/python      # cd \$ARROW_REPO && uv sync
EOF
