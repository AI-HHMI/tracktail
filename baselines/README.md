# Baselines

The paper compares Tracktail with MVTracker (DUSt3R or GT depth), CoTracker3 + triangulation, and ARROW.
The drivers are in `mvtracker_baseline/`, `cotracker_baseline/`, `arrow_baseline/` and `benchmark_speed/`.

## 1. Code and weights

```bash
bash baselines/setup.sh                       # clones into ./external
BASELINE_ROOT=/data/baselines bash baselines/setup.sh
```

`setup.sh` clones each repo at the commit used for the paper, applies `baselines/patches/<repo>.patch`,
downloads the weights into `$BASELINE_ROOT/weights/`, checks their sha256, and prints the env vars to export.

| Repo | URL | Commit |
|---|---|---|
| mvtracker | https://github.com/ethz-vlg/mvtracker | `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072` |
| co-tracker | https://github.com/facebookresearch/co-tracker | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| dust3r | https://github.com/naver/dust3r | `4c24a6ebf04809f2cfe59915e51779c8984aaa40` (submodule croco `d7de070`) + `patches/dust3r.patch` |
| arrow | https://github.com/ifradlin/arrow | `6fca6f832facf74250ecb64f978f3cb743be327d` |

`patches/dust3r.patch` is one line: `torch.load(..., weights_only=False)` in `dust3r/model.py`, needed with torch >= 2.6.
The other clones had no local changes.

| Weights | Source (pinned revision) | sha256 |
|---|---|---|
| `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth | `9f194a32…9dbd` |
| `mvtracker_200000_june2025.pth` | HF `ethz-vlg/mvtracker` @ `010d5d11` | `a7fa86f2…f8e6` |
| `scaled_offline.pth` (CoTracker3 offline) | HF `facebook/cotracker3` @ `bf55ea50` | `2670d456…7834` |
| `arrow/{model.safetensors,config.json}` | HF `ifradlin/arrow` @ `fd2b2f74c49990f739264a6aee786029227a93d4` | `113b6f83…af25` |

Full hashes are in `setup.sh`. They match the files used for the paper runs.

MVTracker is built with `torch.hub.load(..., "mvtracker_predictor", pretrained=False)` and the checkpoint is
loaded afterwards. Set `MVT_HUB_DIR=$BASELINE_ROOT/mvtracker` so `torch.hub` uses the pinned clone
(`source="local"`). Without it, `torch.hub` fetches the current GitHub `main` of `ethz-vlg/mvtracker`.

## 2. Python environments

Three environments were used. Versions are from the environments of the paper runs.

**CoTracker3**: this repo's pixi env (Python 3.11, torch 2.11.0+cu128). CoTracker is not installed;
the drivers put `$COTRACKER_REPO` on `PYTHONPATH`.

```bash
pixi install
PYTHONPATH=$COTRACKER_REPO:$PWD pixi run python cotracker_baseline/cotracker_inference.py --help
```

**MVTracker + DUSt3R**: a conda env with Python 3.10.12, torch 2.11.0+cu128, numpy 1.26.4, kornia 0.7.3,
roma 1.6.1, einops 0.7.0. (The MVTracker inference runs used a venv on top of
an older conda env: Python 3.10, torch 2.10.0, kornia 0.8.2.) The scripts also import `tracktail` from this repo,
so it needs this repo's runtime deps (yaml, opencv, scipy, tqdm).

```bash
conda create -n mvtracker python=3.10.12 -y
conda activate mvtracker
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r $MVTRACKER_REPO/requirements.txt          # its numpy/hf-hub pins may be relaxed: numpy 1.26.4 was used
pip install roma scipy trimesh pyyaml tqdm                # DUSt3R + tracktail imports
export MVTRACKER_PYTHON=$(which python) DUST3R_PYTHON=$(which python)
```

The optional DUSt3R `curope` CUDA kernel was not compiled for the paper runs (DUSt3R falls back to its PyTorch RoPE).

**ARROW**: ARROW's own uv env from its `uv.lock` (Python 3.13, torch 2.14.1+cu130, numpy 2.5.3).

```bash
(cd $ARROW_REPO && uv sync)
export ARROW_PYTHON=$ARROW_REPO/.venv/bin/python
```

## 3. Env vars read by the drivers

| Variable | Default after `setup.sh` | Used by |
|---|---|---|
| `BASELINE_ROOT` | `./external` | all of the below |
| `MVT_HUB_DIR` | `$BASELINE_ROOT/mvtracker` | `mvtracker_inference.py` (torch.hub local source) |
| `MVTRACKER_REPO` | `$BASELINE_ROOT/mvtracker` | `PYTHONPATH` of the MVTracker drivers, `benchmark_speed` |
| `DUST3R_REPO` | `$BASELINE_ROOT/dust3r` | `PYTHONPATH` of the DUSt3R precompute + MVTracker drivers |
| `COTRACKER_REPO` | `$BASELINE_ROOT/co-tracker` | `PYTHONPATH` of the CoTracker drivers, `benchmark_speed` |
| `ARROW_REPO` | `$BASELINE_ROOT/arrow` | default location of `ARROW_PYTHON` |
| `MVTRACKER_CKPT` | `$BASELINE_ROOT/weights/mvtracker_200000_june2025.pth` | MVTracker drivers, `bench.py` |
| `DUST3R_CKPT` | `$BASELINE_ROOT/weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth` | DUSt3R precompute, `bench.py` |
| `COTRACKER_CKPT` | `$BASELINE_ROOT/weights/scaled_offline.pth` | CoTracker drivers, `bench.py` |
| `ARROW_CKPT` | `$BASELINE_ROOT/weights/arrow` | ARROW drivers, `bench.py` |
| `MVTRACKER_PYTHON`, `DUST3R_PYTHON` | — | MVTracker / DUSt3R drivers |
| `ARROW_PYTHON` | `$ARROW_REPO/.venv/bin/python` | ARROW drivers |

`ARROW_CKPT` can also be the hub id `ifradlin/arrow`, but that takes the latest revision.
