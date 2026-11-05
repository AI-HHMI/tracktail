# Tracktail

**Multi-view 3D point tracking without depth input.** You give Tracktail synchronized videos from
calibrated cameras and a set of 3D query points. It returns each point's 3D trajectory and
visibility over time.

To reproduce the paper, see the [`paper` branch](https://github.com/AI-HHMI/tracktail/tree/paper).

- Uses only RGB video and camera calibration. No depth sensor and no monocular depth model.
- Works with any number of cameras (1–N) and different rigs, because coordinates are normalized
  by a per-camera scene scale.
- Pretrained models on the Hugging Face Hub, a Python API, and an [HTTP inference server](docs/SERVER.md).

---

## Contents
1. [Installation](#installation)
2. [Python API](#python-api)
3. [Pretrained models](#pretrained-models)
4. [Conventions: cameras, coordinates, outputs](#conventions)
5. [Long videos](#long-videos)
6. [Visualization & evaluation](#visualization--evaluation)
7. [Inference server](#inference-server)
8. [Training](#training)
9. [Repository layout](#repository-layout)
10. [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/AI-HHMI/tracktail.git
cd tracktail
pixi install            # Python 3.11, CUDA 12.8 PyTorch, all deps
pixi shell              # or prefix commands with `pixi run`
```

Library only, without the server: `pip install tracktail`. You need to install PyTorch yourself.

**GPU.** Any recent NVIDIA GPU works. Peak memory on an A100, 96 frames at 256 px, 2–10 views
(paper speed benchmark, `benchmark_speed/` on the `paper` branch): about 2 GB for 10 query points, 3.5 GB for 100, and 20 GB for 1000 (up to
34 GB reserved by PyTorch at 10 views). Memory grows with the number of points, so lower
`max_kpts` (points per forward pass) on smaller GPUs. CPU works but is slow.

## Python API

```python
from tracktail.model.tracker_encoder import TrackerEncoder
from tracktail.inference.inference_utils import (
    load_camera_group_from_metadata, run_tracker_encoder_on_videos, run_inference)

model = TrackerEncoder.from_pretrained("ai-hhmi/tracktail-static-animal", revision="2026-10-12")

# a) arbitrary videos + calibration + query points
cams = load_camera_group_from_metadata("trial/metadata.yaml")
res = run_tracker_encoder_on_videos(model, video_paths, cams, queries_xyz,   # (N,3) tensor
                                    start_frame=0, n_frames=1000, query_times=None)
res["coords_pred"], res["vis_pred"], res["frame_numbers"]

# b) a trial directory (see layout below); optionally saves an .npz
run_inference(model, None, None, trial_path="trial/", n_frames=1000, outpath="out.npz")
```

`run_tracker_encoder_on_videos` takes care of windowing, re-anchoring queries between windows,
and a crop that follows the subject. Useful arguments:
- `query_times`: per-point query frame (relative to `start_frame`); each point starts at its own frame.
- `motion_margin` (default on): widen each window's crop by the previous window's motion so fast
  subjects stay in view.
- `max_kpts`: points per forward pass; lower it to save GPU memory.
- `vis_thresh` (default 0.1): visibility threshold for the occlusion state carried between windows.
- `run_inference(..., n_views=K, seed=S)` uses K randomly chosen cameras.

**Trial directory layout** (used by `run_inference`, `scripts/inference_video.py`, and
`server/example_client.py`):
```
trial/
  metadata.yaml   # intrinsic_matrices, extrinsic_matrices (world->cam), distortion_matrices,
                  # camera_heights, camera_widths, [offset_dict], [cam_type], [moving_cams]
  pose3d.npz      # 'pose' (subjects, T, kpts, 3): query source (and GT if present); optional 'vis'
  vid/<cam>.mp4   # or img/<cam>/*.png
```

## Pretrained models
| Repo | Trained on | Clip / res | Release |
|---|---|---|---|
| `ai-hhmi/tracktail-static-animal` (server default) | `tracktail-static`, further finetuned on animal datasets | 24 frames / 256 px | `2026-10-12` |
| `ai-hhmi/tracktail-static` | The paper model: trained on Multi-view Kubric + Multi-view Point Odyssey (human, animal) | 24 frames / 256 px | `2026-10-12` |

"static" means static (calibrated, non-moving) cameras. Use `tracktail-static-animal` for animal
recordings; use `tracktail-static` for general scenes and to compare with the paper's numbers.

## Conventions
- **Extrinsics** are world→camera 4×4 matrices. **Units** can be anything, because the model
  normalizes by a per-camera scene scale `s_c = 1 / median σ_max(J_c)`.
- **Logits:** `vis_*` and `conf_*` outputs are logits, so apply a sigmoid. For visibility use a
  threshold of 0.1 (`tracktail.model.eval_metrics.TRACKTAIL_VIS_THRESH`; used in the paper).
- **Camera order:** in `metadata.yaml` cameras are sorted by name (numerically if every name is a
  number).
- **Moving cameras:** `moving_cams: true` with (T,4,4) extrinsics (Python API only).

## Long videos
The model sees 24 frames at a time. The Python API and `scripts/inference_video.py` track through
longer videos automatically: they run overlapping windows, start each window from the previous
window's predictions, and move the crop with the subject. With the server, you do the windowing on
the client (see [docs/SERVER.md](docs/SERVER.md)).

```bash
pixi run python scripts/inference_video.py --help     # windowed inference over a whole trial
```

## Visualization & evaluation
- `scripts/render_video.py --input-npz out.npz --output-dir viz/ [--crop --trails]`: draws
  predictions over the video.
- `scripts/viz3d.py`: 3D viewer built on [rerun](https://rerun.io). It's a function library,
  so import it in a notebook.
- `scripts/inference_metrics.py`: computes AJ, δ_avg, OA and MTE using
  `tracktail.model.eval_metrics`.

## Inference server

`server/server.py` serves the model over HTTP, one clip per request, so you can call it from any
language or another machine:

```bash
pixi run python server/server.py      # downloads ai-hhmi/tracktail-static-animal, serves on :8000
```

See [docs/SERVER.md](docs/SERVER.md) for the options, the request format, the outputs, and a
Python client.

---

## Training

See [docs/TRAINING.md](docs/TRAINING.md). The configs behind the paper's model are on the
[`paper` branch](https://github.com/AI-HHMI/tracktail/tree/paper).

---

## Repository layout
```
tracktail/model/      model (tracker_encoder.py, encoder_decoder.py, vjepa2.py, cube.py), losses, metrics, train_utils
tracktail/datasets/   training (tracktail_dataset.py) and inference datasets, scorer corruptions
tracktail/inference/  inference_utils.py (loading, windowed tracking, trial I/O)
server/              FastAPI inference server + example client (docs/SERVER.md)
docs/                server and training docs
scripts/             inference / rendering / metrics / packaging CLIs
configs/             training configs (TOML)
cluster/             LSF launch scripts
```

## Citation
<!-- TODO: BibTeX once public -->
