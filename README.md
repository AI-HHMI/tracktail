# Tracktail

<!-- OUTLINE DRAFT — sections marked TODO still need content. Python package / PyPI name: `tracktail`. -->

**Multi-view 3D point tracking without depth input.** You give Tracktail synchronized videos from
calibrated cameras and a set of 3D query points. It returns each point's 3D trajectory and
visibility over time.

<!-- TODO: teaser figure (paper Fig. 1, 3D-POP) + link to paper / project page -->

- Uses only RGB video and camera calibration. No depth sensor and no monocular depth model.
- Works with any number of cameras (1–N) and different rigs, because coordinates are normalized
  by a per-camera scene scale.
- Ships with a pretrained checkpoint on the Hugging Face Hub and an HTTP inference server.

---

## Contents
1. [Installation](#installation)
2. [Quick start: inference server](#quick-start-inference-server)
3. [HTTP API reference](#http-api-reference)
4. [Python API](#python-api)
5. [Pretrained models](#pretrained-models)
6. [Conventions: cameras, coordinates, outputs](#conventions)
7. [Long videos](#long-videos)
8. [Visualization & evaluation](#visualization--evaluation)
9. [Training](#training)
10. [Repository layout](#repository-layout)
11. [Citation](#citation)

---

## Installation

```bash
git clone git@github.com:AI-HHMI/tracktail.git
cd tracktail
pixi install            # Python 3.11, CUDA 12.8 PyTorch, all deps
pixi shell              # or prefix commands with `pixi run`
```

Library only, without the server: `pip install tracktail`. You need to install PyTorch yourself.
<!-- TODO: GPU / VRAM requirements; CPU works but is slow -->

## Quick start: inference server

```bash
pixi run python server/server.py               # downloads ai-hhmi/posetail-static-animal, serves on :8000
curl localhost:8000/info                       # n_frames, image_size, device, model source
```

Minimal Python client:

```python
import io, json, cv2, numpy as np, requests

meta = {
    "cameras": [  # one entry per camera, in the order the model should use them
        {"name": "cam0", "mat": K0.tolist(), "dist": d0.tolist(),
         "ext": E0.tolist(), "offset": [0, 0]},
        ...
    ],
    "coords": queries_xyz.tolist(),        # (N, 3) world coordinates
    "query_times": query_t.tolist(),       # (N,) frame index within this clip (optional, default 0)
}
files = [("images", (f"{cam}__{t:06d}.jpg", cv2.imencode(".jpg", frame)[1].tobytes()))
         for cam, frames in clips.items() for t, frame in enumerate(frames)]
r = requests.post("http://localhost:8000/predict",
                  data={"metadata": json.dumps(meta)}, files=files)
out = np.load(io.BytesIO(r.content))
xyz = out["coords_pred"][0]                          # (T, N, 3)
visible = 1 / (1 + np.exp(-out["vis_pred"][0, ..., 0])) > 0.5
```

`server/example_client.py` is an end-to-end smoke test that reads a trial directory, sends one
window, and prints the output shapes.

### Server options
| Flag | Purpose |
|---|---|
| `--hf-repo REPO [--revision TAG]` | Load from the HF Hub (default `ai-hhmi/posetail-static-animal`). Pin a dated release, e.g. `2026-08-24`. Set `HF_TOKEN` for private repos. |
| `--wandb RUN_DIR [--checkpoint-number N]` | Load a local training run (`files/config.toml` + `files/checkpoints/`) |
| `--config C.toml --checkpoint X.pth` | Explicit config and checkpoint paths |
| `--device cuda:0`, `--host`, `--port` | Runtime |
| `--scorer-wandb` / `--scorer-config --scorer-checkpoint` | Optional track-quality scorer; enables `/score` |

Only one request uses the GPU at a time (enforced by an asyncio lock). <!-- TODO: throughput numbers -->

## HTTP API reference

### `GET /info`
Returns `n_frames`, `image_size`, `device`, `model_source`, `model_revision`, `config_path`,
`checkpoint_path`, `mode_3d`, `scorer_loaded`, and the scorer paths.

### `POST /predict`
Runs one forward pass over one clip. The request is `multipart/form-data`:

- **`metadata`** (JSON string)
  - `cameras[]`:
    - `name`
    - `type` (`pinhole` | `fisheye`)
    - `mat` (3×3 K)
    - `dist` (`[k1,k2,p1,p2,k3]`)
    - `ext` (4×4 world→camera)
    - `offset` (`[ox, oy]` if the images are crops of the calibrated frame)
  - `coords`: (N,3) world-space query points
  - `query_times`: (N,) optional
- **`images`**: one file per (camera, frame), named `<cam_name>__<frame_idx>.<ext>`.
  - Any format OpenCV can decode.
  - Every camera must send the same number of frames. The clip length must be even and at
    least 2. Use `n_frames` from `/info` to match training.
  - Frames are ordered by `frame_idx`, and `query_times` index into that order.
- Images are resized server-side so the longer side equals `image_size`. They aren't cropped,
  so crop around the subject yourself and pass `offset`.

**Response:** an `.npz` file (`B=1`, `T` frames, `N` points, `C` cameras):

| Key | Shape | Meaning |
|---|---|---|
| `coords_pred` | (B,T,N,3) | **Final 3D prediction** (confidence-fused across cameras) |
| `vis_pred` | (B,T,N,1) | Visibility logit (visible in ≥1 camera) |
| `conf_pred` | (B,T,N) | 3D confidence logit |
| `2d_pred` | (C,B,T,N,2) | Per-camera 2D prediction, in uploaded-image pixels |
| `vis_pred_2d`, `conf_pred_2d`, `conf_3d`, `depth_pred` | (C,B,T,N) | Per-camera visibility / 2D confidence / 3D confidence / depth |
| `3d_pred_direct`, `3d_pred_rays`, `3d_pred_triangulate` | (B,T,N,3) | Alternative 3D estimates |
| `3d_pred_cams_direct`, `3d_pred_cams_rays` | (C,B,T,N,3) | Per-camera 3D estimates |

Validation errors return `400` with JSON `{"detail": "..."}`. Training-only outputs of grid-head models (`2d_logits`, `grid`) are never included in the response.

### `POST /score` (optional)
Takes a full trajectory `coords` (T,K,3) plus the same cameras and images. Returns `scores` and
`precision` from the track-quality scorer. <!-- TODO: describe what the score means -->

## Python API

```python
from tracktail.model.tracker_encoder import TrackerEncoder
from tracktail.inference.inference_utils import (
    load_camera_group_from_metadata, run_tracker_encoder_on_videos, run_inference)

model = TrackerEncoder.from_pretrained("ai-hhmi/posetail-static-animal", revision="2026-08-24")

# a) arbitrary videos + calibration + query points
cams = load_camera_group_from_metadata("trial/metadata.yaml")
res = run_tracker_encoder_on_videos(model, video_paths, cams, queries_xyz,   # (N,3) tensor
                                    start_frame=0, n_frames=1000, query_times=None)
res["coords_pred"], res["vis_pred"], res["frame_numbers"]

# b) a trial directory (see layout below); optionally saves an .npz
run_inference(model, None, None, trial_path="trial/", n_frames=1000, outpath="out.npz")
```

`run_tracker_encoder_on_videos` takes care of windowing, re-anchoring queries between windows,
and a crop that follows the subject. <!-- TODO: document query_first / motion_margin / max_kpts / n_views -->

**Trial directory layout** (used by `run_inference`, `scripts/inference_video.py`, and
`example_client.py`):
```
trial/
  metadata.yaml   # intrinsic_matrices, extrinsic_matrices (world->cam), distortion_matrices,
                  # camera_heights, camera_widths, [offset_dict], [cam_type], [moving_cams]
  pose3d.npz      # 'pose' (subjects, T, kpts, 3): query source (and GT if present); optional 'vis'
  vid/<cam>.mp4   # or img/<cam>/*.png
```
<!-- TODO: converter from aniposelib calibration.toml -> metadata.yaml; a way to pass queries without pose3d.npz -->

## Pretrained models
| Repo | Trained on | Clip / res | Release |
|---|---|---|---|
| `ai-hhmi/posetail-static-animal` (server default) | TODO | 24 frames / 256 px | `2026-08-24` |
| `ai-hhmi/posetail-static` | TODO | TODO | `2026-08-24` |

<!-- TODO: what "static" means (static cameras); animal vs general; expected accuracy -->

## Conventions
- **Extrinsics** are world→camera 4×4 matrices. **Units** can be anything, because the model
  normalizes by a per-camera scene scale `s_c = 1 / median σ_max(J_c)`.
- **Logits:** `vis_*` and `conf_*` outputs are logits, so apply a sigmoid. A threshold of 0.5
  is a reasonable default.
- **Camera order:** in the server it's the order of `cameras`. In `metadata.yaml` cameras are
  sorted by name (numerically if every name is a number).
- **Moving cameras:** `moving_cams: true` with (T,4,4) extrinsics. Python API only.

## Long videos
The server processes one clip per request. For longer videos you have two options:
- Use the Python API or `scripts/inference_video.py`, which handle windowing automatically.
- Chunk on the client: send overlapping windows and use the last predicted frame as the next
  window's `coords` (with `query_times=0`). <!-- TODO: client helper / recipe -->

## Visualization & evaluation
- `scripts/render_video.py --input-npz out.npz --output-dir viz/ [--crop --trails]`: draws
  predictions over the video.
- `scripts/viz3d.py`: 3D viewer built on [rerun](https://rerun.io). It's a function library,
  so import it in a notebook.
- `scripts/inference_metrics.py`: computes AJ, δ_avg, OA and MTE using
  `tracktail.model.eval_metrics`.
- `cotracker_baseline/`: per-view CoTracker3 followed by triangulation, used as a paper
  baseline. It needs the `cotracker` package installed separately.

---

## Training

### Method overview
- **Video encoder:** V-JEPA 2.1 ViT-L/16 with hierarchical features, run on each view.
- **Query encoder:** builds a per-camera embedding from the 2D projection, a 9×9 patch, the
  times, the normalized depth, and an in-bounds flag.
- **Decoder:** N layers, each with camera self-attention (using PRoPE ray encodings),
  query→scene cross-attention, and an FFN.
- **Heads:** 3D/2D position, depth, visibility, and 2D/3D confidence. Per-camera predictions are
  fused by confidence.
<!-- TODO: architecture figure (paper Fig. 3); reconcile layer count / frames / head type with v6 config -->

### Data
- Data lives under `<prefix>/<dataset>/<split>/<session>/<trial>/`, with each trial in the same
  layout as above (`metadata.yaml`, `pose3d.npz`/`pose2d.npz`, `img/` or `vid/`).
- Datasets: Multi-View Kubric, Multiview Point Odyssey (Human/Animal), Panoptic Studio, DexYCB,
  plus 2D-only sets. <!-- TODO: full list + download/conversion scripts -->
- Sampling (cameras, keypoints, crops, augmentation, dataset balancing) is set in
  `[dataset.train]` in the config.

### Configs
- Pretrain: `configs/<TODO pretrain-v6>.toml`.
- Finetune: `configs/config_tracktail_finetuning_v6.toml`.
- Scorer: `configs/config_scorer.toml`.
<!-- TODO: move stale configs to configs/archive/; commit the pretrain-v6 config -->

Key sections: `[model]`, `[dataset.*]`, `[training]` (optimizer: AdamW, schedule-free, or
Muon; losses; `checkpoint_path`/`finetune`), and `[wandb]`. Iteration counts are **totals
across all GPUs**.

### Launching
```bash
pixi run python train.py --config-path configs/config_tracktail_finetuning_v6.toml \
    --devices -1 --strategy ddp_find_unused_parameters_true      # Lightning Fabric DDP
# LSF cluster: cluster/big_bsub_{pretraining,finetuning}.sh -> cluster/train_script.sh
```
- **Outputs:** `<wandb.path>/wandb/run-*/files/{config.toml, checkpoints/checkpoint_XXXXXXXX.pth}`.
- **Resuming:** set `training.checkpoint_path` and `finetune = false`. To warm-start from
  another stage, use `finetune = true`.
- **Scorer:** `train_scorer.py`.

### Releasing a checkpoint
`scripts/package_checkpoint.py` strips a run down to the eval weights plus the model config,
producing `model.pth`. Upload it with `hf upload ai-hhmi/<repo> model.pth` and tag a dated
revision. <!-- TODO: script the upload -->

---

## Repository layout
```
tracktail/model/      model (tracker_encoder.py, encoder_decoder.py, vjepa2.py, cube.py), losses, metrics, train_utils
tracktail/datasets/   training (tracktail_dataset.py) and inference datasets, scorer corruptions
tracktail/inference/  inference_utils.py (loading, windowed tracking, trial I/O)
server/              FastAPI inference server + example client
scripts/             inference / rendering / metrics / packaging CLIs
configs/             training configs (TOML)
cluster/             LSF launch scripts
cotracker_baseline/  CoTracker3 + triangulation baseline
```

## Citation
<!-- TODO: BibTeX once public -->
