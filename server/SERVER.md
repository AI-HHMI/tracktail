# TrackerEncoder Inference Server

A FastAPI server that loads a single TrackerEncoder checkpoint and serves single-window predictions over HTTP.
Optionally, it also loads a ScorerEncoder (track-quality scorer) and serves it at `/score`.

## Starting the server

With no model-source arguments, the server loads the latest revision of
`ai-hhmi/posetail-static-animal` from the Hugging Face Hub:

```bash
python server/server.py
```

Other model sources are supported:

```bash
# A different Hugging Face model or a dated release
python server/server.py --hf-repo ai-hhmi/posetail-static --revision 2026-08-24

# From a wandb run directory (same as --base-folder in inference_video.py)
python server/server.py --wandb /path/to/wandb/run-YYYYMMDD_HHMMSS-XXXXXXXX

# With a specific checkpoint number (default: latest)
python server/server.py --wandb /path/to/wandb/run-... --checkpoint-number 10000

# From explicit config and checkpoint paths
python server/server.py --config /path/to/config.toml --checkpoint /path/to/checkpoint_00010000.pth

# Override device, host, or port (defaults: device from config / cpu, host 0.0.0.0, port 8000)
python server/server.py --hf-repo ai-hhmi/posetail-static-animal --device cuda:1 --port 8080

# Also load a track-quality scorer (enables POST /score)
python server/server.py --scorer-wandb /path/to/scorer/wandb/run-... [--scorer-checkpoint-number N]
python server/server.py --scorer-config scorer_config.toml --scorer-checkpoint scorer.pth
```

`--hf-repo`, `--wandb`, and `--config` are mutually exclusive.
- `--revision` only applies with `--hf-repo`.
- `--checkpoint-number` only applies with `--wandb`.
- `--config` and `--checkpoint` must be given together.

The scorer flags follow the same pattern: `--scorer-wandb` or `--scorer-config` with
`--scorer-checkpoint`. The scorer always runs on the tracker's device.

The server prints `n_frames`, `image_size`, and `device` on startup. Hub checkpoints are cached
by `huggingface_hub`; set `HF_TOKEN` when loading a private repository.

## Endpoints

### `GET /info`

Returns model metadata. Useful for clients to discover `n_frames` before sending images.

**Response** `200 application/json`

```json
{
  "n_frames": 24,
  "image_size": 256,
  "device": "cuda:0",
  "config_path": "hf://ai-hhmi/posetail-static-animal/config.toml@main",
  "checkpoint_path": "hf://ai-hhmi/posetail-static-animal/model.pth@main",
  "model_source": "huggingface:ai-hhmi/posetail-static-animal",
  "model_revision": null,
  "mode_3d": "encoder",
  "scorer_loaded": false,
  "scorer_config_path": null,
  "scorer_checkpoint_path": null
}
```

- `n_frames` is the clip length the model was trained on, which is the recommended number of
  frames per request.
- `image_size` is the long-side resolution that the server resizes frames to.
- `model_source` is `huggingface:<repo>`, `wandb:<run dir>`, or `local`.

---

### `POST /predict`

Runs one forward pass and returns all model outputs.

**Request** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | form field (JSON string) | Cameras, query coords, and optional query times (see below) |
| `images` | repeated file field | One image file per (camera, frame); filename must be `<cam_name>__<frame_idx>.<ext>` |

**`metadata` JSON schema**

```json
{
  "cameras": [
    {
      "name": "0",
      "type": "pinhole",
      "mat":    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "dist":   [k1, k2, p1, p2, k3],
      "ext":    [[r00,r01,r02,tx], [r10,r11,r12,ty], [r20,r21,r22,tz], [0,0,0,1]],
      "offset": [ox, oy]
    }
  ],
  "coords":      [[x, y, z], ...],
  "query_times": [t0, t1, ...]
}
```

- `cameras` — list of cameras in the order the model should consume them. Images are matched to cameras by `name`.
- `coords` — shape `(N, 3)`, 3D query points in world space (float).
- `query_times` — shape `(N,)`, integer frame each query point originates from (optional; defaults to zeros). This is an index into the uploaded clip (`0 … T-1`, frames in ascending `frame_idx` order). It is not the absolute `frame_idx` from the filename.
- `ext` — 4×4 world-to-camera extrinsic matrix. The server computes `ext_inv` and `center` internally.
- `offset` — pixel offset applied when the image is a crop of a larger frame; use `[0, 0]` for full-frame images.
- Image dimensions (`size`) are read from the uploaded pixel data — the client does not need to include them.

**Image files**

- Filename format: `<cam_name>__<frame_idx>.<ext>` (e.g. `0__000064.png`).
- Every camera listed in `cameras` must send the same number of frames `T`.
  - `T` must be even and at least the video encoder's tubelet size (2).
  - `T` doesn't have to equal `n_frames`, because the temporal embeddings are resized to the
    clip length. Sending `n_frames` (from `/info`) matches training and is recommended.
- Frames are ordered per camera by `frame_idx`. Cameras are aligned by position in that order,
  not by the `frame_idx` values, so send the same frame indices for every camera.
- Any format readable by OpenCV is accepted (PNG, JPEG, …).
- Images are decoded as BGR and converted to RGB internally; send them in their natural on-disk format.

**Preprocessing**

- Each camera's frames (and its intrinsics/offset) are resized so that the longer side equals
  `image_size`.
- The server does **not** crop. If the subject is small in a large frame:
  1. Crop around it on the client.
  2. Send the crops.
  3. Set `offset` to the crop's top-left corner in the calibrated frame.

  `example_client.py` does this (`crop_camera_group_to_queries`).
- One request is one window. To track through long videos, the client has to handle it:
  1. Split the video into consecutive windows.
  2. After each window, take the last frame of `coords_pred`.
  3. Send those positions as the next window's `coords`, with `query_times` set to 0.

  The offline pipeline (`run_tracker_encoder_on_videos` in
  `posetail/inference/inference_utils.py`, used by `scripts/inference_video.py`) does this
  windowing automatically, and also moves the crop with the subject.

**Response** `200 application/octet-stream` — a NumPy `.npz` archive (filename `predictions.npz`).

Decode with:
```python
import io, numpy as np
result = np.load(io.BytesIO(response.content))
```

Output keys (shapes use `B=1`, `T=n_frames`, `N=num_query_points`, `C=num_cameras`):

| Key | Shape | Description |
|-----|-------|-------------|
| `coords_pred` | `(B, T, N, 3)` | Final 3D predictions (same as `3d_pred_direct`) |
| `3d_pred_direct` | `(B, T, N, 3)` | 3D predictions via direct regression |
| `3d_pred_rays` | `(B, T, N, 3)` | 3D predictions via ray-based aggregation |
| `3d_pred_triangulate` | `(B, T, N, 3)` | 3D predictions via triangulation (absent for single-camera input) |
| `3d_pred_cams_direct` | `(C, B, T, N, 3)` | Per-camera direct 3D predictions |
| `3d_pred_cams_rays` | `(C, B, T, N, 3)` | Per-camera ray-based 3D predictions |
| `2d_pred` | `(C, B, T, N, 2)` | Per-camera 2D reprojections |
| `vis_pred` | `(B, T, N, 1)` | Visibility logits (noisy-OR across cameras: visible in ≥1 camera) |
| `conf_pred` | `(B, T, N)` | Confidence logits (max across cameras) |
| `vis_pred_2d` | `(C, B, T, N)` | Per-camera visibility logits |
| `conf_pred_2d` | `(C, B, T, N)` | Per-camera confidence logits |
| `conf_3d` | `(C, B, T, N)` | Per-camera 3D confidence logits |
| `depth_pred` | `(C, B, T, N)` | Per-camera depth predictions |

- `2d_pred` is returned in the pixel coordinates of the images you uploaded. It is un-scaled
  from the server's resize, but the crop offset is not removed.
- Logit outputs (`vis_*`, `conf_*`) need a sigmoid to become probabilities.
- Grid/classification checkpoints (`output_mode = grid | gridresid | gridnorm`, including the
  default `ai-hhmi/posetail-static-animal`) also produce training-only outputs (`2d_logits`,
  `grid`). These are never included in the response, so every key loads with plain `np.load`
  (no `allow_pickle`).
**Error responses** `400` — JSON `{"detail": "..."}` describing which validation failed, for example:
- missing `cameras` or `coords`
- missing camera images
- odd or mismatched frame counts
- malformed filename
- `query_times` length mismatch
- JSON parse error

---

### `POST /score`

Scores a full-sequence 3D trajectory with the optional ScorerEncoder. The server returns `404`
if it was started without `--scorer-wandb` or `--scorer-config`.

**Request** `multipart/form-data` — same `metadata` and `images` fields and rules as `/predict`,
except:

- `coords` is a trajectory of shape `(T, K, 3)` (world-space positions of `K` points at every
  uploaded frame), not `(N, 3)` query points.
- `T` must equal the number of uploaded frames per camera.
- `query_times` is ignored.

**Response** `200 application/octet-stream` — `.npz` (`scores.npz`) with keys `scores` and
`precision`, as returned by `ScorerEncoder.forward`.

---

## Example client

`server/example_client.py` is a runnable end-to-end smoke test. It reads a trial directory (same layout as `inference_video.py`), sends one window of frames to the server, and prints shapes and NaN counts for each output key.

```bash
python server/example_client.py \
    --trial-path /path/to/session/trial/ \
    --server-url http://localhost:8000 \
    --start-frame 0
```
