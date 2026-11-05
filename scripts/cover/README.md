# Rat7M cover figure (`fig:rat7m`)

Builds the dense rat point cloud, tracks it with Tracktail, and renders the cover panels.
Paper settings: Rat7M test trial `s5-d1/101500`, query frame 670, 360 frames (3 s at 120 fps).

1. **Cloud** (`cover_rat7m_demo.py --stages cloud`):
   median-background mask → SAM ViT-L masks per camera → voxel visual hull →
   VGGT-1B depth on crops around the rat → align each depth map to the hull (robust affine fit) →
   back-project with the Rat7M calibration and keep multi-view-consistent points.
   Alignment and fusion are repeated 3 times, then the cloud is voxel-downsampled (paper: 5,668 points).
2. **Track** (`--stages track`): track every cloud point with Tracktail
   (`ai-hhmi/tracktail-static`, revision `2026-10-12`, from the Hugging Face Hub) for 360 frames.
3. **Figure** (`--stages figure`): camera-view panels with trails, camera frustums, input grid, and 3D time-lapse renders.
   `render_timelapse.py` re-renders the time-lapse panel used on the right side of the cover with the paper settings
   (poses at 0, 0.95 and 2.1 s; 200 tracks).

Needs a CUDA GPU for the cloud and track stages. The figure stage runs on CPU.

## Extra dependencies

These are not in the pixi env. Install them into a separate directory and pass it with `--extra-path`:

```bash
pixi run python -m pip install --no-deps --target deps_cover \
    "vggt @ git+https://github.com/facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce" \
    "segment-anything @ git+https://github.com/facebookresearch/segment-anything@dca509fe793f601edb92606367a655c15ac00fdf"
```

- `vggt` 0.0.1 (commit above). The `facebook/VGGT-1B` weights are downloaded from the Hugging Face Hub on first use.
  VGGT also needs `safetensors`, `einops` and `huggingface_hub`. These are already in the pixi env (safetensors 0.8.0 was used).
- `segment-anything` 1.0 (commit above). Uses the SAM ViT-L checkpoint
  <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>. It is downloaded to the torch hub cache
  automatically, or you can pass it with `--sam-checkpoint`.
- `render_timelapse.py` needs ghostscript (`gs`) on `PATH` to compress the PDF.

## Input

You need one Rat7M trial in the Tracktail trial layout (`metadata.yaml` with the calibration, plus `vid/<camera>.mp4`).
Download Rat7M from the source ([Dunn et al. 2021](https://doi.org/10.6084/m9.figshare.c.5295370)).
Convert it with [tracktail-preprocessing](https://github.com/AI-HHMI/tracktail-preprocessing) (tag `v1.0-paper`):
`pixi run python generate_processed_dataset.py --dataset rat7m --in RAW/rat7m --out <data> --seed 3`.
Subject s5 is the test split, so the trial is at `<data>/rat7m/test/s5-d1/101500`.

## Commands

Run from the repo root. Replace `<data>` with the converter output dir.

```bash
TRIAL=<data>/rat7m/test/s5-d1/101500
OUT=out/cover_rat7m

# 1-3. cloud -> track -> all panels  (writes $OUT/cloud.npz, $OUT/track.npz, $OUT/cover_*.{png,pdf})
pixi run python scripts/cover/cover_rat7m_demo.py --trial $TRIAL --frame 670 --n-frames 360 \
    --out-dir $OUT --extra-path deps_cover

# 4. time-lapse panel for the cover (right side)
pixi run python scripts/cover/render_timelapse.py --trial $TRIAL --track $OUT/track.npz \
    --out $OUT/rat7m_cloud_timelapse_tracks
```

Use `--stages cloud`, `--stages track` or `--stages figure` to run single stages.
Later stages read `cloud.npz` and `track.npz` from `--out-dir`.

## Outputs used by the paper

- `$OUT/rat7m_cloud_timelapse_tracks.pdf`: time-lapse panel on the right side of the cover.
- `$OUT/cloud.npz`: read by `tracktail-paper/figure_rat7m.ipynb` (`cloud_path`, code cell 4),
  which draws `rat7m_camera_views_cloud.pdf`.
- The other `figure_rat7m.ipynb` panels (`rat7m_examples.pdf` and the `rat7m_with_tracks*.pdf` views) read the trial
  videos and a separate Tracktail prediction file, committed in tracktail-paper as `data/rat7m/tracktail_output.npz`.
  They are not part of the cover and this pipeline does not make them.
- The final cover (`figures/cover_figure_draft_liliedit2.pdf` in the paper source) was assembled by hand from these panels in a vector editor.
