# Tracktail: reproducing the paper

This is the `paper` branch of [Tracktail](https://github.com/AI-HHMI/tracktail). It holds the exact training
configs, evaluation drivers and baseline setup behind the numbers in the Tracktail paper.
<!-- TODO: paper title + arXiv link -->
For general use (inference server, Python API, latest models) see the
[`main` branch](https://github.com/AI-HHMI/tracktail).

Related repositories:
- [tracktail-paper](https://github.com/AI-HHMI/tracktail-paper): builds every table and figure from result summaries.
- [tracktail-preprocessing](https://github.com/AI-HHMI/tracktail-preprocessing) (tag `v1.0-paper`): converts the
  public datasets into the trial layout used here.

Quick path: the result summaries behind every table and figure are committed in tracktail-paper (`data/`), so
the tables and figures rebuild without any of the steps below:

```bash
git clone https://github.com/AI-HHMI/tracktail-paper && cd tracktail-paper
pixi install && pixi run paper        # tables/*.tex, figures/*.pdf
```

The rest of this README regenerates those summaries from data and checkpoints.

## Contents
1. [Environment](#environment)
2. [Paths](#paths)
3. [Data](#data)
4. [Models](#models)
5. [Training](#training)
6. [Reproducing each table and figure](#reproducing-each-table-and-figure)
7. [Metric notes](#metric-notes)
8. [Citation](#citation)

## Environment

```bash
git clone -b paper https://github.com/AI-HHMI/tracktail.git
cd tracktail
pixi install                 # Python 3.11, CUDA 12.8 PyTorch; Tracktail, metrics, CoTracker3 driver
```

The baselines (MVTracker + DUSt3R, CoTracker3, ARROW) need their own code, weights and environments:

```bash
bash baselines/setup.sh      # clones the pinned commits into $BASELINE_ROOT and downloads the weights (sha256-checked)
```

See [`baselines/README.md`](baselines/README.md) for the pinned commits, weights and the Python environments.

## Paths

Every paper driver sources [`paper.env`](paper.env). Edit it (or export the variables first, or put
overrides in an untracked `paper.local.env` next to it):

| Variable | Meaning | Default |
|---|---|---|
| `TRACKTAIL_DATA` | dataset root in the trial layout (below) | `./data` |
| `MVTRACKER_DATA` | MVTracker's own benchmark datasets (for the MVTracker baseline and the cached-track files) | `./data/mvtracker` |
| `TRACKTAIL_RESULTS` | all outputs: training runs, predictions, metrics, logs | `./results` |
| `TRACKTAIL_CKPTS` | local copy of the paper checkpoints (only to start finetuning from the released pretrain) | `./checkpoints` |
| `BASELINE_ROOT` | baseline repos and weights (`baselines/setup.sh`) | `./external` |
| `TRACKTAIL_WANDB_BASE` | optional: original W&B run dirs; if set, drivers load those instead of Hugging Face | unset |

Results trees (subfolder names as in the paper's results):
`$TRACKTAIL_RESULTS/tracktail-outputs/`, `mvtracker-outputs/`, `mvtracker-outputs-dexycb-sensordepth/`,
`cotracker-outputs/`, `arrow-outputs/`.

No driver needs a job scheduler. Drivers that can also submit LSF jobs do so only with `SUBMIT=1`.

## Data

Every dataset uses the same trial layout:

```
$TRACKTAIL_DATA/<dataset>/{train,val,test}/<session>/<trial>/
    metadata.yaml        # intrinsics, world->camera extrinsics, distortion, image sizes per camera
    img/<camera>/*.jpg   # or vid/<camera>.mp4
    pose3d.npz           # pose (subjects, T, points, 3), vis (subjects, T, points, cameras)
    pose3d_*.npz         # evaluation track selections (see below)
```

**Multi-view Kubric and Multi-view Point Odyssey (training + test):**

> TODO: download links for Multi-view Kubric and Multi-view Point Odyssey (human/animal) — upload in progress.

Expected tree after download:

```
$TRACKTAIL_DATA/kubric-multiview/{train,val,test}/<session>/trial/          # 4,989 / 2 / 30 scenes, 10 views, 24 frames
$TRACKTAIL_DATA/point-odyssey-human/{train,val,test}/<session>/trial/       # 113 / 1 / 8, 9 views, test 499 frames
$TRACKTAIL_DATA/po-animal-sub256-motion/{train,val,test}/<session>/trial/   # 210 / 1 / 10, 9 views, test 499 frames
```

The Point Odyssey test trials ship `pose3d.npz` = the 512-track speed-stratified evaluation selection and
`pose3d_full.npz` = all cleaned tracks. `scripts/make_pose3d_512.py --overwrite` regenerates `pose3d.npz` from
`pose3d_full.npz`.

**Other evaluation datasets** (download from the original source, then convert with
[tracktail-preprocessing](https://github.com/AI-HHMI/tracktail-preprocessing), tag `v1.0-paper`):

| Dataset dir | Source | Used for |
|---|---|---|
| Dataset dir | Raw data | Used for |
|---|---|---|
| `dex_ycb` | MVTracker's [`dex-ycb-multiview.tar.gz`](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/dex-ycb-multiview.tar.gz) | Tab. main_results, Fig. ncams |
| `cmupanoptic_3dgs` | MVTracker's [`panoptic-multiview.tar.gz`](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/panoptic-multiview.tar.gz) | Tab. main_results |
| `tapvid3d_adt`, `tapvid3d_drivetrack`, `tapvid3d_pstudio` | per-clip npz from the official [TAPVid-3D scripts](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) | Tab. TAPVid-3D (mini-val subset: `arrow_baseline/tapvid3d_minival.json`) |
| `rat7m` | [Rat 7M](https://doi.org/10.6084/m9.figshare.c.5295370) (`data/`, `videos/`) | cover figure |

```bash
git clone https://github.com/AI-HHMI/tracktail-preprocessing && cd tracktail-preprocessing
git checkout v1.0-paper && pixi install
# --in is the raw dataset folder; output goes to $TRACKTAIL_DATA/<dataset>/; seed 3 = paper
pixi run python generate_processed_dataset.py --dataset dex_ycb          --in RAW/dex-ycb-multiview  --out $TRACKTAIL_DATA --seed 3
pixi run python generate_processed_dataset.py --dataset cmupanoptic_3dgs --in RAW/panoptic-multiview --out $TRACKTAIL_DATA --seed 3
pixi run python generate_processed_dataset.py --dataset tapvid3d_adt     --in RAW/tapvid3d/adt       --out $TRACKTAIL_DATA --seed 3   # same for drivetrack, pstudio
pixi run python generate_processed_dataset.py --dataset rat7m            --in RAW/rat7m              --out $TRACKTAIL_DATA --seed 3
```

The Kubric test set is part of the Multi-view Kubric download. How the released Multi-view Point Odyssey data was
made from the Blender renders (conversion, 256-frame animal windows, track cleaning) is documented in the
tracktail-preprocessing README; the renderer itself is not released.

Also keep the extracted MVTracker archives (`kubric-multiview`, `dex-ycb-multiview`, `panoptic-multiview`, with
their `cache/cachedtracks--seed72-*.npz` files) under `$MVTRACKER_DATA`: the cached-track files and the MVTracker
baseline's GT depths and DUSt3R depths are read from there.

**Evaluation track files** (written into each test trial; run once after the data is in place):

```bash
source paper.env
pixi run python scripts/make_pose3d_cached.py        # pose3d_cached.npz: MVTracker's cached tracks (Kubric, DexYCB, Panoptic)
pixi run python scripts/make_pose3d_cached_100.py    # pose3d_cached_100.npz: 100-track subset for the qualitative figure
pixi run python scripts/make_pose3d_ncam.py          # pose3d_ncam{1..8}_512.npz: DexYCB camera-count ablation
```

## Models

All checkpoints are packaged eval weights (`model.pth` with the model config embedded) on Hugging Face.

| Paper row | Checkpoint (repo / subfolder @ revision) | Config | W&B run @ iteration |
|---|---|---|---|
| Tracktail (all main results) | `ai-hhmi/tracktail-static` @ `2026-10-12` (= `ai-hhmi/tracktail-paper-checkpoints` / `finetune`) | `configs/paper/finetune.toml` | `vzkjqrys` @ 2314240 |
| Pretrained model (ablation reference) | `ai-hhmi/tracktail-paper-checkpoints` / `pretrain` @ `paper-v1` | `configs/paper/pretrain.toml` | `e25w9v86` @ 799996 |
| No camera self-attention | … / `ablation-cam-attn-off` | `configs/paper/ablation_cam_attn_off.toml` | `folhop3e` @ 799996 |
| No temporal self-attention | … / `ablation-temporal-attn-off` | `configs/paper/ablation_temporal_attn_off.toml` | `b3de55ap` @ 799996 |
| Neither | … / `ablation-cam-temporal-attn-off` | `configs/paper/ablation_cam_temporal_attn_off.toml` | `cnmw9dcf` @ 799996 |
| Encoder Base | … / `ablation-encoder-base` | `configs/paper/ablation_encoder_base.toml` | `hv8nguwx` @ 799996 |
| Encoder Giant | … / `ablation-encoder-giant` | `configs/paper/ablation_encoder_giant.toml` | `ubk4lvez` @ 799996 |
| Encoder Gigantic | … / `ablation-encoder-gigantic` | `configs/paper/ablation_encoder_gigantic.toml` | `u1c34ou7` @ 799996 |
| Latent 256 | … / `ablation-latent-256` | `configs/paper/ablation_latent_256.toml` | `lczwsdx3` @ 799996 |
| Latent 1024 | … / `ablation-latent-1024` | `configs/paper/ablation_latent_1024.toml` | `wk2i4o1u` @ 799996 |

`scripts/eval_testset_metrics.py`, `scripts/tracktail_tapvid3d_official.py` and `benchmark_speed/bench.py` take
`--hf-repo REPO --revision REV --subfolder DIR` (default: the paper model), `--checkpoint model.pth`, or
`--wandb-folder RUN_DIR`. The shell drivers take a network name (`NETWORKS=...`, list in
`scripts/paper_common.sh`) and map it to these arguments.

Parameter count (`pixi run python scripts/count_params.py`): 350.5M = 307.8M scene encoder (V-JEPA 2.1 ViT-L
304.7M + 2.1M projection) + 6.4M query encoder + 36.3M decoder.

## Training

Two stages; details, wall times and the loss-weight mapping in [`configs/paper/RUNS.md`](configs/paper/RUNS.md).

1. Pretrain, `configs/paper/pretrain.toml`: 4× H100 80GB, 800k iterations (≈49 h), frozen V-JEPA 2.1 Large encoder.
2. Finetune, `configs/paper/finetune.toml`: 8× H100 80GB, from the pretrain weights, encoder unfrozen, Kubric
   upweighted 3×; the paper model is checkpoint 2314240 (the run was stopped there; `n_iterations` is larger).
3. Ablations: `configs/paper/ablation_*.toml`, copies of the pretrain config with one model setting changed,
   evaluated at checkpoint 799996.

```bash
source paper.env
pixi run python train.py --config-path configs/paper/pretrain.toml --devices -1 \
    --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true
hf download ai-hhmi/tracktail-paper-checkpoints --revision paper-v1 --include 'pretrain/*' --local-dir "$TRACKTAIL_CKPTS"
pixi run python train.py --config-path configs/paper/finetune.toml --devices -1 \
    --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true
```

`train.py` multiplies the configured learning rate by `sqrt(n_gpus)`. The paper LRs are 1e-3 (pretrain, 5e-4 × √4)
and 2.5e-4 (finetune, 9e-5 × √8); with a different GPU count, scale `learning_rate` so the product stays the same.
Iteration counts in the configs are totals over all GPUs. The V-JEPA 2.1 weights download automatically.

## Reproducing each table and figure

Run from the repo root after `source paper.env`. Tracktail drivers default to the paper protocol: query
occlusion unknown (`QUERY_OCC=unknown`), visibility threshold 0.1 (`VIS_THRESH=0.1`). Predictions already in an
output folder are reused, so a driver can be rerun after an interruption. Each sequence ends in tracktail-paper,
which reads the metrics files (point it at `$TRACKTAIL_RESULTS`, or use its committed `data/`).

After the reruns, rebuild tracktail-paper's `data/` from your results tree and then the tables and figures:

```bash
cd ../tracktail-paper && source ../tracktail/paper.env
pixi run python build_paper_data.py collect ncam survival   # copies summaries, builds survival + ncam CSVs
pixi run python cotracker_triangulation_stats.py
pixi run paper
```

| Paper item | Drivers here (below) | tracktail-paper |
|---|---|---|
| Tab. main_results, po_results, ablations ×3, cotracker_triangulation, TAPVid-3D | see below | `tables.py` → `tables/<label>.tex` |
| Fig. ncams, survival ×2 | see below | `build_figures.py` |
| Fig. speed ×2 | `benchmark_speed/run_all.sh` | `figure_speed.ipynb` |
| Fig. qualitative | `*_100.sh`, `scripts/render_qualitative_gtvis.sh` | `figure_qualitative.ipynb` (`RENDERS_ROOT`) |
| Fig. po_examples, architecture crops | – (MV-PO data) | `figure_po_examples.ipynb`, `figure_po_architecture.ipynb` |
| Cover (Rat7M) | `scripts/cover/` | `figure_rat7m.ipynb` |
| Tab. loss_weights | `configs/paper/RUNS.md` | hand-written |

**Tab. main_results** (Kubric, DexYCB, Panoptic; 4 canonical views, MVTracker's cached tracks):

```bash
bash scripts/tracktail_cached.sh                                   # Tracktail -> tracktail-outputs/main_finetune_vzkjqrys_canonical_cached
bash mvtracker_baseline/precompute_dust3r_native.sh canonical      # DUSt3R depths for MVTracker (GPU-heavy)
bash mvtracker_baseline/mvtracker_cached.sh                        # MVTracker, DUSt3R and GT depth -> mvtracker-outputs/paper-canonical-cached-{duster,gt}
bash mvtracker_baseline/mvtracker_dexycb_sensordepth.sh linear     # MVTracker DexYCB with Kinect depth (the † row)
bash cotracker_baseline/cotracker_cached.sh                        # CoTracker3 + triangulation
bash arrow_baseline/arrow_cached.sh                                # ARROW
```

**Tab. po_results** (Point Odyssey human + animal, 4 and 9 views, 512 tracks per trial):

```bash
bash scripts/tracktail_cached_po.sh                                # 4 and 9 cameras
bash mvtracker_baseline/precompute_dust3r_point_odyssey.sh 4       # DUSt3R depths, 4 views (view2/4/5/7)
bash mvtracker_baseline/precompute_dust3r_point_odyssey.sh         # DUSt3R depths, 9 views
bash mvtracker_baseline/mvtracker_cached_po.sh                     # VIS_THRESHOLD=0.5 (default here) -> paper-po-*_0.5
bash cotracker_baseline/cotracker_cached_po.sh
bash arrow_baseline/arrow_cached_po.sh                             # 24-frame windows; 9 views needs an 80 GB GPU
```

The MVTracker GT-depth rows need the PO GT depth maps (`preprocess_gt_depths_point_odyssey.py`), which need the
raw Point Odyssey depth renders. <!-- TODO: say whether gt-depths ship with the data release -->

**Ablation tables** (camera/temporal attention, encoder size, latent size; pretrain-stage models):

```bash
NETWORKS=all bash scripts/tracktail_cached.sh                      # pretrain + 8 ablations (+ the paper model)
```

**Fig. ncams** (DexYCB, 1–8 views):

```bash
bash scripts/eval_ncam_ablation_local.sh                           # Tracktail -> tracktail-outputs/ncam_ablation_dexycb
bash mvtracker_baseline/mvtracker_ncam_ablation_native.sh
bash cotracker_baseline/cotracker_ncam_ablation.sh
bash arrow_baseline/arrow_ncam_ablation.sh
```

**Fig. survival** (Point Odyssey, 4 views): uses the 4-camera Point Odyssey outputs above
(`build_paper_data.py survival` in tracktail-paper; `scripts/compute_survival.py` here does the same computation).

**Tab. cotracker_triangulation**: from the CoTracker3 outputs above (`cotracker_triangulation_stats.py` in tracktail-paper).

**Tab. TAPVid-3D** (official protocol, mini-val):

```bash
bash scripts/tracktail_tapvid3d_official.sh                        # -> tracktail-outputs/main_finetune_vzkjqrys_tapvid3d_official_minival_vis0.1_occunk
```

The ARROW and other baseline rows are copied from the ARROW paper.

**Fig. qualitative**: 100-track runs, then renders.

```bash
bash scripts/tracktail_100.sh
bash mvtracker_baseline/mvtracker_100.sh
bash cotracker_baseline/cotracker_100.sh
bash arrow_baseline/arrow_100.sh
bash scripts/render_qualitative_gtvis.sh
```

The paper's Tracktail renders came from an earlier 100-track run with ground-truth visibility at the query
frame; `QUERY_OCC=gt bash scripts/tracktail_100.sh` matches it.

**Fig. speed / speed_tracktail**: `bash benchmark_speed/run_all.sh` (all methods on one GPU; the paper used an
A100-SXM4-80GB). The paper's raw results are in `benchmark_speed/results/a100_mvtracker_20261026_133500/`.
At 10 views the 10th input repeats view 2 (Point Odyssey has 9 cameras).

**Cover figure (Rat7M)**: see [`scripts/cover/README.md`](scripts/cover/README.md) (extra dependencies: SAM, VGGT).

## Metric notes

Implemented in `tracktail/model/eval_metrics_paper.py`; these details are not all spelled out in the paper.

- Each track is scored from its query frame on; the query frame itself and earlier frames are not counted for
  AJ / δ_avg / OA. MTE is the lower median of the per-frame 3D error and includes the query frame.
- A track needs at least 2 visible frames to be scored.
- Kubric and DexYCB: tracks are split by GT path length into static (< 0.01) and dynamic (> 0.1); the reported
  number is the mean of the two groups, and tracks in between are excluded.
- Aggregation: mean over tracks within a trial, then mean over trials. Trials where a method fails are dropped
  for that method.
- Thresholds per dataset are in `SETTINGS` in `scripts/eval_testset_metrics.py`.
- CoTracker3 uses its built-in visibility threshold (0.9); MVTracker uses its official per-dataset thresholds
  (Kubric 0.5, DexYCB/Panoptic 0.01) except on Point Odyssey (0.5).
- Expect small run-to-run differences: rerunning the DexYCB MVTracker-DUSt3R row gave AJ 73.8 vs 73.4 in the paper.

## Citation

<!-- TODO: BibTeX once public -->
