# Tracktail: reproducing the paper

> **Anonymized for double-blind review.** Repository links, the model-weights location and
> author/institution details have been removed. Links and the Hugging Face model id below
> (`anonymous/tracktail-static`) are placeholders; the code, the companion repositories and the
> released model weights will be made public upon acceptance.

This code accompanies the submission *Tracktail: Multi-View 3D Point Tracking without Depth Input*.
It contains everything needed to reproduce the results in the paper: the exact training configs,
the evaluation scripts, and the setup for the baseline methods we compare against.

The paper is reproduced with three repositories:

| Repository | What it does |
|---|---|
| **tracktail** (this branch) | runs Tracktail and the baselines, and computes the metrics |
| tracktail-paper *(anonymized, released upon acceptance)* | turns the metrics into the paper's tables and figures |
| tracktail-preprocessing *(anonymized, released upon acceptance)* (tag `v1.0-paper`) | converts public datasets into the format used here |

You can go as deep as you like:

1. [Rebuild the tables and figures](#1-rebuild-the-tables-and-figures-5-minutes) from our saved results (5 minutes, no GPU).
2. [Re-evaluate Tracktail](#2-re-evaluate-tracktail) with the released model (one GPU).
3. [Re-run the baselines](#3-re-run-the-baselines) (MVTracker, CoTracker3, ARROW).
4. [Retrain Tracktail](#4-retrain-tracktail) from scratch (about 1,700 H100 GPU-hours).

---

## 1. Rebuild the tables and figures (5 minutes)

The metrics behind every table and figure are saved in the tracktail-paper repository, so you can
regenerate them without any data or GPU:

```bash
git clone <tracktail-paper repository URL>   # anonymized for review
cd tracktail-paper
pixi install
pixi run paper          # writes tables/*.tex and figures/*.pdf
```

The rest of this README explains how to produce those saved metrics yourself.

---

## Setup (for steps 2–4)

**Install.** We use [pixi](https://pixi.sh) for the environment (Python 3.11, PyTorch with CUDA 12.8):

```bash
cd tracktail   # this repository
pixi install
```

**Choose where things go.** All scripts read their paths from [`paper.env`](paper.env). The defaults
put everything inside the repository. To use other locations, create a `paper.local.env` file next to
it (it is not tracked by git), for example:

```bash
export TRACKTAIL_DATA=/big/disk/tracktail-data        # datasets
export TRACKTAIL_RESULTS=/big/disk/tracktail-results  # predictions, metrics, training runs
```

Then, in every new shell, run `source paper.env` before the commands below. The full list of
variables is in [Reference: paths](#paths).

**Get the data.** All datasets use the same folder layout, one folder per recording ("trial"):

```
$TRACKTAIL_DATA/<dataset>/{train,val,test}/<session>/<trial>/
    metadata.yaml     # camera calibration
    img/<camera>/     # video frames
    pose3d.npz        # ground-truth 3D tracks and per-camera visibility
```

- **Multi-view Kubric and Multi-view Point Odyssey** (our training and test data):

  > TODO: download links for Multi-view Kubric and Multi-view Point Odyssey (human/animal). Upload in progress.

  After downloading you should have:

  ```
  $TRACKTAIL_DATA/kubric-multiview/          # 4,989 train / 2 val / 30 test scenes, 10 cameras, 24 frames
  $TRACKTAIL_DATA/point-odyssey-human/       # 113 / 1 / 8 sequences, 9 cameras, test sequences are 499 frames
  $TRACKTAIL_DATA/po-animal-sub256-motion/   # 210 / 1 / 10 sequences, 9 cameras, test sequences are 499 frames
  ```

  In the Point Odyssey test sets, `pose3d.npz` already holds the 512 evaluation tracks per sequence;
  all tracks are in `pose3d_full.npz`.

- **DexYCB, Panoptic Studio, TAPVid-3D and Rat7M** (test only) are downloaded from their original
  sources and converted with tracktail-preprocessing:

  | Dataset folder | Download | Used for |
  |---|---|---|
  | `dex_ycb` | MVTracker's [`dex-ycb-multiview.tar.gz`](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/dex-ycb-multiview.tar.gz) | main table, camera-count figure |
  | `cmupanoptic_3dgs` | MVTracker's [`panoptic-multiview.tar.gz`](https://huggingface.co/datasets/ethz-vlg/mv3dpt-datasets/resolve/main/panoptic-multiview.tar.gz) | main table |
  | `tapvid3d_adt`, `tapvid3d_drivetrack`, `tapvid3d_pstudio` | the official [TAPVid-3D scripts](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d) | TAPVid-3D table |
  | `rat7m` | [Rat 7M](https://doi.org/10.6084/m9.figshare.c.5295370) | cover figure |

  ```bash
  git clone <tracktail-preprocessing repository URL>   # anonymized for review
  cd tracktail-preprocessing && git checkout v1.0-paper && pixi install
  # --in is the downloaded folder; --seed 3 is what the paper used
  pixi run python generate_processed_dataset.py --dataset dex_ycb          --in RAW/dex-ycb-multiview  --out $TRACKTAIL_DATA --seed 3
  pixi run python generate_processed_dataset.py --dataset cmupanoptic_3dgs --in RAW/panoptic-multiview --out $TRACKTAIL_DATA --seed 3
  pixi run python generate_processed_dataset.py --dataset tapvid3d_adt     --in RAW/tapvid3d/adt       --out $TRACKTAIL_DATA --seed 3   # same for drivetrack, pstudio
  pixi run python generate_processed_dataset.py --dataset rat7m            --in RAW/rat7m              --out $TRACKTAIL_DATA --seed 3
  ```

  Also keep the extracted MVTracker archives (`kubric-multiview`, `dex-ycb-multiview`,
  `panoptic-multiview`, including their `cache/` folders) in `$MVTRACKER_DATA`. The evaluation tracks
  and the MVTracker baseline are read from there.

**Create the evaluation track files.** We evaluate on the same tracks and query frames as MVTracker.
This step copies them into each test trial (run once):

```bash
pixi run python scripts/make_pose3d_cached.py        # MVTracker's tracks: 512 per Kubric trial, 384 per DexYCB/Panoptic trial
pixi run python scripts/make_pose3d_cached_100.py    # 100 tracks for the qualitative figure
pixi run python scripts/make_pose3d_ncam.py          # DexYCB tracks for 1 to 8 cameras
```

---

## 2. Re-evaluate Tracktail

The model from the paper will be released on Hugging Face upon acceptance (anonymized here as
`anonymous/tracktail-static`, revision `2026-10-12`); the scripts then download it automatically.
To use a local copy of the weights instead, set `TRACKTAIL_HF_REPO` / `TRACKTAIL_HF_REVISION`, pass
`--checkpoint model.pth`, or place it at `$TRACKTAIL_MODELS/main_finetune_vzkjqrys/model.pth`.

By default the scripts use the paper's settings: 4 cameras on the MVTracker benchmarks, the query
point's visibility is not given to the model, and points count as visible above a probability of 0.1.
Finished trials are skipped, so you can rerun a script after an interruption.

| Paper result | Command | Output (under `$TRACKTAIL_RESULTS/tracktail-outputs/`) |
|---|---|---|
| Main table (Kubric, DexYCB, Panoptic Studio) | `bash scripts/tracktail_cached.sh` | `main_finetune_vzkjqrys_canonical_cached/` |
| Point Odyssey table (4 and 9 cameras) | `bash scripts/tracktail_cached_po.sh` | `main_finetune_vzkjqrys_po_{4,9}cams/` |
| Camera-count figure (DexYCB, 1–8 cameras) | `bash scripts/eval_ncam_ablation_local.sh` | `ncam_ablation_dexycb/` |
| TAPVid-3D table | `bash scripts/tracktail_tapvid3d_official.sh` | `main_finetune_vzkjqrys_tapvid3d_official_minival_vis0.1_occunk/` |
| Survival figures | uses the Point Odyssey 4-camera output above | |
| Speed figures | `bash benchmark_speed/run_all.sh` (all methods; the paper used one A100 80 GB) | `benchmark_speed/results/` |

Each output folder has the metrics in `summary.json` (`ncam_ablation_summary.csv` for the camera-count run). To turn your results into the paper's
tables and figures, point tracktail-paper at them:

```bash
cd ../tracktail-paper
source ../tracktail/paper.env
pixi run python build_paper_data.py collect ncam survival   # copy your metrics into data/
pixi run paper                                              # rebuild tables and figures
```

Anything you didn't rerun (for example the ablations, see step 4) is taken from the saved results in
tracktail-paper, with a warning.

Our raw speed results are in `benchmark_speed/results/a100_mvtracker_20261026_133500/`. In the
10-camera setting, the 10th camera repeats camera 2, because Point Odyssey has 9 cameras.

---

## 3. Re-run the baselines

The baselines need their own code, weights and Python environments. This script clones the exact
versions we used and downloads their weights:

```bash
bash baselines/setup.sh
```

See [`baselines/README.md`](baselines/README.md) for the pinned versions and how to set up each environment.

**Main table** (Kubric, DexYCB, Panoptic Studio):

```bash
bash mvtracker_baseline/precompute_dust3r_native.sh canonical      # DUSt3R depth maps for MVTracker (slow)
bash mvtracker_baseline/mvtracker_cached.sh                        # MVTracker with DUSt3R depth and with ground-truth depth
bash mvtracker_baseline/mvtracker_dexycb_sensordepth.sh linear     # MVTracker on DexYCB with Kinect depth (the † row)
bash cotracker_baseline/cotracker_cached.sh                        # CoTracker3 per camera + triangulation
bash arrow_baseline/arrow_cached.sh                                # ARROW
```

**Point Odyssey table:**

```bash
bash mvtracker_baseline/precompute_dust3r_point_odyssey.sh 4       # DUSt3R depth maps, 4 cameras
bash mvtracker_baseline/precompute_dust3r_point_odyssey.sh         # DUSt3R depth maps, 9 cameras
bash mvtracker_baseline/mvtracker_cached_po.sh
bash cotracker_baseline/cotracker_cached_po.sh
bash arrow_baseline/arrow_cached_po.sh                             # 9 cameras needs an 80 GB GPU
```

The MVTracker rows with ground-truth depth need Point Odyssey depth maps, which are not part of the
data release yet. Those rows can only be checked against the saved results in tracktail-paper.

**Camera-count figure:**

```bash
bash mvtracker_baseline/mvtracker_ncam_ablation_native.sh
bash cotracker_baseline/cotracker_ncam_ablation.sh
bash arrow_baseline/arrow_ncam_ablation.sh
```

The **CoTracker3 failure table** in the supplement is computed from the CoTracker3 outputs above by
`cotracker_triangulation_stats.py` in tracktail-paper. The TAPVid-3D numbers for D4RT and ARROW are
copied from their papers.

**Qualitative figure.** This runs every method on 100 tracks and renders the trajectories:

```bash
QUERY_OCC=gt bash scripts/tracktail_100.sh     # the paper's renders gave Tracktail the query visibility
bash mvtracker_baseline/mvtracker_100.sh
bash cotracker_baseline/cotracker_100.sh
bash arrow_baseline/arrow_100.sh
bash scripts/render_qualitative_gtvis.sh
```

Then run `figure_qualitative.ipynb` in tracktail-paper.

**Cover figure (Rat7M):** see [`scripts/cover/README.md`](scripts/cover/README.md). It needs SAM and VGGT.

---

## 4. Retrain Tracktail

Training has two stages. The configs in [`configs/paper/`](configs/paper/) are the exact configs of
the paper's runs, and [`configs/paper/RUNS.md`](configs/paper/RUNS.md) has the details of each run.

1. **Pretrain** with the video encoder frozen: 800k iterations on 4× H100 (about 49 hours).
2. **Finetune** everything, starting from your pretrained checkpoint: 8× H100. The paper's model is
   checkpoint 2,314,240 of this run (about 8 days). The config's iteration limit is higher because we
   stopped the run by hand.

```bash
source paper.env
pixi run python train.py --config-path configs/paper/pretrain.toml --devices -1 \
    --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true

export PRETRAIN_CKPT=$TRACKTAIL_RESULTS/training/wandb/run-<date>-<id>/files/checkpoints/checkpoint_00799996.pth
pixi run python train.py --config-path configs/paper/finetune.toml --devices -1 \
    --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true
```

The V-JEPA 2.1 encoder weights download automatically.

**Learning rate and GPU count.** `train.py` multiplies the learning rate in the config by the square
root of the number of GPUs. With the paper's GPU counts this gives 1e-3 for pretraining and 2.5e-4
for finetuning. If you use a different number of GPUs, adjust `learning_rate` to keep the same
product. Iteration counts in the configs are totals over all GPUs.

**Ablations.** Only the final model is released. To reproduce the ablation tables (camera and
temporal attention, encoder size, latent size), train each `configs/paper/ablation_*.toml`. Each is the
pretraining config with one setting changed. Then package the checkpoint and evaluate it:

```bash
pixi run python scripts/package_checkpoint.py --run-dir <run> --checkpoint 799996 \
    --output $TRACKTAIL_MODELS/<network>/model.pth
NETWORKS=all bash scripts/tracktail_cached.sh
```

The network names are listed in `scripts/paper_common.sh`. If you skip this, the ablation rows come
from the saved results in tracktail-paper.

---

## Reference

### Paths

All set in [`paper.env`](paper.env); override them in `paper.local.env`.

| Variable | What it is | Default |
|---|---|---|
| `TRACKTAIL_DATA` | datasets, in the layout above | `./data` |
| `MVTRACKER_DATA` | MVTracker's original benchmark downloads | `./data/mvtracker` |
| `TRACKTAIL_RESULTS` | all outputs: predictions, metrics, training runs | `./results` |
| `TRACKTAIL_MODELS` | models you trained yourself, as `<network>/model.pth` | `$TRACKTAIL_RESULTS/models` |
| `PRETRAIN_CKPT` | your pretraining checkpoint, used by the finetune config | not set |
| `BASELINE_ROOT` | baseline code and weights from `baselines/setup.sh` | `./external` |
| `TRACKTAIL_WANDB_BASE` | optional: folder of W&B training runs to load models from | not set |

Results are written to `$TRACKTAIL_RESULTS/{tracktail,mvtracker,cotracker,arrow}-outputs/`, with the
same subfolder names as in the paper's results. No script needs a job scheduler.

### Loading models

The Python evaluation scripts (`scripts/eval_testset_metrics.py`, `scripts/tracktail_tapvid3d_official.py`,
`benchmark_speed/bench.py`) accept `--hf-repo REPO --revision REV` (default: the paper model),
`--checkpoint model.pth`, or `--wandb-folder RUN_DIR`.

The shell scripts take a network name and look for its model in this order:
1. the W&B run under `$TRACKTAIL_WANDB_BASE`, if set
2. `$TRACKTAIL_MODELS/<network>/model.pth`, or `$TRACKTAIL_MODELS/<network>` as a link to a W&B run folder
3. Hugging Face (the paper model only)

| Paper row | Config | Training run (W&B id @ checkpoint) |
|---|---|---|
| Tracktail, all main results (released) | `finetune.toml` | `vzkjqrys` @ 2314240 |
| Pretrained model, the ablation reference | `pretrain.toml` | `e25w9v86` @ 799996 |
| No camera self-attention | `ablation_cam_attn_off.toml` | `folhop3e` @ 799996 |
| No temporal self-attention | `ablation_temporal_attn_off.toml` | `b3de55ap` @ 799996 |
| Neither | `ablation_cam_temporal_attn_off.toml` | `cnmw9dcf` @ 799996 |
| Encoder Base / Giant / Gigantic | `ablation_encoder_{base,giant,gigantic}.toml` | `hv8nguwx` / `ubk4lvez` / `u1c34ou7` @ 799996 |
| Latent size 256 / 1024 | `ablation_latent_{256,1024}.toml` | `lczwsdx3` / `wk2i4o1u` @ 799996 |

`pixi run python scripts/count_params.py` prints the parameter count: 350.5M in total, of which
307.8M is the video encoder and projection, 6.4M the query encoder, and 36.3M the decoder.

### How the metrics are computed

The code is in `tracktail/model/eval_metrics_paper.py`. Some details are not spelled out in the paper:

- Each track is scored from its query frame onward. AJ, δ_avg and OA skip the query frame itself; MTE
  includes it and uses the lower median.
- A track needs at least 2 visible frames to be scored.
- On Kubric and DexYCB, tracks are split by how far they move into static (path length below 0.01)
  and dynamic (above 0.1). The reported number is the average of the two groups; tracks in between
  are not used.
- Scores are averaged over the tracks in each sequence, then over sequences. If a method fails on a
  sequence, that sequence is left out for that method.
- The distance thresholds per dataset are in `SETTINGS` in `scripts/eval_testset_metrics.py`.
- Visibility thresholds: Tracktail 0.1; MVTracker uses its official values (Kubric 0.5, DexYCB and
  Panoptic 0.01) and 0.5 on Point Odyssey; ARROW 0.5; CoTracker3 uses its built-in 0.9.
- Expect small differences between runs. For example, rerunning MVTracker with DUSt3R depth on DexYCB
  gave AJ 73.8, compared with 73.4 in the paper.

## Citation

Anonymized for review.
