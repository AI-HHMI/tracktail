# Paper training runs

Configs in this folder are exact copies of each run's W&B `files/config.toml`, with three
machine-specific values replaced by environment variables (expanded at load time by
`tracktail.model.train_utils.load_config`; an unset variable is an error):

| Key | Value | Meaning |
|---|---|---|
| `[dataset].prefix` | `${TRACKTAIL_DATA}` | dataset root with `kubric-multiview/`, `point-odyssey-human/`, `po-animal-sub256-motion/` (only these have `train/`) |
| `[wandb].path` | `${TRACKTAIL_RESULTS}/training` | W&B run dirs; checkpoints land in `<run>/files/checkpoints/` |
| `[training].checkpoint_path` (finetune only) | `${PRETRAIN_CKPT}` | your pretrain checkpoint to start from |

Set them with `source paper.env` (repo root). The `run_id` / `run_dir` keys that train.py
writes into the saved config were removed.

All runs used code commit `462eda2` (an ancestor of this branch) and were launched as
`pixi run python train.py --config-path <cfg> --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true`
with all GPUs of one node (`--devices -1`).

| Config | W&B run | Checkpoint (iteration) | GPUs | Wall time | Paper |
|---|---|---|---|---|---|
| `pretrain.toml` | `e25w9v86` | 799996 | 4× H100 80GB | 49 h | Pretrained model (Tab. ablations, reference row ✓✓ / Large / 512) |
| `finetune.toml` | `vzkjqrys` | 2314240 | 8× H100 80GB | ~192 h (run start → checkpoint file) | **Tracktail (paper model)**, released as `ai-hhmi/tracktail-static` @ `2026-10-12`: all main results, PO, TAPVid-3D, ncams, survival, speed |
| `ablation_cam_attn_off.toml` | `folhop3e` | 799996 | 4× H100 80GB | 46 h | Tab. ablation_camattn: camera self-attn ✗, temporal ✓ |
| `ablation_temporal_attn_off.toml` | `b3de55ap` | 799996 | 4× H100 80GB | 46 h | Tab. ablation_camattn: camera ✓, temporal ✗ |
| `ablation_cam_temporal_attn_off.toml` | `cnmw9dcf` | 799996 | 4× H100 80GB | 44 h | Tab. ablation_camattn: camera ✗, temporal ✗ |
| `ablation_encoder_base.toml` | `hv8nguwx` | 799996 | 4× H100 80GB | 41 h | Tab. ablation_encoder: Base (86M) |
| `ablation_encoder_giant.toml` | `ubk4lvez` | 799996 | 4× H100 80GB | 68 h | Tab. ablation_encoder: Giant (1B) |
| `ablation_encoder_gigantic.toml` | `u1c34ou7` | 799996 | 4× H100 80GB | 88 h | Tab. ablation_encoder: Gigantic (2B) |
| `ablation_latent_256.toml` | `lczwsdx3` | 799996 | 4× H100 80GB | 54 h | Tab. ablation_latent: 256 |
| `ablation_latent_1024.toml` | `wk2i4o1u` | 799996 | 4× H100 80GB | 52 h | Tab. ablation_latent: 1024 |

Wall time is W&B `_runtime` (pretrain/ablations stopped at `n_iterations` = 800k). The finetune
run has no W&B summary; its time is the gap between run start and the checkpoint file and may
include queue restarts.

Only the paper model is released. The pretrain and ablation models are not: train them with these
configs, or use the committed result summaries in tracktail-paper (`data/`) for the ablation tables.

## Recipe

1. Pretrain (`pretrain.toml`, 4 GPUs): frozen V-JEPA 2.1 Large encoder, decoder from scratch,
   800k iterations (counted over all GPUs), Huber δ = 4, seed 3.
2. Finetune (`finetune.toml`, 8 GPUs): start from your pretrain checkpoint (`PRETRAIN_CKPT`) (`finetune = true`
   loads weights only, fresh optimizer), whole encoder unfrozen with `encoder_lr_scale = 0.25`,
   Kubric upweighted 3× (`[dataset.train.dataset_weights]`), Huber δ = 6, seed 0.
   `n_iterations` is 8M but the run was stopped by hand; the paper model is checkpoint
   2314240 (`ai-hhmi/tracktail-static` revision `2026-10-12`, packaged eval weights).
3. Ablations: copies of `pretrain.toml` with one model key changed (and val/test
   `n_frames = 16`, validation only). Evaluated at checkpoint 799996, like the pretrain row.

```bash
source paper.env
# pretrain (4 GPUs); ablations: same with configs/paper/ablation_*.toml
pixi run python train.py --config-path configs/paper/pretrain.toml --devices -1 \
    --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true
# finetune from your pretrain run (8 GPUs)
export PRETRAIN_CKPT=$TRACKTAIL_RESULTS/training/wandb/run-<date>-<id>/files/checkpoints/checkpoint_00799996.pth
pixi run python train.py --config-path configs/paper/finetune.toml --devices -1 \
    --num-nodes 1 --precision 32 --strategy ddp_find_unused_parameters_true
```

To evaluate your models with the paper drivers, package each run
(`pixi run python scripts/package_checkpoint.py --run-dir <run> --output $TRACKTAIL_MODELS/<network>/model.pth`,
network names in `scripts/paper_common.sh`) or symlink `$TRACKTAIL_MODELS/<network>` to the run dir.

## Learning rate and GPU count

`train.py` multiplies `learning_rate` by `sqrt(world_size)`:

| Run | `learning_rate` | GPUs | Effective LR (paper) |
|---|---|---|---|
| pretrain + ablations | 5e-4 | 4 | 1e-3 |
| finetune | 9e-5 | 8 | 2.5e-4 |

With a different GPU count, set `learning_rate = paper_LR / sqrt(n_gpus)` to keep the paper LR.
`n_iterations`, `warmup_steps`, `checkpoint_freq` etc. are totals over all GPUs, so they need no change.

## Loss weights (Tab. loss_weights)

- λ_3D = `coords_loss_weight` + `coords_loss_direct_weight` = 0.1 + 0.1 = 0.2. The fused
  prediction `coords_pred` is `3d_pred_direct`, so both terms supervise it.
- λ_cam = `coords_loss_direct_weight` = 0.1 on the per-camera predictions `3d_pred_cams_direct`.
- λ_rep = `coords_loss_*_reproj_weight` / 16 (`losses.py` divides reprojection weights by 16).

## Latent-dim ablation

Only `[model].latent_dim` changes (256 / 1024). `cross_attn_dim` and `scene_proj_dim` stay 512.
