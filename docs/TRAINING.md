# Training Tracktail

This page covers training on `main`. The exact configs and recipe behind the paper's model
(`ai-hhmi/tracktail-static`) are on the [`paper` branch](https://github.com/AI-HHMI/tracktail/tree/paper)
in `configs/paper/`.

## Method overview
- **Video encoder:** V-JEPA 2.1 ViT-L/16 with hierarchical features, run on each view.
- **Query encoder:** builds a per-camera embedding from the 2D projection, a 9×9 patch, the
  times, the normalized depth, and an in-bounds flag.
- **Decoder:** N layers, each with camera self-attention (using PRoPE ray encodings), temporal
  self-attention, query→scene cross-attention, and an FFN.
- **Heads:** 3D/2D position, depth, visibility, and 2D/3D confidence. Per-camera predictions are
  fused by confidence.

## Data
- Data lives under `<prefix>/<dataset>/<split>/<session>/<trial>/`. Each trial has the same layout
  as for inference (`metadata.yaml`, `pose3d.npz`/`pose2d.npz`, `img/` or `vid/`; see the
  [README](../README.md#python-api)).
- Datasets: Multi-View Kubric, Multiview Point Odyssey (Human/Animal), Panoptic Studio, DexYCB,
  plus 2D-only sets. How to get the paper's datasets is described on the
  [`paper` branch](https://github.com/AI-HHMI/tracktail/tree/paper).
- Sampling (cameras, keypoints, crops, augmentation, dataset balancing) is set in
  `[dataset.train]` in the config.

## Configs
- Animal finetune of `tracktail-static-animal`: `configs/config_tracktail_finetuning_v6.toml`.
- Scorer: `configs/config_scorer.toml`.
- `configs/archive/`: older development configs.

Key sections: `[model]`, `[dataset.*]`, `[training]` (optimizer: AdamW, schedule-free, or
Muon; losses; `checkpoint_path`/`finetune`), and `[wandb]`. Iteration counts are **totals
across all GPUs**. `train.py` multiplies the configured learning rate by the square root of the
number of GPUs.

## Launching
Config values like `${TRACKTAIL_DATA}` are read from the environment when the config is loaded.

```bash
export TRACKTAIL_DATA=/path/to/datasets TRACKTAIL_RESULTS=/path/to/results INIT_CKPT=/path/to/checkpoint.pth
pixi run python train.py --config-path configs/config_tracktail_finetuning_v6.toml \
    --devices -1 --strategy ddp_find_unused_parameters_true      # Lightning Fabric DDP
# LSF cluster: cluster/big_bsub_{pretraining,finetuning}.sh -> cluster/train_script.sh
```
- **Outputs:** `<wandb.path>/wandb/run-*/files/{config.toml, checkpoints/checkpoint_XXXXXXXX.pth}`.
- **Resuming:** set `training.checkpoint_path` and `finetune = false`. To warm-start from
  another stage, use `finetune = true`.
- **Scorer:** `train_scorer.py`.

## Releasing a checkpoint
`scripts/package_checkpoint.py` strips a run down to the eval weights plus the model config,
producing `model.pth`. Upload it with `hf upload ai-hhmi/<repo> model.pth` and tag a dated
revision (`hf repo tag create ai-hhmi/<repo> <date>`). Load it with
`TrackerEncoder.from_pretrained("ai-hhmi/<repo>", revision="<date>")`.
