#!/usr/bin/env python
"""Count Tracktail parameters, in total and per top-level submodule.

The paper reports 351M parameters: 308M (V-JEPA 2 video encoder) + 6M + 36M.

    pixi run python scripts/count_params.py                      # anonymous/tracktail-static @ 2026-10-12
    pixi run python scripts/count_params.py --revision main
    pixi run python scripts/count_params.py --checkpoint path/to/model.pth   # packaged checkpoint
    pixi run python scripts/count_params.py --checkpoint path/to/run_dir     # training run folder

Runs on CPU. Use --depth 2 for a finer breakdown.
"""
import argparse
import os

import torch


def load(args):
    if args.checkpoint is None:
        from tracktail.inference.inference_utils import load_model_from_hub
        model, _ = load_model_from_hub(args.repo, device='cpu', revision=args.revision)
        return model, f'{args.repo}@{args.revision}'
    if os.path.isdir(args.checkpoint):
        from tracktail.inference.inference_utils import load_model_from_base_folder
        model, _, _, path = load_model_from_base_folder(args.checkpoint, device=torch.device('cpu'))
        return model, str(path)
    # packaged checkpoint: {'model_state': ..., 'model_config': ...}
    from tracktail.model.tracker_encoder import TrackerEncoder
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = TrackerEncoder(**{**ckpt['model_config'], 'video_encoder_pretrained': False})
    model.load_state_dict(ckpt['model_state'])
    return model, args.checkpoint


def count(module):
    return sum(p.numel() for p in module.parameters())


def report(module, prefix, depth, total, out):
    for name, child in module.named_children():
        n = count(child)
        if n == 0:
            continue
        full = f'{prefix}{name}'
        out.append((full, n))
        print(f'  {full:<48s} {n:>13,d}  {n / 1e6:8.2f}M  {100 * n / total:5.1f}%')
        if depth > 1:
            report(child, full + '.', depth - 1, total, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', default='anonymous/tracktail-static')
    p.add_argument('--revision', default='2026-10-12')
    p.add_argument('--checkpoint', default=None, help='packaged model.pth or a training run folder (overrides --repo)')
    p.add_argument('--depth', type=int, default=1, help='submodule depth of the breakdown')
    args = p.parse_args()

    model, source = load(args)
    total = count(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model: {source}')
    print(f'total parameters: {total:,d} ({total / 1e6:.1f}M)')
    print(f'requires_grad:    {trainable:,d} ({trainable / 1e6:.1f}M)')
    print('by submodule:')
    out = []
    report(model, '', args.depth, total, out)
    direct = sum(p.numel() for p in model.parameters(recurse=False))
    if direct:
        print(f'  {"(parameters held directly by the model)":<48s} {direct:>13,d}  {direct / 1e6:8.2f}M')


if __name__ == '__main__':
    main()
