"""Shared command-line model selection for evaluation scripts.

Sources, in order of precedence:
  --wandb-folder RUN_DIR [--checkpoint STEP]   a training run (files/config.toml + files/checkpoints/)
  --checkpoint FILE.pth [--config CONFIG.toml] a local checkpoint: a packaged model.pth
                                               (model_state + model_config), or a raw training
                                               checkpoint together with its training config
  --hf-repo REPO [--revision REV] [--subfolder DIR]   a packaged checkpoint on the Hugging Face Hub
The default is the paper model, anonymous/tracktail-static at revision 2026-10-12.
"""

import os

PAPER_HF_REPO = 'anonymous/tracktail-static'
PAPER_HF_REVISION = '2026-10-12'


def add_model_args(parser, default_hf_repo=PAPER_HF_REPO, default_revision=PAPER_HF_REVISION):
    g = parser.add_argument_group('model (default: the paper model from the Hugging Face Hub)')
    g.add_argument('--hf-repo', default=None,
                   help=f'Hugging Face model repo holding a packaged model.pth (default {default_hf_repo})')
    g.add_argument('--revision', default=None,
                   help=f'Hub revision: tag, branch or commit (default {default_revision} for {default_hf_repo})')
    g.add_argument('--subfolder', default=None,
                   help='folder inside the Hub repo that holds model.pth (default: the repo root)')
    g.add_argument('--wandb-folder', default=None,
                   help='training run dir (files/config.toml + files/checkpoints/) instead of the Hub')
    g.add_argument('--checkpoint', default=None,
                   help='with --wandb-folder: checkpoint step (default latest); otherwise a local .pth '
                        '(packaged model.pth, or a training checkpoint together with --config)')
    g.add_argument('--config', default=None,
                   help='training config.toml for a raw training checkpoint given by --checkpoint')
    parser.set_defaults(_default_hf_repo=default_hf_repo, _default_revision=default_revision)
    return g


def describe_model_source(args):
    if args.wandb_folder:
        return f'wandb run {args.wandb_folder} (checkpoint={args.checkpoint or "latest"})'
    if args.checkpoint:
        return f'checkpoint {args.checkpoint}' + (f' (config {args.config})' if args.config else '')
    repo, rev = _hub_repo_revision(args)
    return f'hf://{repo}' + (f'/{args.subfolder}' if args.subfolder else '') + (f'@{rev}' if rev else '')


def _hub_repo_revision(args):
    repo = args.hf_repo or getattr(args, '_default_hf_repo', PAPER_HF_REPO)
    rev = args.revision
    if rev is None and args.hf_repo is None:
        rev = getattr(args, '_default_revision', PAPER_HF_REVISION)
    return repo, rev


def load_model_from_args(args, device):
    """Return ``(model, config_label, checkpoint_label)``.

    The labels are what gets recorded in the prediction files (``config_path`` /
    ``checkpoint_path``); for Hub models they are ``hf://repo[/subfolder]@revision``.
    """
    from tracktail.inference.inference_utils import load_model_from_base_folder, load_model_from_hub
    from tracktail.model.tracker_encoder import TrackerEncoder

    if args.wandb_folder:
        step = None
        if args.checkpoint is not None:
            try:
                step = int(args.checkpoint)
            except ValueError:
                raise SystemExit('--checkpoint must be an integer step when --wandb-folder is given')
        model, _cfg, cfg_path, ckpt = load_model_from_base_folder(args.wandb_folder, checkpoint=step, device=device)
        return model, cfg_path, ckpt

    if args.checkpoint:
        path = os.path.expanduser(args.checkpoint)
        if args.config:
            from tracktail.model.train_utils import load_checkpoint
            model = load_checkpoint(args.config, path, device=device)['model']
            model.eval()
            return model, args.config, path
        model = TrackerEncoder.from_packaged_checkpoint(path, device=device)
        return model, path, path

    repo, rev = _hub_repo_revision(args)
    model, _cfg = load_model_from_hub(repo, device=device, revision=rev, subfolder=args.subfolder)
    label = describe_model_source(args)
    return model, label, label
