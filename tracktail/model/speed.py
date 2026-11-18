"""Training speedups: scoped bf16 autocast, per-block torch.compile, fused grad norm.

Driven by the optional ``[training.speed]`` config section. **These are on by default** (agreed
H100 config; finetune v6 on 1xH100: 0.890 -> 0.470 s/step = 1.89x, val MTE within noise), i.e.
with the section absent training runs as if it said:

    [training.speed]
    enabled = true
    autocast_bf16 = ['encoder', 'decoder']   # bf16 inside these parts, fp32 handed back
    compile = ['encoder']                    # torch.compile per V-JEPA encoder block
    fused_grad_norm = true                   # one device sync for the grad-norm log
    # encoder_activation_checkpointing unset = keep the model's setting (on)

Each key can be overridden on its own. The previous behaviour (fp32, no compile, per-parameter
grad norm) is one line away:

    [training.speed]
    enabled = false

Graceful fallbacks: bf16 autocast is used only on CUDA devices with bf16 support, and
torch.compile only on CUDA when ``torch.compile`` is available; otherwise those parts are skipped
(with a message) and training runs in fp32 eager.

Other options:

    # compile_mode = 'max-autotune-no-cudagraphs'   # optional torch.compile mode
    # compile = ['encoder', 'decoder']   # the decoder sees a different camera count per sample,
    #                                    # so it is compiled with dynamic shapes (adds nothing)
    # cuDNN SDPA re-plans a graph for every new shape (camera / point counts change each step),
    # costing ~10 ms CPU per call; it is turned off whenever autocast_bf16 is active (flash is used).
    # sdpa_cudnn = false
    # Override the V-JEPA encoder's activation checkpointing (code default: keep the model's
    # setting, which is on). false = ~2.2x total speedup at ~45 GB peak on the finetune config.
    # encoder_activation_checkpointing = false

Autocast parts (module-name prefixes on TrackerEncoder):
    encoder       scene_encoder.encoder (V-JEPA ViT; per-camera 3072 tokens x 1024)
    kv_proj       scene_encoder.kv_proj (4096 -> 512 scene projection)
    query         query_encoder (patch CNN + fusion MLP)
    decoder       decoder transformer: camera/temporal/cross attention + MLPs (512 wide)
    heads         decoder output heads (grid logits etc.)
Each listed part runs its forward under torch.autocast(bf16) and returns float32, so the residual
stream, heads, losses and all camera geometry stay fp32. bf16 for kv_proj/query/heads costs
accuracy (+11% train loss in the benchmark); keep it to encoder+decoder.
"""

import functools

import torch

AUTOCAST_PARTS = ('encoder', 'kv_proj', 'query', 'decoder', 'heads')
COMPILE_PARTS = ('encoder', 'decoder')

DEFAULTS = {
    'enabled': True,
    'autocast_bf16': ['encoder', 'decoder'],
    'compile': ['encoder'],
    'compile_mode': None,
    'fused_grad_norm': True,
    'sdpa_cudnn': None,                         # None = off iff autocast_bf16 is active
    'encoder_activation_checkpointing': None,   # None = keep the model's setting
}
_OFF = {'autocast_bf16': [], 'compile': [], 'fused_grad_norm': False}

_DEC_ATTN = ('camera_attns', 'temporal_attns', 'cross_attns')


def speed_options(config):
    """Effective [training.speed] options: DEFAULTS overridden by the config section.
    ``enabled = false`` turns autocast, compile and fused grad norm off (other keys still apply)."""
    training = config.get('training', {}) if isinstance(config, dict) else getattr(config, 'training', {})
    section = dict((training or {}).get('speed', None) or {})
    opts = dict(DEFAULTS)
    if not section.get('enabled', True):
        opts.update(_OFF)
    opts.update(section)
    for k in ('autocast_bf16', 'compile'):
        opts[k] = list(opts[k] or [])
    return opts


def bf16_available(device_type):
    return (device_type == 'cuda' and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported())


def compile_available(device_type):
    return device_type == 'cuda' and torch.cuda.is_available() and hasattr(torch, 'compile')


def _to_float(x):
    if torch.is_tensor(x):
        return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
    if isinstance(x, tuple):
        return tuple(_to_float(v) for v in x)
    if isinstance(x, list):
        return [_to_float(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_float(v) for k, v in x.items()}
    return x


def _autocast_forward(orig_forward, *args, **kwargs):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = orig_forward(*args, **kwargs)
    return _to_float(out)


def _wrap_autocast(module):
    """Run `module`'s forward under bf16 autocast and hand float32 back (instance-level patch,
    so parameter names / state_dict keys are unchanged)."""
    module.forward = functools.partial(_autocast_forward, module.forward)


def _decoder_block_modules(decoder):
    mods = []
    for name in _DEC_ATTN + ('mlps',):
        ml = getattr(decoder, name, None)
        if ml is not None:
            mods.extend(list(ml))
    return mods


def _head_modules(decoder):
    mods = []
    for name, child in decoder.named_children():
        if name.startswith('heads_') and child is not None:
            mods.extend(list(child))
    return mods


def apply_training_speedups(model, config, device_type=None, verbose=True):
    """Apply the effective `[training.speed]` options to an (unwrapped) TrackerEncoder in place.
    Call before fabric.setup / optimizer construction. `device_type` is the training device
    ('cuda' / 'cpu'; default: 'cuda' if available). Returns a short description dict."""
    if device_type is None:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    speed = speed_options(config)
    info = {}

    ckpt = speed.get('encoder_activation_checkpointing', None)
    if ckpt is not None:
        model.scene_encoder.encoder.use_activation_checkpointing = bool(ckpt)
        info['encoder_activation_checkpointing'] = bool(ckpt)

    comp = set(speed['compile'])
    if comp - set(COMPILE_PARTS):
        raise ValueError(f'unknown compile parts {sorted(comp - set(COMPILE_PARTS))}')
    ac = set(speed['autocast_bf16'])
    if ac - set(AUTOCAST_PARTS):
        raise ValueError(f'unknown autocast_bf16 parts {sorted(ac - set(AUTOCAST_PARTS))}')

    if comp and not compile_available(device_type):
        info['compile_skipped'] = f'torch.compile unavailable on {device_type}'
        comp = set()
    if ac and not bf16_available(device_type):
        info['autocast_bf16_skipped'] = f'no bf16 support on {device_type}'
        ac = set()

    sdpa_cudnn = speed.get('sdpa_cudnn', None)
    if sdpa_cudnn is None:
        sdpa_cudnn = not ac
    if not sdpa_cudnn:
        torch.backends.cuda.enable_cudnn_sdp(False)
        info['sdpa_cudnn'] = False

    # 1) per-block compile
    cmode = speed.get('compile_mode', None)  # None | 'max-autotune-no-cudagraphs' | ...
    ckw = {'mode': cmode} if cmode else {}
    if 'encoder' in comp:
        for blk in model.scene_encoder.encoder.blocks:
            blk.compile(**ckw)
    if 'decoder' in comp:
        for m in _decoder_block_modules(model.decoder):
            m.compile(dynamic=True, **ckw)
    if comp:
        info['compile'] = sorted(comp)

    # 2) scoped bf16 autocast
    if 'encoder' in ac:
        _wrap_autocast(model.scene_encoder.encoder)
    if 'kv_proj' in ac and getattr(model.scene_encoder, 'kv_proj', None) is not None:
        _wrap_autocast(model.scene_encoder.kv_proj)
    if 'query' in ac:
        _wrap_autocast(model.query_encoder)
    if 'decoder' in ac:
        for m in _decoder_block_modules(model.decoder):
            _wrap_autocast(m)
    if 'heads' in ac:
        for m in _head_modules(model.decoder):
            _wrap_autocast(m)
    if ac:
        info['autocast_bf16'] = sorted(ac)
    info['fused_grad_norm'] = bool(speed['fused_grad_norm'])

    if verbose:
        print(f'[speed] {info}')
    return info


# --------------------------------------------------------------------------------------------
# Profiling helpers (training.profile_dir): forward ranges per network part + op/kernel tables.

PROFILE_PARTS = {
    'fwd/encoder_vit': 'scene_encoder.encoder',
    'fwd/kv_proj': 'scene_encoder.kv_proj',
    'fwd/query_encoder': 'query_encoder',
    'fwd/decoder': 'decoder',
}


def _add_range_hooks(root, name, module):
    rf = torch.profiler.record_function

    def pre(mod, args):
        ctx = rf(name)
        ctx.__enter__()
        mod._tt_prof_ctx = getattr(mod, '_tt_prof_ctx', []) + [ctx]

    def post(mod, args, out):
        mod._tt_prof_ctx.pop().__exit__(None, None, None)

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def start_profiler(model, profile_dir, wait=30, active=6):
    import os
    os.makedirs(profile_dir, exist_ok=True)
    base = getattr(model, 'module', model)
    base = getattr(base, '_forward_module', base)
    base = getattr(base, 'module', base)
    for rname, path in PROFILE_PARTS.items():
        mod = base
        try:
            for a in path.split('.'):
                mod = getattr(mod, a)
        except AttributeError:
            continue
        if mod is not None:
            _add_range_hooks(base, rname, mod)
    # decoder sub-blocks (camera/temporal/cross attention, MLP) and heads
    dec = getattr(base, 'decoder', None)
    if dec is not None:
        for nm in _DEC_ATTN + ('mlps',):
            for m in getattr(dec, nm, []) or []:
                _add_range_hooks(base, f'fwd/decoder.{nm}', m)
        for m in _head_modules(dec):
            _add_range_hooks(base, 'fwd/decoder.heads', m)
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=2, active=active, repeat=1),
        record_shapes=True, profile_memory=False, with_stack=False)
    prof.start()
    return prof


def finish_profiler(prof, profile_dir, trace=False):
    import os
    prof.stop()
    ka = prof.key_averages()
    with open(os.path.join(profile_dir, 'ops_by_cuda_time.txt'), 'w') as f:
        f.write(ka.table(sort_by='cuda_time_total', row_limit=80, max_name_column_width=80))
    with open(os.path.join(profile_dir, 'ops_by_shape.txt'), 'w') as f:
        f.write(prof.key_averages(group_by_input_shape=True).table(
            sort_by='cuda_time_total', row_limit=120, max_name_column_width=60,
            max_shapes_column_width=120))
    if not trace:
        print(f'[profile] wrote {profile_dir}')
        return
    try:
        prof.export_chrome_trace(os.path.join(profile_dir, 'trace.json'))
    except Exception as e:  # trace export is best-effort
        print(f'[profile] trace export failed: {e}')
    print(f'[profile] wrote {profile_dir}')
