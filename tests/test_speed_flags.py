"""[training.speed]: agreed defaults when absent, `enabled = false` restores fp32 eager, CPU falls back."""
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from tracktail.model import speed


class _Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.use_activation_checkpointing = True

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene_encoder = nn.Module()
        self.scene_encoder.encoder = _Enc()
        self.decoder = nn.Module()
        self.decoder.mlps = nn.ModuleList([nn.Linear(8, 8)])


def _cfg(section=None):
    return SimpleNamespace(training={} if section is None else {'speed': section})


def test_defaults_when_absent():
    o = speed.speed_options(_cfg())
    assert o['autocast_bf16'] == ['encoder', 'decoder']
    assert o['compile'] == ['encoder']
    assert o['fused_grad_norm'] is True
    assert o['encoder_activation_checkpointing'] is None   # model default (on) is kept
    assert 'float8' not in o


def test_enabled_false_restores_old_behaviour():
    o = speed.speed_options(_cfg({'enabled': False}))
    assert o['autocast_bf16'] == [] and o['compile'] == [] and o['fused_grad_norm'] is False


def test_single_key_override():
    o = speed.speed_options(_cfg({'compile': []}))
    assert o['compile'] == [] and o['autocast_bf16'] == ['encoder', 'decoder']


def test_cpu_falls_back_to_fp32_eager():
    m = _Model()
    before = m.scene_encoder.encoder.forward
    info = speed.apply_training_speedups(m, _cfg(), device_type='cpu', verbose=False)
    assert 'compile' not in info and 'autocast_bf16' not in info
    assert 'compile_skipped' in info and 'autocast_bf16_skipped' in info
    assert m.scene_encoder.encoder.forward == before
    assert m.scene_encoder.encoder.use_activation_checkpointing is True
    assert torch.backends.cuda.cudnn_sdp_enabled()


def test_activation_checkpointing_override():
    m = _Model()
    info = speed.apply_training_speedups(m, _cfg({'encoder_activation_checkpointing': False}),
                                         device_type='cpu', verbose=False)
    assert m.scene_encoder.encoder.use_activation_checkpointing is False
    assert info['encoder_activation_checkpointing'] is False


@pytest.mark.parametrize('name', ['pretrain', 'finetune'])
def test_shipped_paper_configs_disable_act_ckpt(name):
    import toml, os
    path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'paper', f'{name}.toml')
    o = speed.speed_options(SimpleNamespace(training=toml.load(path)['training']))
    assert o['encoder_activation_checkpointing'] is False
    assert o['autocast_bf16'] == ['encoder', 'decoder'] and o['compile'] == ['encoder']


def test_autocast_wrapper_returns_fp32_and_keeps_state_dict():
    m = nn.Sequential(nn.Linear(8, 8))
    keys = list(m.state_dict())
    speed._wrap_autocast(m)
    y = m(torch.randn(2, 8))
    assert y.dtype == torch.float32
    assert list(m.state_dict()) == keys
