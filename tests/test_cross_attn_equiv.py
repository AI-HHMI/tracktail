#!/usr/bin/env python
"""Unit-level equivalence: nn.MultiheadAttention -> DecoupledCrossAttention via the
load-time converter _convert_cross_attn. Covers the three checkpoint regimes:

  fused : kdim==embed_dim   (in_proj_weight)        cross_attn_dim==latent_dim -> EXACT
  split : kdim!=embed_dim   ({q,k,v}_proj_weight)   cross_attn_dim==latent_dim -> EXACT
  widen : kdim==embed_dim   cross_attn_dim > latent_dim -> function-preserving warm-start

No VJEPA / GPU needed.
"""
import torch
import torch.nn as nn

from tracktail.model.encoder_decoder import DecoupledCrossAttention
from tracktail.model.train_utils import _convert_cross_attn


class _Stub(nn.Module):
    """Minimal module exposing model.decoder.cross_attns[i] so the converter can read
    target shapes (state_dict keys) and num_heads."""
    def __init__(self, xattn):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.cross_attns = nn.ModuleList([xattn])


def _check(name, latent, kv, cross_attn_dim, num_heads, atol):
    torch.manual_seed(0)
    mha = nn.MultiheadAttention(embed_dim=latent, num_heads=num_heads,
                                kdim=kv, vdim=kv, dropout=0.0, batch_first=True)
    mha.eval()

    param_dict = {f'decoder.cross_attns.0.{k}': v.clone()
                  for k, v in mha.state_dict().items()}

    xattn = DecoupledCrossAttention(latent, kv, cross_attn_dim, num_heads)
    stub = _Stub(xattn)
    converted = _convert_cross_attn(param_dict, stub)
    missing, unexpected = stub.load_state_dict(converted, strict=False)
    assert not missing, f'{name}: missing {missing}'
    assert not unexpected, f'{name}: unexpected {unexpected}'

    xattn.eval()
    B, Lq, Lk = 2, 7, 11
    q = torch.randn(B, Lq, latent)
    kv_in = torch.randn(B, Lk, kv)
    with torch.no_grad():
        out_mha, _ = mha(q, kv_in, kv_in, need_weights=False)
        out_new = xattn(q, kv_in)
    maxd = (out_mha - out_new).abs().max().item()
    ok = torch.allclose(out_mha, out_new, atol=atol, rtol=atol)
    print(f'{"PASS" if ok else "FAIL"} {name}: max|Δ|={maxd:.3e} '
          f'(latent={latent} kv={kv} cad={cross_attn_dim} H={num_heads})')
    return ok


def main():
    ok = True
    ok &= _check('fused-exact', latent=512, kv=512,  cross_attn_dim=512,  num_heads=8, atol=1e-5)
    ok &= _check('split-exact', latent=512, kv=1024, cross_attn_dim=512,  num_heads=8, atol=1e-5)
    ok &= _check('widen-fused', latent=512, kv=512,  cross_attn_dim=1024, num_heads=8, atol=1e-4)
    ok &= _check('widen-split', latent=512, kv=1024, cross_attn_dim=1024, num_heads=8, atol=1e-4)
    print('\n=== CROSS-ATTN EQUIV:', 'PASS ===' if ok else 'FAIL ===')
    import sys
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
