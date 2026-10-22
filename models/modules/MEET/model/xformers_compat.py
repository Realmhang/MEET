#!/usr/bin/python
# -*- coding:utf-8 -*-
"""xformers compatibility layer for DCU/HIP (no usable xformers build).

Provides BlockDiagonalMask + memory_efficient_attention with mathematically
equivalent semantics.

Backend selection (v4):
1. flash-attn varlen: packed layout, no padding waste. Requires
   same head_dim for q/k/v (zero-padded to max(Dqk,Dv) rounded to multiple of 8)
   and head_dim <= 256, dtype bf16/fp16. Used for VAE (Dqk=73/Dv=80).
2. custom triton varlen kernel (models/modules/MEET/model/triton_mha.py):
   supports Dqk/Dv <= 288 via 256+32 head-dim chunking, fwd+bwd, bf16/fp16.
   Used for LDM tiers whose padded head dim exceeds 256 (Dqk=265/Dv=272),
   where FA2/SDPA flash are unavailable on DCU.
3. per-sequence SDPA fallback: correct for any head dim / dtype; used when
   triton is unavailable, dropout != 0, or inputs are fp32 without cast.
   (v1 padded the whole batch to Lmax, wasting compute on length-skewed
   peptide batches; v2+ removes that.)
"""
import math
import os

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func as _fa_varlen
except Exception:
    _fa_varlen = None

try:
    from . import triton_mha as _triton_mha
except Exception:
    try:
        import triton_mha as _triton_mha
    except Exception:
        _triton_mha = None


class _TritonVarlenAttn(torch.autograd.Function):
    """Autograd wrapper around the custom triton varlen kernel (fwd + tiled bwd)."""

    @staticmethod
    def forward(ctx, q, k, v, cu, scale, seqlens, max_len, static_key=None, n2max=None):
        q_, k_, v_ = q.contiguous(), k.contiguous(), v.contiguous()
        o, lse = _triton_mha.mha_varlen_fwd(q_, k_, v_, cu, scale, max_len=max_len, bm=64, bn=16)
        ctx.save_for_backward(q_, k_, v_, o, lse, cu)
        ctx.scale = scale
        ctx.seqlens = seqlens  # tuple of python ints (for the tiled bwd)
        ctx.max_len = max_len
        ctx.static_key = static_key
        ctx.n2max = n2max
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, cu = ctx.saved_tensors
        if ctx.seqlens is not None:
            dq, dk, dv = _triton_mha.mha_varlen_bwd_tiled5(
                q, k, v, o, lse, do.contiguous(), cu, list(ctx.seqlens), ctx.scale,
                max_len=ctx.max_len)
        else:
            dq, dk, dv = _triton_mha.mha_varlen_bwd(
                q, k, v, o, lse, do.contiguous(), cu, ctx.scale)
        return dq, dk, dv, None, None, None, None, None, None

# Baseline-reproduction mode: emulates the customer's "per-segment PyTorch math
# SDPA" reference implementation. Set MEET_BASELINE_MODE=1 (fp16 feature cast,
# as described by the customer) or MEET_BASELINE_MODE=2 (pure fp32, no cast).
_BASELINE_MODE = os.environ.get('MEET_BASELINE_MODE', '0')
_GRAPH = os.environ.get('MEET_GRAPH', '0') == '1'


class BlockDiagonalMask:
    def __init__(self, seqlens):
        self._seqlens = [int(s) for s in seqlens if int(s) > 0]

    @classmethod
    def from_seqlens(cls, seqlens_q, seqlens_kv=None, device=None):
        if seqlens_kv is not None:
            kv = [int(s) for s in seqlens_kv if int(s) > 0]
            if kv != [int(s) for s in seqlens_q if int(s) > 0]:
                raise NotImplementedError('xformers_compat: q/kv seqlens must match')
        return cls(seqlens_q)


def _flat(x):
    if x.dim() == 4:
        assert x.shape[0] == 1, f'expected packed layout, got batch={x.shape[0]}'
        return x.squeeze(0)
    assert x.dim() == 3, f'expected 3D/4D tensor, got {x.dim()}D'
    return x


# cu_seqlens cache: within one training step all 12+ layer calls share the same
# seqlens, so build cu (one HtoD + cumsum) once per step instead of per call.
# max_len comes from the CPU-side python list: no device sync (.item()) at all.
_CU_CACHE = {}
_DUMP = os.environ.get('MEET_DUMP_SEQ', '')


def _get_cu(seqlens, device):
    key = (tuple(seqlens), str(device))
    hit = _CU_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_CU_CACHE) > 128:
        _CU_CACHE.clear()
    cu = torch.zeros(len(seqlens) + 1, device=device, dtype=torch.int32)
    torch.cumsum(torch.tensor(seqlens, device=device, dtype=torch.int32), 0, out=cu[1:])
    val = (cu, max(seqlens))
    _CU_CACHE[key] = val
    if _DUMP:
        with open(_DUMP, 'a') as f:
            f.write(repr(seqlens) + '\n')
    return val


def memory_efficient_attention(q, k, v, scale=None, attn_bias=None, p=0.0):
    """Packed varlen attention, equivalent to xops.memory_efficient_attention
    with a BlockDiagonalMask.

    q, k: [N, H, Dqk] or [1, N, H, Dqk]
    v:    [N, H, Dv]  or [1, N, H, Dv]
    returns: [N, H, Dv]
    """
    if not isinstance(attn_bias, BlockDiagonalMask):
        raise NotImplementedError('xformers_compat: only BlockDiagonalMask attn_bias is supported')
    seqlens = attn_bias._seqlens
    if len(seqlens) == 0:
        raise ValueError('xformers_compat: empty seqlens')

    qf, kf, vf = _flat(q), _flat(k), _flat(v)
    N, H, Dqk = qf.shape
    Dv = vf.shape[-1]
    assert kf.shape[-1] == Dqk and kf.shape[0] == N and vf.shape[0] == N, 'shape mismatch'
    assert sum(seqlens) == N, f'seqlens sum {sum(seqlens)} != tokens {N}'
    D = max(Dqk, Dv)
    D8 = (D + 7) // 8 * 8

    # v3: fp32 inputs (e.g. InitLayer's coords-derived v, which pad/expand keep
    # in fp32 even under autocast) are cast to bf16 to match the customer's
    # reference implementation ("feature cast to fp16"). Model trains under
    # enable_mixed_precision anyway; all MEETAttnLayer attention is already bf16.
    if qf.dtype == torch.float32 and qf.is_cuda and _BASELINE_MODE != '2':
        qf, kf, vf = (qf.half(), kf.half(), vf.half()) if _BASELINE_MODE == '1' \
            else (qf.bfloat16(), kf.bfloat16(), vf.bfloat16())

    # ---- baseline mode: per-segment SDPA with forced MATH backend ----
    if _BASELINE_MODE in ('1', '2'):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        out_parts = []
        off = 0
        for s in seqlens:
            qs = qf[off:off + s].transpose(0, 1).unsqueeze(0)
            ks = kf[off:off + s].transpose(0, 1).unsqueeze(0)
            vs = vf[off:off + s].transpose(0, 1).unsqueeze(0)
            with sdpa_kernel([SDPBackend.MATH]):
                o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None, dropout_p=p, scale=scale)
            out_parts.append(o.squeeze(0).transpose(0, 1))
            off += s
        return torch.cat(out_parts, dim=0)

    # ---- FA2 varlen path: no padding waste, requires D8<=256 and half dtype ----
    if (
        _fa_varlen is not None
        and qf.is_cuda
        and D8 <= 256
        and qf.dtype in (torch.bfloat16, torch.float16)
        and vf.dtype == qf.dtype
        and kf.dtype == qf.dtype
    ):
        dev = qf.device
        cu, max_len = _get_cu(seqlens, dev)

        def _pad(x, d_src):
            if d_src == D8:
                return x
            out = F.pad(x, (0, D8 - d_src))
            return out

        out = _fa_varlen(
            _pad(qf, Dqk), _pad(kf, Dqk), _pad(vf, Dv),
            cu, cu, N, N,
            softmax_scale=scale, dropout_p=p if p > 0 else 0.0, causal=False,
        )
        return out[..., :Dv] if Dv < D8 else out

    # ---- custom triton varlen kernel: Dqk/Dv <= 288, half dtype, no dropout ----
    if (
        _triton_mha is not None
        and qf.is_cuda
        and qf.dtype in (torch.bfloat16, torch.float16)
        and kf.dtype == qf.dtype
        and vf.dtype == qf.dtype
        and Dqk <= 288 and Dv <= 288
        and p == 0.0
        and D8 > 256  # <=256 goes to FA2 above when available
    ):
        dev = qf.device
        cu, max_len = _get_cu(seqlens, dev)
        eff_scale = scale if scale is not None else 1.0 / math.sqrt(Dqk)
        if _GRAPH:
            # graph-capture mode: static-keyed bwd buffers, constant n2max stride,
            # max_len comes from the (driver-seeded) cache as a bucket bound.
            static_key = ('bwd', N)
            n2max = N * max_len
            return _TritonVarlenAttn.apply(qf, kf, vf, cu, eff_scale, tuple(seqlens),
                                           max_len, static_key, n2max)
        return _TritonVarlenAttn.apply(qf, kf, vf, cu, eff_scale, tuple(seqlens), max_len)

    # ---- fallback: per-sequence SDPA (no pad waste; correct for any head dim) ----
    out_parts = []
    off = 0
    for s in seqlens:
        qs = qf[off:off + s].transpose(0, 1).unsqueeze(0)  # [1, H, s, Dqk]
        ks = kf[off:off + s].transpose(0, 1).unsqueeze(0)
        vs = vf[off:off + s].transpose(0, 1).unsqueeze(0)  # [1, H, s, Dv]
        o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None, dropout_p=p, scale=scale)
        out_parts.append(o.squeeze(0).transpose(0, 1))
        off += s
    return torch.cat(out_parts, dim=0)
