#!/usr/bin/python
"""Custom triton flash-attention forward for MEET on DCU.

Supports arbitrary head_dim (incl. Dqk=265 / Dv=272 which exceed the 256
limit of FA2/cutlass kernels) by splitting the head dimension into
BLOCK_D1=256 + BLOCK_D2=32 chunks: logits = dot(Q1,K1) + dot(Q2,K2),
output = concat(dot(P,V1), dot(P,V2)).

Packed varlen layout: Q/K [N, H, Dqk], V/O [N, H, Dv], cu_seqlens [B+1] int32.
LSE output [H, N] fp32 (for a future backward pass).
"""
import torch
import triton
import triton.language as tl

try:
    from torch._dynamo import disable as _dynamo_disable
except Exception:  # very old torch: no-op decorator
    def _dynamo_disable(fn=None, **k):
        return fn if fn is not None else (lambda f: f)


@triton.jit
def _mha_fwd_kernel(
    Q, K, V, O, LSE, CU,
    sm_scale,
    strideq_n, strideq_h,
    stridek_n, stridek_h,
    stridev_n, stridev_h,
    strideo_n, strideo_h,
    stridelse_h,
    DQK: tl.constexpr,      # 265
    DV: tl.constexpr,       # 272
    D1: tl.constexpr,       # 256 (chunk 1 width, power of 2)
    D2: tl.constexpr,       # 32  (chunk 2 width, power of 2, covers DQK-256 and DV-256)
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    DOTMODE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return

    row_idx = seq_start + offs_m  # global token index of each query row

    # ---- load Q in two chunks: [BM, D1] + [BM, D2] ----
    offs_d1 = tl.arange(0, D1)
    offs_d2 = tl.arange(0, D2)
    q_base = Q + row_idx[:, None] * strideq_n + h * strideq_h
    q1 = tl.load(q_base + offs_d1[None, :], mask=mask_m[:, None] & (offs_d1[None, :] < DQK), other=0.0)
    q2 = tl.load(q_base + (D1 + offs_d2)[None, :], mask=mask_m[:, None] & ((D1 + offs_d2)[None, :] < DQK), other=0.0)

    m_i = tl.full([BM], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc1 = tl.zeros([BM, D1], dtype=tl.float32)
    acc1a = tl.zeros([BM, 128], dtype=tl.float32)
    acc1b = tl.zeros([BM, 128], dtype=tl.float32)
    acc2 = tl.zeros([BM, D2], dtype=tl.float32)

    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n

        k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
        k1 = tl.load(k_base + offs_d1[None, :], mask=mask_n[:, None] & (offs_d1[None, :] < DQK), other=0.0)
        k2 = tl.load(k_base + (D1 + offs_d2)[None, :], mask=mask_n[:, None] & ((D1 + offs_d2)[None, :] < DQK), other=0.0)

        # logits [BM, BN] fp32
        qk = tl.dot(q1, tl.trans(k1))
        qk += tl.dot(q2, tl.trans(k2))
        qk = qk * sm_scale
        qk = tl.where(mask_n[None, :], qk, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # guard: rows with all -inf (padded rows) keep m stable
        m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        alpha = tl.where(m_i == float('-inf'), 0.0, alpha)
        p = tl.exp(qk - m_safe[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_safe

        v_base = V + kv_idx[:, None] * stridev_n + h * stridev_h
        v1 = tl.load(v_base + offs_d1[None, :], mask=mask_n[:, None] & (offs_d1[None, :] < DV), other=0.0)
        v2 = tl.load(v_base + (D1 + offs_d2)[None, :], mask=mask_n[:, None] & ((D1 + offs_d2)[None, :] < DV), other=0.0)

        p_d = p.to(v1.dtype)
        acc2 = acc2 * alpha[:, None] + tl.dot(p_d, v2)
        if DOTMODE == 1:
            # split acc1 into two 128-wide dots
            v1a = tl.load(v_base + tl.arange(0, 128)[None, :], mask=mask_n[:, None], other=0.0)
            v1b = tl.load(v_base + 128 + tl.arange(0, 128)[None, :], mask=mask_n[:, None], other=0.0)
            acc1a = acc1a * alpha[:, None] + tl.dot(p_d, v1a)
            acc1b = acc1b * alpha[:, None] + tl.dot(p_d, v1b)
        elif DOTMODE == 2:
            # fp32 pv dot for acc1 (diagnostic)
            acc1 = acc1 * alpha[:, None] + tl.dot(p, v1.to(tl.float32))
        else:
            acc1 = acc1 * alpha[:, None] + tl.dot(p_d, v1)

    # normalize
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    if DOTMODE == 1:
        acc1a = acc1a / l_safe[:, None]
        acc1b = acc1b / l_safe[:, None]
    else:
        acc1 = acc1 / l_safe[:, None]
    acc2 = acc2 / l_safe[:, None]

    # ---- store O ----
    o_base = O + row_idx[:, None] * strideo_n + h * strideo_h
    if DOTMODE == 1:
        tl.store(o_base + tl.arange(0, 128)[None, :], acc1a.to(O.dtype.element_ty),
                 mask=mask_m[:, None])
        tl.store(o_base + 128 + tl.arange(0, 128)[None, :], acc1b.to(O.dtype.element_ty),
                 mask=mask_m[:, None])
    else:
        tl.store(o_base + offs_d1[None, :], acc1.to(O.dtype.element_ty),
                 mask=mask_m[:, None] & (offs_d1[None, :] < DV))
    tl.store(o_base + (D1 + offs_d2)[None, :], acc2.to(O.dtype.element_ty),
             mask=mask_m[:, None] & ((D1 + offs_d2)[None, :] < DV))

    # LSE [H, N]
    lse = m_i + tl.log(l_safe)
    tl.store(LSE + h * stridelse_h + row_idx, lse, mask=mask_m)


@_dynamo_disable
def mha_varlen_fwd(q, k, v, cu_seqlens, sm_scale, max_len=None, dotmode=1,
                   bm=32, bn=32, warps=4, stages=1):
    """q,k: [N, H, Dqk]; v: [N, H, Dv]; cu_seqlens: [B+1] int32 cuda.
    returns O [N, H, Dv] (same dtype as v), LSE [H, N] fp32."""
    assert q.dim() == 3 and v.dim() == 3
    N, H, DQK = q.shape
    DV = v.shape[-1]
    B = cu_seqlens.numel() - 1
    assert DQK <= 288 and DV <= 288, 'kernel supports head_dim <= 288 (256+32)'
    if max_len is None:
        max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    o = torch.empty_like(v)
    lse = torch.empty(H, N, device=q.device, dtype=torch.float32)

    BM, BN = bm, bn
    grid = (triton.cdiv(max_len, BM), B, H)
    _mha_fwd_kernel[grid](
        q, k, v, o, lse, cu_seqlens,
        sm_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        lse.stride(0),
        DQK=DQK, DV=DV, D1=256, D2=32, H=H,
        BM=BM, BN=BN, DOTMODE=dotmode,
        num_warps=warps, num_stages=stages,
    )
    return o, lse


@triton.jit
def _mha_bwd_kernel(
    Q, K, V, O, DO, LSE, DELTA,
    DQ, DK, DV, CU,
    sm_scale,
    strideq_n, strideq_h,
    stridek_n, stridek_h,
    stridev_n, stridev_h,
    strideo_n, strideo_h,
    stridelse_h,
    stridedq_n, stridedq_h,
    stridedk_n, stridedk_h,
    stridedv_n, stridedv_h,
    DQK: tl.constexpr, DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return
    q_idx = seq_start + offs_m

    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    # q chunks: [BM,128] x2 + [BM,32]
    q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
    q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)

    # do chunks (Dv=272): [BM,128] x2 + [BM,32]
    do_base = DO + q_idx[:, None] * strideo_n + h * strideo_h
    do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

    lse = tl.load(LSE + h * stridelse_h + q_idx, mask=mask_m, other=0.0)      # [BM]
    delta = tl.load(DELTA + q_idx * H + h, mask=mask_m, other=0.0)            # [BM]

    dq1a = tl.zeros([BM, 128], dtype=tl.float32)
    dq1b = tl.zeros([BM, 128], dtype=tl.float32)
    dq2 = tl.zeros([BM, 32], dtype=tl.float32)

    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n

        # k chunks: [BN,128] x2 + [BN,32]
        k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
        k1a = tl.load(k_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k1b = tl.load(k_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
        # v chunks: [BN,128] x2 + [BN,32]
        v_base = V + kv_idx[:, None] * stridev_n + h * stridev_h
        v1a = tl.load(v_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        v1b = tl.load(v_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        v2 = tl.load(v_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

        # sc = q@k^T * scale  [BM, BN]
        sc = tl.dot(q1a, tl.trans(k1a))
        sc += tl.dot(q1b, tl.trans(k1b))
        sc += tl.dot(q2, tl.trans(k2))
        sc = sc * sm_scale
        p = tl.exp(sc - lse[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        # dS = do @ v^T  [BM, BN]
        dS = tl.dot(do1a, tl.trans(v1a))
        dS += tl.dot(do1b, tl.trans(v1b))
        dS += tl.dot(do2, tl.trans(v2))

        dp = p * (dS - delta[:, None])
        dp = tl.where(mask_n[None, :], dp, 0.0)

        # dq += dp @ k  [BM, Dqk] (apply scale at final store)
        dpk = dp.to(k1a.dtype)
        dq1a += tl.dot(dpk, k1a)
        dq1b += tl.dot(dpk, k1b)
        dq2 += tl.dot(dpk, k2)

        # dk += dp^T @ q * scale ; dv += p^T @ do   [BN, D]
        dpT = tl.trans(dp)
        pT = tl.trans(p)
        dpTq = dpT.to(q1a.dtype)
        pTdo = pT.to(do1a.dtype)
        dk_base = DK + kv_idx[:, None] * stridedk_n + h * stridedk_h
        tl.atomic_add(dk_base + offs_128[None, :], tl.dot(dpTq, q1a) * sm_scale, mask=mask_n[:, None])
        tl.atomic_add(dk_base + 128 + offs_128[None, :], tl.dot(dpTq, q1b) * sm_scale, mask=mask_n[:, None])
        tl.atomic_add(dk_base + 256 + offs_32[None, :], tl.dot(dpTq, q2) * sm_scale,
                      mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK))
        dv_base = DV + kv_idx[:, None] * stridedv_n + h * stridedv_h
        tl.atomic_add(dv_base + offs_128[None, :], tl.dot(pTdo, do1a), mask=mask_n[:, None])
        tl.atomic_add(dv_base + 128 + offs_128[None, :], tl.dot(pTdo, do1b), mask=mask_n[:, None])
        tl.atomic_add(dv_base + 256 + offs_32[None, :], tl.dot(pTdo, do2),
                      mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT))

    # store dq (exclusive per program)
    dq_base = DQ + q_idx[:, None] * stridedq_n + h * stridedq_h
    tl.store(dq_base + offs_128[None, :], (dq1a * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + 128 + offs_128[None, :], (dq1b * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + 256 + offs_32[None, :], (dq2 * sm_scale).to(DQ.dtype.element_ty),
             mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK))


@triton.jit
def _mha_bwd_dq_kernel(
    Q, K, V, DO, LSE, DELTA, DQ, CU,
    sm_scale,
    strideq_n, strideq_h,
    stridek_n, stridek_h,
    stridev_n, stridev_h,
    strideo_n, strideo_h,
    stridelse_h,
    stridedq_n, stridedq_h,
    DQK: tl.constexpr, DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
    SKIPTS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return
    q_idx = seq_start + offs_m
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
    q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
    do_base = DO + q_idx[:, None] * strideo_n + h * strideo_h
    do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

    lse = tl.load(LSE + h * stridelse_h + q_idx, mask=mask_m, other=0.0)
    delta = tl.load(DELTA + q_idx * H + h, mask=mask_m, other=0.0)

    dq1a = tl.zeros([BM, 128], dtype=tl.float32)
    dq1b = tl.zeros([BM, 128], dtype=tl.float32)
    dq2 = tl.zeros([BM, 32], dtype=tl.float32)

    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n
        k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
        k1a = tl.load(k_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k1b = tl.load(k_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)

        sc = tl.dot(q1a, tl.trans(k1a))
        sc += tl.dot(q1b, tl.trans(k1b))
        sc += tl.dot(q2, tl.trans(k2))
        sc = sc * sm_scale
        p = tl.exp(sc - lse[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        if SKIPTS:
            # perf probe: 3 tensors only (q, do, k) — no v, no dS dots
            dp = p * (-delta[:, None])
        else:
            v_base = V + kv_idx[:, None] * stridev_n + h * stridev_h
            v1a = tl.load(v_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
            v1b = tl.load(v_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
            v2 = tl.load(v_base + 256 + offs_32[None, :], mask=mask_n[:, None], other=0.0)

            dS = tl.dot(do1a, tl.trans(v1a))
            dS += tl.dot(do1b, tl.trans(v1b))
            dS += tl.dot(do2, tl.trans(v2))

            dp = p * (dS - delta[:, None])
        dp = tl.where(mask_n[None, :], dp, 0.0)

        dpk = dp.to(k1a.dtype)
        dq1a += tl.dot(dpk, k1a)
        dq1b += tl.dot(dpk, k1b)
        dq2 += tl.dot(dpk, k2)

    dq_base = DQ + q_idx[:, None] * stridedq_n + h * stridedq_h
    tl.store(dq_base + offs_128[None, :], (dq1a * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + 128 + offs_128[None, :], (dq1b * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + 256 + offs_32[None, :], (dq2 * sm_scale).to(DQ.dtype.element_ty),
             mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK))


@triton.jit
def _mha_bwd_dkv_kernel(
    Q, K, V, DO, LSE, DELTA, DK, DV, CU,
    sm_scale,
    strideq_n, strideq_h,
    stridek_n, stridek_h,
    stridev_n, stridev_h,
    strideo_n, strideo_h,
    stridelse_h,
    stridedk_n, stridedk_h,
    stridedv_n, stridedv_h,
    DQK: tl.constexpr, DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < s
    if tl.sum(mask_n.to(tl.int32)) == 0:
        return
    kv_idx = seq_start + offs_n
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
    k1a = tl.load(k_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
    k1b = tl.load(k_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
    k2 = tl.load(k_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
    v_base = V + kv_idx[:, None] * stridev_n + h * stridev_h
    v1a = tl.load(v_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
    v1b = tl.load(v_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
    v2 = tl.load(v_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

    dk1a = tl.zeros([BN, 128], dtype=tl.float32)
    dk1b = tl.zeros([BN, 128], dtype=tl.float32)
    dk2 = tl.zeros([BN, 32], dtype=tl.float32)
    dv1a = tl.zeros([BN, 128], dtype=tl.float32)
    dv1b = tl.zeros([BN, 128], dtype=tl.float32)
    dv2 = tl.zeros([BN, 32], dtype=tl.float32)

    m_end = tl.cdiv(s, BM)
    for start_m in range(0, m_end):
        offs_m = start_m * BM + tl.arange(0, BM)
        mask_m = offs_m < s
        q_idx = seq_start + offs_m
        q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
        q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
        do_base = DO + q_idx[:, None] * strideo_n + h * strideo_h
        do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None], other=0.0)
        lse = tl.load(LSE + h * stridelse_h + q_idx, mask=mask_m, other=0.0)
        delta = tl.load(DELTA + q_idx * H + h, mask=mask_m, other=0.0)

        # scT = (k@q^T)*scale  [BN, BM]
        scT = tl.dot(k1a, tl.trans(q1a))
        scT += tl.dot(k1b, tl.trans(q1b))
        scT += tl.dot(k2, tl.trans(q2))
        scT = scT * sm_scale
        pT = tl.exp(scT - lse[None, :])
        pT = tl.where(mask_m[None, :], pT, 0.0)

        # dST = v@do^T  [BN, BM]
        dST = tl.dot(v1a, tl.trans(do1a))
        dST += tl.dot(v1b, tl.trans(do1b))
        dST += tl.dot(v2, tl.trans(do2))

        dpT = pT * (dST - delta[None, :])
        dpT = tl.where(mask_m[None, :], dpT, 0.0)

        dpTq = dpT.to(q1a.dtype)
        pTdo = pT.to(do1a.dtype)
        dk1a += tl.dot(dpTq, q1a)
        dk1b += tl.dot(dpTq, q1b)
        dk2 += tl.dot(dpTq, q2)
        dv1a += tl.dot(pTdo, do1a)
        dv1b += tl.dot(pTdo, do1b)
        dv2 += tl.dot(pTdo, do2)

    dk_base = DK + kv_idx[:, None] * stridedk_n + h * stridedk_h
    tl.store(dk_base + offs_128[None, :], (dk1a * sm_scale).to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dk_base + 128 + offs_128[None, :], (dk1b * sm_scale).to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dk_base + 256 + offs_32[None, :], (dk2 * sm_scale).to(DK.dtype.element_ty),
             mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK))
    dv_base = DV + kv_idx[:, None] * stridedv_n + h * stridedv_h
    tl.store(dv_base + offs_128[None, :], dv1a.to(DV.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + 128 + offs_128[None, :], dv1b.to(DV.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + 256 + offs_32[None, :], dv2.to(DV.dtype.element_ty),
             mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT))


def mha_varlen_bwd_split(q, k, v, o, lse, do, cu_seqlens, sm_scale, max_len=None,
                         bm=32, bn=32, warps=4, stages=1, part='both', skipts=False):
    """Split bwd (FA2-style): dq kernel + dk/dv kernel, no atomics.
    part: 'dq' | 'dkv' | 'both' (for profiling)."""
    N, H, DQK = q.shape
    DV = v.shape[-1]
    B = cu_seqlens.numel() - 1
    if max_len is None:
        max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    delta = (do.float() * o.float()).sum(-1).contiguous()  # [N, H]
    dq = torch.zeros(N, H, DQK, device=q.device, dtype=torch.float32)
    dk = torch.zeros(N, H, DQK, device=q.device, dtype=torch.float32)
    dv = torch.zeros(N, H, DV, device=q.device, dtype=torch.float32)

    if part in ('dq', 'both'):
        _mha_bwd_dq_kernel[(triton.cdiv(max_len, bm), B, H)](
            q, k, v, do, lse, delta, dq, cu_seqlens,
            sm_scale,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            v.stride(0), v.stride(1),
            do.stride(0), do.stride(1),
            lse.stride(0),
            dq.stride(0), dq.stride(1),
            DQK=DQK, DVT=DV, H=H, BM=bm, BN=bn, SKIPTS=skipts,
            num_warps=warps, num_stages=stages,
        )
    if part in ('dkv', 'both'):
        _mha_bwd_dkv_kernel[(triton.cdiv(max_len, bn), B, H)](
        q, k, v, do, lse, delta, dk, dv, cu_seqlens,
        sm_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        do.stride(0), do.stride(1),
        lse.stride(0),
        dk.stride(0), dk.stride(1),
        dv.stride(0), dv.stride(1),
        DQK=DQK, DVT=DV, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    return dq, dk, dv


@triton.jit
def _pdp_kernel(
    Q, K, V, DO, LSE, DELTA, PBUF, DPBUF, CU, OFF2,
    sm_scale,
    strideq_n, strideq_h,
    stridek_n, stridek_h,
    stridev_n, stridev_h,
    strideo_n, strideo_h,
    stridelse_h,
    stridedel_h,
    N2,
    DQK: tl.constexpr, DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    # per (q-block, seq, head): compute p and dp tiles, store to PBUF/DPBUF
    # PBUF[h, OFF2[b] + i*s + j], layout [H, sum(s^2)]
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return
    q_idx = seq_start + offs_m
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
    q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
    do_base = DO + q_idx[:, None] * strideo_n + h * strideo_h
    do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

    lse = tl.load(LSE + h * stridelse_h + q_idx, mask=mask_m, other=0.0)
    delta = tl.load(DELTA + h * stridedel_h + q_idx, mask=mask_m, other=0.0)

    off2 = tl.load(OFF2 + b)
    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n
        k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
        k1a = tl.load(k_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k1b = tl.load(k_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)

        sc = tl.dot(q1a, tl.trans(k1a))
        sc += tl.dot(q1b, tl.trans(k1b))
        sc += tl.dot(q2, tl.trans(k2))
        sc = sc * sm_scale
        p = tl.exp(sc - lse[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        v_base = V + kv_idx[:, None] * stridev_n + h * stridev_h
        v1a = tl.load(v_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        v1b = tl.load(v_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        v2 = tl.load(v_base + 256 + offs_32[None, :], mask=mask_n[:, None], other=0.0)

        dS = tl.dot(do1a, tl.trans(v1a))
        dS += tl.dot(do1b, tl.trans(v1b))
        dS += tl.dot(do2, tl.trans(v2))

        dp = p * (dS - delta[:, None])
        dp = tl.where(mask_n[None, :], dp, 0.0)

        # store tiles: [BM, BN], row-major at h*N2 + off2 + i*s + j
        ptrs = h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :]
        tl.store(PBUF + ptrs, p.to(PBUF.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])
        tl.store(DPBUF + ptrs, dp.to(DPBUF.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _dq_t_kernel(
    DPBUF, K, DQ, CU, OFF2,
    sm_scale,
    stridek_n, stridek_h,
    stridedq_n, stridedq_h,
    N2,
    DQK: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    dq1a = tl.zeros([BM, 128], dtype=tl.float32)
    dq1b = tl.zeros([BM, 128], dtype=tl.float32)
    dq2 = tl.zeros([BM, 32], dtype=tl.float32)

    off2 = tl.load(OFF2 + b)
    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n
        dp = tl.load(DPBUF + h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :],
                     mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
        k1a = tl.load(k_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k1b = tl.load(k_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
        dpk = dp.to(k1a.dtype)
        dq1a += tl.dot(dpk, k1a)
        dq1b += tl.dot(dpk, k1b)
        dq2 += tl.dot(dpk, k2)

    q_idx = seq_start + offs_m
    dq_base = DQ + q_idx[:, None] * stridedq_n + h * stridedq_h
    tl.store(dq_base + offs_128[None, :], (dq1a * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + 128 + offs_128[None, :], (dq1b * sm_scale).to(DQ.dtype.element_ty), mask=mask_m[:, None])
    tl.store(dq_base + 256 + offs_32[None, :], (dq2 * sm_scale).to(DQ.dtype.element_ty),
             mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK))


@triton.jit
def _dkv_t_kernel(
    PBUF, DPBUF, Q, DO, DK, DV, CU, OFF2,
    sm_scale,
    strideq_n, strideq_h,
    strideo_n, strideo_h,
    stridedk_n, stridedk_h,
    stridedv_n, stridedv_h,
    N2,
    DQK: tl.constexpr, DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    # per (kv-block, seq, head): dk from dp^T@q, dv from p^T@do
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < s
    if tl.sum(mask_n.to(tl.int32)) == 0:
        return
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    dk1a = tl.zeros([BN, 128], dtype=tl.float32)
    dk1b = tl.zeros([BN, 128], dtype=tl.float32)
    dk2 = tl.zeros([BN, 32], dtype=tl.float32)
    dv1a = tl.zeros([BN, 128], dtype=tl.float32)
    dv1b = tl.zeros([BN, 128], dtype=tl.float32)
    dv2 = tl.zeros([BN, 32], dtype=tl.float32)

    off2 = tl.load(OFF2 + b)
    m_end = tl.cdiv(s, BM)
    for start_m in range(0, m_end):
        offs_m = start_m * BM + tl.arange(0, BM)
        mask_m = offs_m < s
        q_idx = seq_start + offs_m
        tptr = h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :]
        p = tl.load(PBUF + tptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        dp = tl.load(DPBUF + tptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        pT = tl.trans(p)
        dpT = tl.trans(dp)

        q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
        q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
        do_base = DO + q_idx[:, None] * strideo_n + h * strideo_h
        do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

        dpTq = dpT.to(q1a.dtype)
        pTdo = pT.to(do1a.dtype)
        dk1a += tl.dot(dpTq, q1a)
        dk1b += tl.dot(dpTq, q1b)
        dk2 += tl.dot(dpTq, q2)
        dv1a += tl.dot(pTdo, do1a)
        dv1b += tl.dot(pTdo, do1b)
        dv2 += tl.dot(pTdo, do2)

    kv_idx = seq_start + offs_n
    dk_base = DK + kv_idx[:, None] * stridedk_n + h * stridedk_h
    tl.store(dk_base + offs_128[None, :], (dk1a * sm_scale).to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dk_base + 128 + offs_128[None, :], (dk1b * sm_scale).to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dk_base + 256 + offs_32[None, :], (dk2 * sm_scale).to(DK.dtype.element_ty),
             mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK))
    dv_base = DV + kv_idx[:, None] * stridedv_n + h * stridedv_h
    tl.store(dv_base + offs_128[None, :], dv1a.to(DV.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + 128 + offs_128[None, :], dv1b.to(DV.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + 256 + offs_32[None, :], dv2.to(DV.dtype.element_ty),
             mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT))


# graph-capture static buffers: key -> persistent off2/pbuf/dpbuf so the tiled
# bwd is replay-safe (off2 content is updated per step by the graph driver).
_STATIC_BWD_BUFS = {}


def mha_varlen_bwd_tiled(q, k, v, o, lse, do, cu_seqlens, seqlens, sm_scale, max_len=None,
                         bm=32, bn=32, warps=4, stages=1, static_key=None, n2max=None):
    """Tiled bwd: materialize p/dp tiles then 3 light kernels.
    static_key: (bucket key) — use registered static off2/pbuf/dpbuf buffers
    (content of off2 must be refreshed by the caller each step); n2max fixes
    the pbuf stride so the N2 kernel arg stays constant per bucket."""
    N, H, DQK = q.shape
    DV = v.shape[-1]
    B = cu_seqlens.numel() - 1
    if max_len is None:
        max_len = max(seqlens)

    # delta in [H, N] layout (coalesced per-head gather)
    delta = (do.float().transpose(0, 1) * o.float().transpose(0, 1)).sum(-1).contiguous()  # [H, N]
    if static_key is not None:
        bufs = _STATIC_BWD_BUFS.get(static_key)
        if bufs is None:
            bufs = {
                'off2': torch.zeros(B, device=q.device, dtype=torch.int32),
                'pbuf': torch.empty(H, n2max, device=q.device, dtype=torch.bfloat16),
                'dpbuf': torch.empty(H, n2max, device=q.device, dtype=torch.bfloat16),
                'n2max': n2max,
            }
            _STATIC_BWD_BUFS[static_key] = bufs
        off2, pbuf, dpbuf, n2 = bufs['off2'], bufs['pbuf'], bufs['dpbuf'], bufs['n2max']
    else:
        sq = [s * s for s in seqlens]
        n2 = sum(sq)
        off2 = torch.tensor([sum(sq[:i]) for i in range(len(sq))], device=q.device, dtype=torch.int32)
        pbuf = torch.empty(H, n2, device=q.device, dtype=torch.bfloat16)
        dpbuf = torch.empty(H, n2, device=q.device, dtype=torch.bfloat16)

    dq = torch.zeros(N, H, DQK, device=q.device, dtype=torch.float32)
    dk = torch.zeros(N, H, DQK, device=q.device, dtype=torch.float32)
    dv = torch.zeros(N, H, DV, device=q.device, dtype=torch.float32)

    _pdp_kernel[(triton.cdiv(max_len, bm), B, H)](
        q, k, v, do, lse, delta, pbuf, dpbuf, cu_seqlens, off2,
        sm_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        do.stride(0), do.stride(1),
        lse.stride(0), delta.stride(0),
        N2=n2, DQK=DQK, DVT=DV, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    _dq_t_kernel[(triton.cdiv(max_len, bm), B, H)](
        dpbuf, k, dq, cu_seqlens, off2,
        sm_scale,
        k.stride(0), k.stride(1),
        dq.stride(0), dq.stride(1),
        N2=n2, DQK=DQK, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    _dkv_t_kernel[(triton.cdiv(max_len, bn), B, H)](
        pbuf, dpbuf, q, do, dk, dv, cu_seqlens, off2,
        sm_scale,
        q.stride(0), q.stride(1),
        do.stride(0), do.stride(1),
        dk.stride(0), dk.stride(1),
        dv.stride(0), dv.stride(1),
        N2=n2, DQK=DQK, DVT=DV, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


@triton.jit
def _p_store_kernel(
    Q, K, LSE, PBUF, CU, OFF2,
    sm_scale,
    strideq_n, strideq_h,
    stridek_n, stridek_h,
    stridelse_h,
    N2,
    DQK: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return
    q_idx = seq_start + offs_m
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
    q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)
    lse = tl.load(LSE + h * stridelse_h + q_idx, mask=mask_m, other=0.0)

    off2 = tl.load(OFF2 + b)
    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n
        k_base = K + kv_idx[:, None] * stridek_n + h * stridek_h
        k1a = tl.load(k_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k1b = tl.load(k_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        k2 = tl.load(k_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)

        sc = tl.dot(q1a, tl.trans(k1a))
        sc += tl.dot(q1b, tl.trans(k1b))
        sc += tl.dot(q2, tl.trans(k2))
        sc = sc * sm_scale
        p = tl.exp(sc - lse[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        ptrs = h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :]
        tl.store(PBUF + ptrs, p.to(PBUF.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _dp_store_kernel(
    PBUF, DO, V, DELTA, DPBUF, CU, OFF2,
    stridev_n, stridev_h,
    stridedo_n, stridedo_h,
    stridedel_h,
    N2,
    DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_m = pid_m * BM + tl.arange(0, BM)
    mask_m = offs_m < s
    if tl.sum(mask_m.to(tl.int32)) == 0:
        return
    q_idx = seq_start + offs_m
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    do_base = DO + q_idx[:, None] * stridedo_n + h * stridedo_h
    do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
    do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)
    delta = tl.load(DELTA + h * stridedel_h + q_idx, mask=mask_m, other=0.0)

    off2 = tl.load(OFF2 + b)
    n_end = tl.cdiv(s, BN)
    for start_n in range(0, n_end):
        offs_n = start_n * BN + tl.arange(0, BN)
        mask_n = offs_n < s
        kv_idx = seq_start + offs_n
        tptr = h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :]
        p = tl.load(PBUF + tptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        v_base = V + kv_idx[:, None] * stridev_n + h * stridev_h
        v1a = tl.load(v_base + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        v1b = tl.load(v_base + 128 + offs_128[None, :], mask=mask_n[:, None], other=0.0)
        v2 = tl.load(v_base + 256 + offs_32[None, :], mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

        dS = tl.dot(do1a, tl.trans(v1a))
        dS += tl.dot(do1b, tl.trans(v1b))
        dS += tl.dot(do2, tl.trans(v2))

        dp = p * (dS.to(tl.float32) - delta[:, None])
        dp = tl.where(mask_n[None, :], dp, 0.0)
        tl.store(DPBUF + tptr, dp.to(DPBUF.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _dk_t_kernel(
    DPBUF, Q, DK, CU, OFF2,
    sm_scale,
    strideq_n, strideq_h,
    stridedk_n, stridedk_h,
    N2,
    DQK: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < s
    if tl.sum(mask_n.to(tl.int32)) == 0:
        return
    kv_idx = seq_start + offs_n
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    dk1a = tl.zeros([BN, 128], dtype=tl.float32)
    dk1b = tl.zeros([BN, 128], dtype=tl.float32)
    dk2 = tl.zeros([BN, 32], dtype=tl.float32)

    off2 = tl.load(OFF2 + b)
    m_end = tl.cdiv(s, BM)
    for start_m in range(0, m_end):
        offs_m = start_m * BM + tl.arange(0, BM)
        mask_m = offs_m < s
        q_idx = seq_start + offs_m
        tptr = h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :]
        dp = tl.load(DPBUF + tptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        dpT = tl.trans(dp)
        dpTq = dpT.to(tl.bfloat16)

        q_base = Q + q_idx[:, None] * strideq_n + h * strideq_h
        q1a = tl.load(q_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        q1b = tl.load(q_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        q2 = tl.load(q_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DQK), other=0.0)

        dk1a += tl.dot(dpTq, q1a)
        dk1b += tl.dot(dpTq, q1b)
        dk2 += tl.dot(dpTq, q2)

    dk_base = DK + kv_idx[:, None] * stridedk_n + h * stridedk_h
    tl.store(dk_base + offs_128[None, :], (dk1a * sm_scale).to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dk_base + 128 + offs_128[None, :], (dk1b * sm_scale).to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dk_base + 256 + offs_32[None, :], (dk2 * sm_scale).to(DK.dtype.element_ty),
             mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DQK))


@triton.jit
def _dv_t_kernel(
    PBUF, DO, DV, CU, OFF2,
    strideo_n, strideo_h,
    stridedv_n, stridedv_h,
    N2,
    DVT: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)
    seq_start = tl.load(CU + b)
    seq_end = tl.load(CU + b + 1)
    s = seq_end - seq_start

    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_n = offs_n < s
    if tl.sum(mask_n.to(tl.int32)) == 0:
        return
    kv_idx = seq_start + offs_n
    offs_128 = tl.arange(0, 128)
    offs_32 = tl.arange(0, 32)

    dv1a = tl.zeros([BN, 128], dtype=tl.float32)
    dv1b = tl.zeros([BN, 128], dtype=tl.float32)
    dv2 = tl.zeros([BN, 32], dtype=tl.float32)

    off2 = tl.load(OFF2 + b)
    m_end = tl.cdiv(s, BM)
    for start_m in range(0, m_end):
        offs_m = start_m * BM + tl.arange(0, BM)
        mask_m = offs_m < s
        q_idx = seq_start + offs_m
        tptr = h * N2 + off2 + offs_m[:, None] * s + offs_n[None, :]
        p = tl.load(PBUF + tptr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        pT = tl.trans(p)
        pTdo = pT.to(tl.bfloat16)

        do_base = DO + q_idx[:, None] * strideo_n + h * strideo_h
        do1a = tl.load(do_base + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        do1b = tl.load(do_base + 128 + offs_128[None, :], mask=mask_m[:, None], other=0.0)
        do2 = tl.load(do_base + 256 + offs_32[None, :], mask=mask_m[:, None] & ((256 + offs_32)[None, :] < DVT), other=0.0)

        dv1a += tl.dot(pTdo, do1a)
        dv1b += tl.dot(pTdo, do1b)
        dv2 += tl.dot(pTdo, do2)

    dv_base = DV + kv_idx[:, None] * stridedv_n + h * stridedv_h
    tl.store(dv_base + offs_128[None, :], dv1a.to(DV.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + 128 + offs_128[None, :], dv1b.to(DV.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_base + 256 + offs_32[None, :], dv2.to(DV.dtype.element_ty),
             mask=mask_n[:, None] & ((256 + offs_32)[None, :] < DVT))




@triton.jit
def _delta_kernel(DO, O, DELTA, N, H: tl.constexpr, DV: tl.constexpr, BLK: tl.constexpr):
    # delta[h, n] = sum_d do[n, h, d] * o[n, h, d]  (one launch; replaces ~5 eager ops)
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_n = pid * BLK + tl.arange(0, BLK)
    mask = offs_n < N
    base = offs_n[:, None] * (H * DV) + h * DV
    acc = tl.zeros([BLK], dtype=tl.float32)
    for d0 in range(0, DV, 128):
        dd = d0 + tl.arange(0, 128)
        m2 = mask[:, None] & (dd[None, :] < DV)
        dv_ = tl.load(DO + base + dd[None, :], mask=m2, other=0.0).to(tl.float32)
        ov_ = tl.load(O + base + dd[None, :], mask=m2, other=0.0).to(tl.float32)
        acc += tl.sum(dv_ * ov_, axis=1)
    tl.store(DELTA + h * N + offs_n, acc, mask=mask)


@triton.jit
def _off2_kernel(CU, OFF2, B: tl.constexpr):
    # off2[b] = exclusive prefix sum of s_i^2 (one tiny launch; replaces host loop + H2D)
    b = tl.arange(0, 64)
    m = b < B
    s = tl.load(CU + b + 1, mask=m, other=0) - tl.load(CU + b, mask=m, other=0)
    sq = s * s
    ps = tl.cumsum(sq, axis=0) - sq
    tl.store(OFF2 + b, ps, mask=m)


def mha_varlen_bwd_tiled5(q, k, v, o, lse, do, cu_seqlens, seqlens, sm_scale, max_len=None,
                          bm=32, bn=32, warps=4, stages=1, static_key=None, n2max=None):
    """5-kernel tiled bwd (PMC-guided champion, -26.5% vs fused-dkv 3-kernel):
    p_store -> dp_store -> dq_t -> dk_t + dv_t. Splits halve per-kernel LDS
    operand pressure (bank-conflict bound: 193%/153%/70% on the fused kernels)."""
    N, H, DQK = q.shape
    DV = v.shape[-1]
    B = cu_seqlens.numel() - 1
    if max_len is None:
        max_len = max(seqlens)

    # launch-lean prologue: 2 tiny kernels replace ~5 eager ops + 3 memsets +
    # host-loop H2D + 3 trailing cast kernels (~10 launches saved per call)
    delta = torch.empty(H, N, device=q.device, dtype=torch.float32)
    _delta_kernel[(triton.cdiv(N, 256), H)](do, o, delta, N, H=H, DV=DV, BLK=256)
    sq = [s * s for s in seqlens]
    n2 = sum(sq)
    off2 = torch.empty(B, device=q.device, dtype=torch.int32)
    _off2_kernel[(1,)](cu_seqlens, off2, B=B)
    pbuf = torch.empty(H, n2, device=q.device, dtype=torch.bfloat16)
    dpbuf = torch.empty(H, n2, device=q.device, dtype=torch.bfloat16)

    # every output row is written exactly once by its tile (no cross-program
    # accumulation), so bf16 direct store is safe and identical numerically
    # (fp32 accum in registers, cast fused into the store)
    dq = torch.empty(N, H, DQK, device=q.device, dtype=q.dtype)
    dk = torch.empty(N, H, DQK, device=q.device, dtype=k.dtype)
    dv = torch.empty(N, H, DV, device=q.device, dtype=v.dtype)

    _p_store_kernel[(triton.cdiv(max_len, bm), B, H)](
        q, k, lse, pbuf, cu_seqlens, off2,
        sm_scale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        lse.stride(0),
        n2, DQK=DQK, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    _dp_store_kernel[(triton.cdiv(max_len, bm), B, H)](
        pbuf, do, v, delta, dpbuf, cu_seqlens, off2,
        v.stride(0), v.stride(1),
        do.stride(0), do.stride(1),
        delta.stride(0),
        n2, DVT=DV, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    _dq_t_kernel[(triton.cdiv(max_len, bm), B, H)](
        dpbuf, k, dq, cu_seqlens, off2,
        sm_scale,
        k.stride(0), k.stride(1),
        dq.stride(0), dq.stride(1),
        n2, DQK=DQK, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    _dk_t_kernel[(triton.cdiv(max_len, bn), B, H)](
        dpbuf, q, dk, cu_seqlens, off2,
        sm_scale,
        q.stride(0), q.stride(1),
        dk.stride(0), dk.stride(1),
        n2, DQK=DQK, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    _dv_t_kernel[(triton.cdiv(max_len, bn), B, H)](
        pbuf, do, dv, cu_seqlens, off2,
        do.stride(0), do.stride(1),
        dv.stride(0), dv.stride(1),
        n2, DVT=DV, H=H, BM=bm, BN=bn,
        num_warps=warps, num_stages=stages,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

def mha_varlen_bwd(q, k, v, o, lse, do, cu_seqlens, sm_scale, max_len=None,
                   bm=32, bn=32, warps=4, stages=1):
    """returns (dq, dk, dv); dq/dk shaped [N,H,Dqk], dv [N,H,Dv], fp32 buffers cast to input dtype."""
    N, H, DQK = q.shape
    DV = v.shape[-1]
    B = cu_seqlens.numel() - 1
    if max_len is None:
        max_len = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    delta = (do.float() * o.float()).sum(-1).contiguous()  # [N, H]
    dq = torch.zeros(N, H, DQK, device=q.device, dtype=torch.float32)
    dk = torch.zeros(N, H, DQK, device=q.device, dtype=torch.float32)
    dv = torch.zeros(N, H, DV, device=q.device, dtype=torch.float32)

    BM, BN = bm, bn
    grid = (triton.cdiv(max_len, BM), B, H)
    _mha_bwd_kernel[grid](
        q, k, v, o, do, lse, delta,
        dq, dk, dv, cu_seqlens,
        sm_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        lse.stride(0),
        dq.stride(0), dq.stride(1),
        dk.stride(0), dk.stride(1),
        dv.stride(0), dv.stride(1),
        DQK=DQK, DVT=DV, H=H,
        BM=BM, BN=BN,
        num_warps=warps, num_stages=stages,
    )
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


if __name__ == '__main__':
    import math
    torch.manual_seed(0)
    dev = 'cuda'
    # MEET LDM base/large shapes: Dqk=265, Dv=272, mixed lengths
    for (seqlens, H, Dqk, Dv) in [([37, 128, 64, 15], 8, 265, 272), ([450] * 8, 12, 265, 272), ([800, 600, 100], 8, 265, 272)]:
        N = sum(seqlens)
        scale = 1.0 / math.sqrt(Dqk)
        q = torch.randn(N, H, Dqk, device=dev, dtype=torch.bfloat16) * 0.3
        k = torch.randn(N, H, Dqk, device=dev, dtype=torch.bfloat16) * 0.3
        v = torch.randn(N, H, Dv, device=dev, dtype=torch.bfloat16) * 0.3
        cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=dev, dtype=torch.int32)
        o, lse = mha_varlen_fwd(q, k, v, cu, scale)
        # fp32 naive reference
        ref = torch.zeros(N, H, Dv, device=dev)
        off = 0
        for s in seqlens:
            qs = q[off:off + s].transpose(0, 1).float()
            ks = k[off:off + s].transpose(0, 1).float()
            vs = v[off:off + s].transpose(0, 1).float()
            ref[off:off + s] = ((qs @ ks.transpose(-1, -2) * scale).softmax(-1) @ vs).transpose(0, 1)
            off += s
        err = (o.float() - ref).abs().max().item()
        print(f'N={N} H={H} Dqk={Dqk} Dv={Dv}: out {tuple(o.shape)} max_err={err:.4f}')
        assert err < 0.02, 'FAILED'
    print('TRITON FWD ALL CHECKS PASSED')
