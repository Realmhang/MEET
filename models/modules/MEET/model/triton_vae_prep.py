"""Fused VAE prep (no_grad only): one triton kernel replaces ~25 small aten ops
per attention call in the frozen VAE's InitLayer.

Builds, from normed H_q/H_k [N,H,D], coords [N,3], dist_scale [1,H,1]:
  q [N,H,D+16] = [H_q | -(scale * [x^2,1,-2x, y^2,1,-2y, z^2,1,-2z, 0*7])]
  k [N,H,D+16] = [H_k | +(scale * [1,x^2,x, 1,y^2,y, 1,z^2,z, 0*7])]
  v [1,N,H,8]  = [x, y, z, 0,0,0,0,0] broadcast over heads

Bit-exact vs the eager chain: all ops are elementwise (x*x for x**2, scalar
muls, copies); the bf16->fp32 promotion of the original torch.cat is
reproduced by loading HQ/HK and storing fp32.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _vae_prep_kern(HQ, HK, COORD, SCALE, Q, K, V,
                   N, D: tl.constexpr, H: tl.constexpr,
                   D2: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = rows < N
    cx = tl.load(COORD + rows * 3 + 0, mask=rmask, other=0.0).to(tl.float32)
    cy = tl.load(COORD + rows * 3 + 1, mask=rmask, other=0.0).to(tl.float32)
    cz = tl.load(COORD + rows * 3 + 2, mask=rmask, other=0.0).to(tl.float32)

    offs_d = tl.arange(0, D2)
    dmask = offs_d < D
    offs_16 = tl.arange(0, 16)
    # dist column groups: g = d//3 selects coord; within-group pattern
    g = offs_16 // 3
    w = offs_16 % 3
    c_g = tl.where(g == 0, cx[:, None], tl.where(g == 1, cy[:, None], cz[:, None]))
    zero16 = (offs_16 >= 9)[None, :]
    # q pattern: [sq, 1, -2c] per group; k pattern: [1, sq, c]
    dq = tl.where(zero16, 0.0,
                  tl.where(w[None, :] == 0, c_g * c_g,
                           tl.where(w[None, :] == 1, 1.0, -2.0 * c_g)))
    dk = tl.where(zero16, 0.0,
                  tl.where(w[None, :] == 0, 1.0,
                           tl.where(w[None, :] == 1, c_g * c_g, c_g)))

    for h in tl.static_range(H):
        sc = tl.load(SCALE + h)
        base = rows[:, None] * H * (D + 16) + h * (D + 16)
        hsrc = rows[:, None] * H * D + h * D
        # H part: copy + promote to fp32
        hv = tl.load(HQ + hsrc + offs_d[None, :], mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        tl.store(Q + base + offs_d[None, :], hv, mask=rmask[:, None] & dmask[None, :])
        tl.store(Q + base + D + offs_16[None, :], -(sc * dq),
                 mask=rmask[:, None] & (offs_16 < 16)[None, :])
        hv2 = tl.load(HK + hsrc + offs_d[None, :], mask=rmask[:, None] & dmask[None, :], other=0.0).to(tl.float32)
        tl.store(K + base + offs_d[None, :], hv2, mask=rmask[:, None] & dmask[None, :])
        tl.store(K + base + D + offs_16[None, :], sc * dk,
                 mask=rmask[:, None] & (offs_16 < 16)[None, :])

    # v [N, H, 8]: first 3 cols = coords, rest 0
    offs_8 = tl.arange(0, 8)
    vmask = rmask[:, None] & (offs_8 < 8)[None, :]
    for h in tl.static_range(H):
        vbase = rows[:, None] * H * 8 + h * 8
        vval = tl.where(offs_8[None, :] == 0, cx[:, None],
                        tl.where(offs_8[None, :] == 1, cy[:, None],
                                 tl.where(offs_8[None, :] == 2, cz[:, None], 0.0)))
        tl.store(V + vbase + offs_8[None, :], vval, mask=vmask)


def fused_vae_prep(H_q, H_k, coord, dist_scale, n_heads):
    """Returns (q [N,H,D+16], k [N,H,D+16], v [1,N,H,8]) — all fp32, matching
    the original eager chain's dtypes (cat promotes bf16 H to fp32)."""
    N, H, D = H_q.shape
    assert H == n_heads
    dev = H_q.device
    q = torch.empty(N, H, D + 16, device=dev, dtype=torch.float32)
    k = torch.empty(N, H, D + 16, device=dev, dtype=torch.float32)
    v = torch.empty(N, H, 8, device=dev, dtype=torch.float32)
    scale = dist_scale.reshape(-1).contiguous()
    D2 = triton.next_power_of_2(D)
    BLOCK = 64
    _vae_prep_kern[(triton.cdiv(N, BLOCK),)](
        H_q.contiguous(), H_k.contiguous(), coord.contiguous(), scale,
        q, k, v, N, D=D, H=H, D2=D2, BLOCK=BLOCK)
    return q, k, v.unsqueeze(0)
