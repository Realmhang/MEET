"""Manual full-step CUDA graph capture for MEET LatentGeneratorHX (DCU).

Captures [generator forward + loss + scaler-scaled backward] per (N', L') bucket
with static input buffers. Per step: eager VAE encode + padding -> copy inputs
into static buffers -> update cu/off2 content -> replay.

Env: MEET_GRAPH=1 (master switch), MEET_BUCKET_PAD=128 (padding bucket),
MEET_MAXLEN_BOUND=512 (per-peptide latent length bound, asserted).
"""
import os
import torch


def print_log_safe(msg):
    try:
        print(f'[graph_step] {msg}', flush=True)
    except Exception:
        pass

_GRAPH_WARMUP = 3


class _DummyScaler:
    """scaler stand-in when AMP is disabled."""
    def scale(self, loss):
        return loss


class _GraphedGen:
    def __init__(self, module, scaler=None, autocast_dtype=None, ddp=None):
        self.module = module            # LatentGeneratorHX (inner, unwrapped)
        self.scaler = scaler if scaler is not None else _DummyScaler()
        self.autocast_dtype = autocast_dtype
        self.ddp = ddp                  # DDP wrapper (for reducer prep), optional
        self.graphs = {}                # (N', L') -> dict(graph, statics, loss)
        self.shared_pool = None

    # ---------------- per-step entry ----------------
    def step(self, gen_batch):
        Np = gen_batch['H'].shape[0]
        key = (Np, gen_batch['ligand_idx'].shape[0])
        dev = gen_batch['H'].device
        seqlens = gen_batch['seqlens']
        bound = int(os.environ.get('MEET_MAXLEN_BOUND', '512'))
        assert max(seqlens) <= bound, f'peptide latent len {max(seqlens)} > bound {bound}'

        from models.modules.MEET.model import xformers_compat as xc
        from models.modules.MEET import meet as _meet
        from models.modules.MEET.model import triton_mha as tm

        entry = self.graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(gen_batch, key, seqlens, bound)
            except RuntimeError as e:
                # DCU caching allocator refuses pool reuse when a prior graph's
                # blocks are still referenced (use_count>0 assert) -> fall back
                # to eager for this bucket instead of killing the run.
                print_log_safe(f'graph capture failed for {key}: {e}; eager fallback')
                entry = {'eager': True}
                self.graphs[key] = entry
        if entry.get('eager'):
            # eager path (identical math, no graph)
            if self.autocast_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                    loss, _ = self.module.generator(gen_batch)
            else:
                loss, _ = self.module.generator(gen_batch)
            return loss
        g, statics = entry['graph'], entry['statics']

        # per-step eager prep (all syncs and RNG outside the graph)
        cu = torch.zeros(len(seqlens) + 1, device=dev, dtype=torch.int32)
        torch.cumsum(torch.tensor(seqlens, device=dev, dtype=torch.int32), 0, out=cu[1:])
        sq = [s * s for s in seqlens]
        off2_new = torch.tensor([sum(sq[:i]) for i in range(len(sq))],
                                device=dev, dtype=torch.int32)

        # pre-sample t / eps eagerly (ROCm RNG is not capture-safe)
        Bs = statics['_static_t'].shape[0]
        statics['_static_t'].copy_(torch.rand(Bs, device=dev, dtype=torch.float32))
        statics['_static_eps_H'].copy_(torch.randn(
            statics['_static_eps_H'].shape, device=dev, dtype=torch.float32))
        statics['_static_eps_X'].copy_(torch.randn(
            statics['_static_eps_X'].shape, device=dev, dtype=torch.float32))

        # seed caches so graph-baked buffers resolve on the python paths
        xc._CU_CACHE[(tuple(seqlens), str(dev))] = (statics['cu'], bound)
        _meet._SEQLEN_STASH[statics['batch_ids'].data_ptr()] = seqlens

        for k in ('H', 'X', 'position_ids', 'batch_ids', 'generate_mask', 'ligand_idx'):
            statics[k].copy_(gen_batch[k])
        statics['cu'].copy_(cu)
        bufs = tm._STATIC_BWD_BUFS.get(('bwd', Np))
        if bufs is not None:
            bufs['off2'][:off2_new.numel()].copy_(off2_new)

        g.replay()
        return entry['loss']

    # ---------------- one-time capture per bucket ----------------
    def _capture(self, gen_batch, key, seqlens, bound):
        Np = key[0]
        dev = gen_batch['H'].device

        from models.modules.MEET.model import xformers_compat as xc
        from models.modules.MEET import meet as _meet
        from models.modules.MEET.model import triton_mha as tm

        statics = {k: gen_batch[k].clone() for k in
                   ('H', 'X', 'position_ids', 'batch_ids', 'generate_mask', 'ligand_idx')}
        statics['cu'] = torch.zeros(len(seqlens) + 1, device=dev, dtype=torch.int32)
        torch.cumsum(torch.tensor(seqlens, device=dev, dtype=torch.int32), 0,
                     out=statics['cu'][1:])
        # static RNG buffers (sampled eagerly per step; ROCm RNG not capture-safe)
        Lp = key[1]
        statics['_static_t'] = torch.zeros(len(seqlens), device=dev, dtype=torch.float32)
        statics['_static_eps_H'] = torch.zeros(Lp, statics['H'].shape[1],
                                               device=dev, dtype=torch.float32)
        statics['_static_eps_X'] = torch.zeros(Lp, *statics['X'].shape[1:],
                                               device=dev, dtype=torch.float32)
        rsm = torch.ones(len(seqlens), device=dev, dtype=statics['H'].dtype)
        rsm[-1] = 0.0
        statics['real_sample_mask'] = rsm

        # seed shim caches + stash to the STATIC buffers before any run
        xc._CU_CACHE[(tuple(seqlens), str(dev))] = (statics['cu'], bound)
        _meet._SEQLEN_STASH[statics['batch_ids'].data_ptr()] = seqlens

        # pre-register bwd static buffers with correct off2 content
        n2max = Np * bound
        H = 8
        sq = [s * s for s in seqlens]
        tm._STATIC_BWD_BUFS[('bwd', Np)] = {
            'off2': torch.tensor([sum(sq[:i]) for i in range(len(sq))],
                                 device=dev, dtype=torch.int32),
            'pbuf': torch.empty(H, n2max, device=dev, dtype=torch.bfloat16),
            'dpbuf': torch.empty(H, n2max, device=dev, dtype=torch.bfloat16),
            'n2max': n2max,
        }

        def _run():
            # FWD-ONLY capture: the platform cannot capture backward
            # (hipErrorInvalidHandle in autograd engine). The static loss's
            # grad_fn (recorded during capture with autograd on) is reused:
            # each step the trainer replays, then runs EAGER backward through
            # it (retain_graph=True) — DDP hooks/allreduce fire normally.
            if self.autocast_dtype is not None:
                with torch.autocast(device_type='cuda', dtype=self.autocast_dtype):
                    loss, _ = self.module.generator(statics)
            else:
                loss, _ = self.module.generator(statics)
            return loss

        # warmup 1: one iteration on the DEFAULT stream first (NCCL comm + DDP
        # reducer initialize on the capture stream's handle domain; side-stream-only
        # warmup leads to hipErrorInvalidHandle during backward capture)
        self._zero_grads()
        _run()
        torch.cuda.synchronize()

        # warmup 2: side stream (compiles all kernels; avoids default-pool burn-in)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(_GRAPH_WARMUP):
                self._zero_grads()
                _run()
        torch.cuda.current_stream().wait_stream(s)
        self._zero_grads()
        torch.cuda.synchronize()

        # rocBLAS/cuBLAS workspaces bind to the stream they were first used on;
        # stale handles from warmup cause hipErrorInvalidHandle during capture.
        try:
            torch._C._cuda_clearCublasWorkspaces()
        except Exception:
            pass

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):  # own pool per bucket (shared pools assert on DCU)
            loss = _run()

        self._zero_grads()
        torch.cuda.synchronize()
        return {'graph': g, 'statics': statics, 'loss': loss}

    def _zero_grads(self):
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.zero_()
            else:
                p.grad = torch.zeros_like(p)
