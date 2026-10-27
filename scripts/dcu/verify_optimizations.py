#!/usr/bin/env python3
"""Check host semantics and BC128 against a frozen kernel and FP32 autograd."""
import argparse
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load_kernel(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_checks(torch):
    source = ROOT / 'models/LatentGenHX/model.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_prepare_gen_batch')
    code = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    env = dict(torch=torch, VAEModelInput=object)
    exec(compile(code, str(source), 'exec'), env)
    prepare = env['_prepare_gen_batch']
    vae = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 4))
    records = []

    class ReachedEncoder(Exception):
        pass

    def encode(*args, **kwargs):
        records.append([m.training for m in vae.modules()])
        raise ReachedEncoder()

    vae.encode = encode
    obj = SimpleNamespace(autoencoder=vae, latent_deterministic=False, training=True)
    inputs = SimpleNamespace(**{k: None for k in ('X', 'S', 'A', 'bonds', 'chain_ids', 'ligand_mask', 'block_lengths', 'lengths')})
    for mode in (True, None, False, True, None):
        if mode is not None:
            vae.train(mode)
        try:
            prepare(obj, inputs)
        except ReachedEncoder:
            pass
    assert len(records) == 5 and not any(flag for row in records for flag in row)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
    model.register_parameter('shared', model[0].weight)
    cached = tuple(model.parameters())
    for iteration in range(4):
        for index, param in enumerate(cached):
            param.requires_grad_(index != iteration)
            param.grad = torch.ones_like(param) if (index + iteration) % 2 else None
        old = [id(p) for p in model.parameters() if p.grad is not None and p.requires_grad]
        new = [id(p) for p in cached if p.grad is not None and p.requires_grad]
        assert old == new
    return dict(eval_modes=True, parameter_identity_order_and_filter=True)


def attention_checks(torch, baseline):
    current_path = ROOT / 'models/modules/MEET/model/triton_mha.py'
    if baseline.resolve() == current_path.resolve():
        raise ValueError('Baseline must be the pre-BC128 source, not the current file')
    old = load_kernel(baseline, 'dcu_frozen_kernel')
    new = load_kernel(current_path, 'dcu_current_kernel')
    if hasattr(old, '_grad_channel_kernel'):
        raise ValueError('Baseline already contains BC128; extract the documented pre-change version')
    torch.manual_seed(17)
    scale = 1 / math.sqrt(128)
    cases = [([1, 15, 31, 32, 33, 63, 65, 129], 272),
             ([56, 151, 106, 168, 88, 166, 95, 67], 272),
             ([1025, 257, 33, 1], 272), ([31, 65, 129], 265)]
    reports = []
    for lengths, dimension in cases:
        n = sum(lengths)
        q, k = [(torch.randn(n, 8, dimension, device='cuda') * 0.3).bfloat16() for _ in range(2)]
        v = (torch.randn(n, 8, 272, device='cuda') * 0.3).bfloat16()
        do = torch.randn_like(v)
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), device='cuda', dtype=torch.int32)
        out, lse = new.mha_varlen_fwd(q, k, v, cu, scale, max(lengths), bm=64, bn=16)
        old_out, _ = old.mha_varlen_fwd(q, k, v, cu, scale, max(lengths), bm=64, bn=16)
        assert torch.equal(out, old_out)
        arguments = (q, k, v, out, lse, do, cu, lengths, scale, max(lengths))
        actual = new.mha_varlen_bwd_tiled5(*arguments)
        expected = old.mha_varlen_bwd_tiled5(*arguments)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected)), 'BC128 differs from frozen backward'
        qr, kr, vr = [x.float().requires_grad_() for x in (q, k, v)]
        outputs = []
        offset = 0
        for length in lengths:
            a, b, c = [x[offset:offset + length].transpose(0, 1) for x in (qr, kr, vr)]
            outputs.append(((a @ b.transpose(-1, -2) * scale).softmax(-1) @ c).transpose(0, 1))
            offset += length
        reference = torch.cat(outputs)
        reference.backward(do.float())
        errors = []
        for a, b in zip((out,) + actual, (reference, qr.grad, kr.grad, vr.grad)):
            assert torch.isfinite(a).all().item()
            relative_l2 = ((a.float() - b).norm() / b.norm().clamp_min(1e-8)).item()
            relative_max = ((a.float() - b).abs().max() / b.abs().max().clamp_min(1e-8)).item()
            assert relative_l2 < 0.015 and relative_max < 0.02, (relative_l2, relative_max)
            errors.append(dict(relative_l2=relative_l2, relative_max=relative_max))
        reports.append(dict(lengths=lengths, qk_dimension=dimension, v_dimension=272,
                            baseline_bitwise=True, fp32_errors=errors))
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-kernel', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if not args.cpu_only and args.baseline_kernel is None:
        parser.error('--baseline-kernel is required unless --cpu-only is set')
    os.environ.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', None)
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    report = dict(host=host_checks(torch))
    if not args.cpu_only:
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError('A compatible DCU/DTK runtime is required; no GPU checks were run')
        report['attention'] = attention_checks(torch, args.baseline_kernel)
    report['scope'] = 'Host invariants and attention regression, not full-model convergence or task-quality certification'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
