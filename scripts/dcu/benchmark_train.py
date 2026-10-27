#!/usr/bin/env python3
"""Run the normal trainer with full-step timing, or summarize its rank JSON files."""
import argparse
import functools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def summarize(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--start', type=int, default=80)
    parser.add_argument('--end', type=int, default=145)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error('--start must not exceed --end')
    output = []
    for directory in args.runs:
        paths = sorted(directory.glob('rank*.json'))
        if not paths:
            raise ValueError(f'No rank reports in {directory}')
        reports = [json.loads(path.read_text()) for path in paths]
        world = reports[0]['world_size']
        if {r['rank'] for r in reports} != set(range(world)) or len(reports) != world:
            raise ValueError(f'Missing or duplicate ranks in {directory}')
        batch = reports[0]['global_batch']
        ranks = []
        for report in reports:
            if not report['completed'] or report['global_batch'] != batch:
                raise ValueError(f'Failed or inconsistent run: {directory}')
            if report['fused_opt']:
                flags = report['found_inf']
                if len(flags) != report['optimizer_calls'] or any(v != 0 for v in flags):
                    raise ValueError(f'Missing found_inf samples or skipped updates: {directory}')
                if report['optimizer_calls'] != report['global_steps']:
                    raise ValueError(f'Not every step updated parameters: {directory}')
            rows = [r for r in report['rows'] if args.start <= r['step'] <= args.end]
            if sorted(r['step'] for r in rows) != list(range(args.start, args.end + 1)):
                raise ValueError(f'Incomplete steady window: {directory}')
            if not all(math.isfinite(r['wall_ms']) and r['wall_ms'] > 0 for r in rows):
                raise ValueError(f'Invalid timing: {directory}')
            wall = statistics.median(r['wall_ms'] for r in rows)
            event = statistics.median(r['event_ms'] for r in rows)
            if not 0.95 < event / wall < 1.05:
                raise ValueError(f'Event/wall mismatch: {directory}')
            mean = statistics.mean(r['wall_ms'] for r in rows)
            ranks.append(dict(rank=report['rank'], median_ms=wall, mean_ms=mean,
                              typical_samples_s=batch * 1000 / wall,
                              window_samples_s=batch * 1000 / mean,
                              peak_allocated_gib=report['peak_allocated_bytes'] / 2**30,
                              optimizer_calls=report['optimizer_calls']))
        if max(r['median_ms'] for r in ranks) - min(r['median_ms'] for r in ranks) > 1.0:
            raise ValueError(f'Rank timing differs by >1 ms: {directory}')
        output.append(dict(run=str(directory), global_batch=batch, window=[args.start, args.end], ranks=ranks))
    print(json.dumps(output, indent=2))


def run(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    options, training_args = parser.parse_known_args(argv)
    source, output = options.source_root.resolve(), options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', '0'))
    report_path = output / f'rank{rank}.json'
    if report_path.exists():
        raise FileExistsError(f'Refusing to overwrite {report_path}; use a new output directory')
    sys.path.insert(0, str(source))
    os.chdir(source)
    import torch
    import torch.profiler as tp
    import train as training
    import trainer.abs_trainer as trainer_module

    original_create = training.create_trainer

    def scoped(name, function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with tp.record_function(name):
                return function(*args, **kwargs)
        return call

    def create(config, model, train_loader, valid_loader):
        trainer = original_create(config, model, train_loader, valid_loader)
        batch = config['dataloader']['train']['batch_size']
        world = int(os.environ.get('WORLD_SIZE', '1'))
        if batch % world or train_loader.batch_size != batch // world:
            raise ValueError('Global batch must be divisible by world size')
        measurements, flags = [], []
        state = dict(current=None, scope=None, optimizer_calls=0)

        def close_step():
            if state['current'] is None:
                return
            step, wall, start, end = state['current']
            end.record()
            elapsed = (time.perf_counter() - wall) * 1000
            state['scope'].__exit__(None, None, None)
            measurements.append((step, elapsed, start, end))
            state['current'] = state['scope'] = None

        class Loader:
            def __getattr__(self, name):
                return getattr(train_loader, name)

            def __len__(self):
                return len(train_loader)

            def __iter__(self):
                iterator = iter(train_loader)
                while True:
                    close_step()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    step = int(trainer.global_step)
                    scope = tp.record_function(f'MEET_STEP_{step}')
                    scope.__enter__()
                    wall = time.perf_counter()
                    start.record()
                    state.update(current=(step, wall, start, end), scope=scope)
                    try:
                        with tp.record_function('MEET_DATA_WAIT'):
                            value = next(iterator)
                    except StopIteration:
                        close_step()
                        return
                    yield value

        trainer.train_loader = Loader()
        trainer.train_step = scoped('MEET_FORWARD', trainer.train_step)
        trainer.to_device = scoped('MEET_TO_DEVICE', trainer.to_device)
        trainer_module.safe_backward = scoped('MEET_BACKWARD', trainer_module.safe_backward)
        model.autoencoder.encode = scoped('MEET_VAE_ENCODE', model.autoencoder.encode)
        model.generator.forward = scoped('MEET_LDM_FORWARD', model.generator.forward)
        optimizer_step = trainer.optimizer.step

        @functools.wraps(optimizer_step)
        def audited_step(*args, **kwargs):
            state['optimizer_calls'] += 1
            flag = getattr(trainer.optimizer, 'found_inf', None)
            if isinstance(flag, torch.Tensor):
                flags.append(flag)
            with tp.record_function('MEET_ADAMW'):
                return optimizer_step(*args, **kwargs)

        trainer.optimizer.step = audited_step
        original_train = trainer.train

        def measured_train(*args, **kwargs):
            completed = False
            torch.cuda.reset_peak_memory_stats()
            try:
                result = original_train(*args, **kwargs)
                completed = True
                return result
            finally:
                close_step()
                torch.cuda.synchronize()
                report = dict(completed=completed, rank=rank, world_size=world, global_batch=batch,
                              source=str(source), torch=torch.__version__, hip=torch.version.hip,
                              device=torch.cuda.get_device_name(), global_steps=int(trainer.global_step),
                              fused_opt=os.environ.get('MEET_FUSED_OPT', '0') == '1',
                              optimizer_calls=state['optimizer_calls'],
                              found_inf=torch.stack(flags).float().cpu().flatten().tolist() if flags else [],
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                              rows=[dict(step=s, wall_ms=w, event_ms=a.elapsed_time(b))
                                    for s, w, a, b in measurements])
                report_path.write_text(json.dumps(report, indent=2))
        trainer.train = measured_train
        return trainer

    training.create_trainer = create
    sys.argv = [str(source / 'train.py')] + training_args
    args, overrides = training.parse()
    training.setup_seed(args.seed)
    training.main(args, overrides)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summarize':
        summarize(sys.argv[2:])
    else:
        run(sys.argv[1:])
