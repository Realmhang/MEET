#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import yaml
from copy import deepcopy
from tqdm import tqdm

import atexit
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb # for users preferring wandb over tensorboard
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None

from utils.oom_decorator import OOMReturn, safe_backward
from utils.logger import print_log

########### Import your packages below ##########



_FUSED_OPT = __import__("os").environ.get("MEET_FUSED_OPT", "0") == "1"
_GRAPH = __import__("os").environ.get("MEET_GRAPH", "0") == "1"

class TrainConfig:
    def __init__(self, save_dir, max_epoch=None, max_step=None, warmup=0,
                 metric_min_better=True, patience=3,
                 grad_clip=None, save_topk=-1,  # -1 for save all
                 grad_interval=1,  # parameter update interval
                 val_freq=1,       # frequence for validation
                 val_freq_mode='epoch',  # epoch / step
                 logger='tensorboard',          # tensorboard / wandb / swanlab
                 find_unused_parameters=True,   # for DDP training
                 enable_mixed_precision=False,  # use mixed precision (FP16 or BF16, conditioned on the GPU)
                 continue_version=None,         # the version to continue (None for starting from ground up)
                 **kwargs):
        self.save_dir = save_dir
        self.max_epoch = max_epoch
        self.max_step = max_step
        assert not ((self.max_epoch is None) and (self.max_step is None)), 'At least one of max_epoch and max_step should be specified.'
        self.warmup = warmup
        self.metric_min_better = metric_min_better
        self.patience = patience if patience > 0 else max_epoch
        self.grad_clip = grad_clip
        self.save_topk = save_topk
        self.grad_interval = grad_interval
        self.val_freq = val_freq
        self.val_freq_mode = val_freq_mode
        assert self.val_freq_mode in ['epoch', 'step'], 'val_freq_mode must be epoch or step'
        self.logger = logger
        self.find_unused_parameters = find_unused_parameters
        self.enable_mixed_precision = enable_mixed_precision
        self.continue_version = continue_version
        self.__dict__.update(kwargs)

    def add_parameter(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


def move_optimizer_to_device(optimizer, device):
    state_dict = optimizer.state_dict()
    state_dict = Trainer.to_device(state_dict, device)
    optimizer.load_state_dict(state_dict)
    return optimizer


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict):
        self.model = model
        self.config = TrainConfig(**config)
        self.save_config = save_config
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.version = self._get_version() if self.config.continue_version is None else self.config.continue_version
        self.config.save_dir = os.path.join(self.config.save_dir, f'version_{self.version}')
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}

        # training process recording
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []  # smaller index means better ckpt
        self.patience = self.config.patience

        # for mixed precision training
        self.half_precision_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler() if self.config.enable_mixed_precision else None

        if self.config.continue_version is not None: self._load_states()

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data
    
    
    def _load_states(self):
        '''
            Continue training on the given version number.
            If saved states are not detected, train from ground up.
        '''
        # find the last checkpoint
        found, max_epoch, result_file = False, -1, None
        if os.path.exists(self.model_dir):
            for f in os.listdir(self.model_dir):
                match = re.search(r'epoch(\d+)_step(\d+).ckpt$', f)
                if match:
                    epoch = int(match.group(1))
                    if epoch > max_epoch:
                        found = True
                        max_epoch, result_file = epoch, f
        if not found:
            print_log(f'States not found at {self.model_dir}. Train from ground up.', level='WARN')
            return
        result_file = os.path.join(self.model_dir, result_file)
        print_log(f'Loading states from {result_file}')
        # load checkpoint
        model = torch.load(result_file, weights_only=False, map_location='cpu')
        self.model.load_state_dict(model.state_dict())
        # attributes
        states = torch.load(result_file + '.states', weights_only=False, map_location='cpu')
        attrs = states['attrs']
        self.global_step = attrs['global_step']
        self.valid_global_step = attrs['valid_global_step']
        self.epoch = attrs['epoch'] + 1 # start of the next epoch
        self.last_valid_metric = attrs['last_valid_metric']
        self.topk_ckpt_map = attrs['topk_ckpt_map']
        self.patience = attrs['patience']
        # optimizer
        optimizer_states = states['optimizer_state']
        self.optimizer.load_state_dict(optimizer_states)
        # scheduler
        scheduler_states = states['scheduler_state']
        if (scheduler_states is not None) and (self.scheduler is not None):
            self.scheduler.load_state_dict(scheduler_states)
    
    def _save_states(self):

        # model weights
        save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
        module_to_save = self.model.module if self.local_rank == 0 else self.model
        torch.save(module_to_save, save_path)

        # trainer attributes and states for optimizer and scheduler
        attrs = {
            'global_step': self.global_step,
            'valid_global_step': self.valid_global_step,
            'epoch': self.epoch,
            'last_valid_metric': self.last_valid_metric,
            'topk_ckpt_map': self.topk_ckpt_map,
            'patience': self.patience
        }
        states_save_path = save_path + '.states'
        torch.save({
            'attrs': attrs,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': None if self.scheduler is None else self.scheduler.state_dict()
        }, states_save_path)

        return save_path, states_save_path

    def _is_main_proc(self):
        if 'RANK' in os.environ:
            return int(os.environ['RANK']) == 0
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def is_oom_return(self, value):
        return isinstance(value, OOMReturn)

    def _train_epoch_begin(self, device):
        return
    
    def _train_step_begin(self, device):
        return

    def _train_epoch(self, device):
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        self._train_epoch_begin(device)
        _epoch_params = tuple(self.model.parameters()) if _FUSED_OPT else ()
        t_iter = tqdm(self.train_loader, ascii=True) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            _graphed = False
            _inner = getattr(self.model, 'module', self.model)
            if _GRAPH and hasattr(_inner, '_meet_graph_step') and self.config.grad_interval == 1:
                # fwd-only graph capture: replay returns a static loss; backward
                # runs eagerly below (platform cannot capture backward)
                loss = _inner._meet_graph_step(
                    batch,
                    scaler=None if _FUSED_OPT else (self.scaler if self.config.enable_mixed_precision else None),
                    autocast_dtype=self.half_precision_type if self.config.enable_mixed_precision else None,
                    ddp=None)
                _graphed = True
            elif self.config.enable_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=self.half_precision_type):
                    loss = self.train_step(batch, self.global_step)
            else:
                loss = self.train_step(batch, self.global_step)

            skip_step = torch.tensor([0], dtype=torch.int, device=device) # abnormal behavior detection
            
            if self.is_oom_return(loss):
                print_log(f'Out of memory, local rank {self.local_rank}', level='WARN')
                # loss = loss.fake_loss
                skip_step.fill_(1)
            elif torch.isnan(loss):
                print_log(f'Loss is nan, local_rank {self.local_rank}', level='WARN')
                skip_step.fill_(1)
                # loss = torch.nan_to_num(loss, nan=0.0)
                # loss = sum([p.norm() for p in self.model.parameters() if p.dtype == torch.float]) * 0.0
            elif torch.isinf(loss):
                print_log(f'Loss is inf, local_rank {self.local_rank}', level='WARN')
                skip_step.fill_(1)
                # loss = torch.nan_to_num(loss, posinf=0.0, neginf=0.0)
            if self.config.grad_interval > 1:
                loss = loss / self.config.grad_interval

            if self.global_step % self.config.grad_interval == 0:
                self.optimizer.zero_grad()
                self._train_step_begin(device)
            
            if dist.is_initialized(): dist.all_reduce(skip_step, op=dist.ReduceOp.MAX) # collect flag

            if not torch.any(skip_step):
            # if skip_step.item() == 0:
                if self.config.enable_mixed_precision and not _FUSED_OPT and not _graphed:
                    backward_ok = safe_backward(self.scaler.scale(loss), self.model)
                else:
                    if _graphed:
                        backward_ok = True
                        try:
                            loss.backward(retain_graph=True)
                        except Exception:
                            backward_ok = False
                    else:
                        backward_ok = safe_backward(loss, self.model)
                if not backward_ok:
                    print_log(f'Backward out of memory, skip', level='WARN')
                    loss = loss.detach() # manually delete the computing graph
                    skip_step.fill_(1)              
            
                if dist.is_initialized(): dist.all_reduce(skip_step, op=dist.ReduceOp.MAX) # collect flag
            else:
                if isinstance(loss, torch.Tensor): loss = loss.detach()   # forward loss has some problems, manually delete the computing graph

            if not torch.any(skip_step):
            # if skip_step.item() == 0:
                if (self.global_step + 1) % self.config.grad_interval == 0:
                    if _FUSED_OPT:
                        # fused clip+step: one foreach_norm pass on raw grads; the
                        # clip factor is folded into the fused adamw kernel via its
                        # grad_scale hook (g_eff = grad * coeff) — skips the separate
                        # unscale pass AND the clip mul pass (~2GB traffic/step).
                        # bf16 autocast keeps grads fp32, so loss scaling is unneeded;
                        # non-finite norm -> found_inf -> kernel skips the update.
                        _ps = [p for p in _epoch_params if p.grad is not None and p.requires_grad]
                        _gs = [p.grad for p in _ps]
                        _norms = torch._foreach_norm(_gs)
                        ori_grad_norm = torch.linalg.vector_norm(torch.stack(_norms))
                        if self.config.grad_clip is not None:
                            _coeff = torch.clamp(self.config.grad_clip / (ori_grad_norm + 1e-6), max=1.0)
                        else:
                            _coeff = torch.ones_like(ori_grad_norm)
                        self.optimizer.grad_scale = 1.0 / _coeff
                        self.optimizer.found_inf = (~torch.isfinite(ori_grad_norm)).float()
                        self.optimizer.step()
                        del self.optimizer.grad_scale, self.optimizer.found_inf
                        # recording gradients
                        self.log('Grad Norm', ori_grad_norm.detach().cpu(), self.global_step)
                    elif self.config.grad_clip is not None:
                        if self.config.enable_mixed_precision: self.scaler.unscale_(self.optimizer)
                        ori_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        # recording gradients
                        self.log('Grad Norm', ori_grad_norm.cpu(), self.global_step)
                        if self.config.enable_mixed_precision:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else: self.optimizer.step()
                    else:
                        ori_grad_norm = 0
                        for p in self.model.parameters():
                            if p.grad is not None and p.requires_grad:
                                ori_grad_norm += p.grad.detach().data.norm(2) ** 2
                        ori_grad_norm = ori_grad_norm ** 0.5
                        # recording gradients
                        self.log('Grad Norm', ori_grad_norm.cpu(), self.global_step)
                        if self.config.enable_mixed_precision:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else: self.optimizer.step()
                    self._train_step_end(device)
            else:
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()  # delete gradient memory

            if hasattr(t_iter, 'set_postfix'):
                if self.is_oom_return(loss): t_iter.set_postfix(loss=0, version=self.version)
                else: t_iter.set_postfix(loss=loss.item(), version=self.version)
            self.global_step += 1
            # validate if frequency is defined by step
            if (self.config.val_freq_mode == 'step') and (self.global_step % self.config.val_freq == 0):
                print_log(f'validating ...') if self._is_main_proc() else 1
                self._valid_epoch(device)

            if self.sched_freq == 'step':
                if self.global_step % self.config.grad_interval == 0:
                    self.scheduler.step()
            
            # stop criterion might be steps
            if self._check_train_finished(): break

        if self.sched_freq == 'epoch':
            self.scheduler.step()
        self._train_epoch_end(device)
    
    def _train_epoch_end(self, device):
        return
    
    def _train_step_end(self, device):
        return

    def _aggregate_val_metric(self, metric_arr):
        return np.mean(metric_arr)

    def _valid_epoch_begin(self, device):
        return

    def _valid_epoch(self, device):
        metric_arr = []
        self.model.eval()
        self._valid_epoch_begin(device)
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader, ascii=True) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                # metric = self.valid_step(batch, self.valid_global_step)

                if self.config.enable_mixed_precision:
                    with torch.autocast(device_type='cuda', dtype=self.half_precision_type):
                        metric = self.valid_step(batch, self.valid_global_step)
                else:
                    metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        
        # judge
        valid_metric = self._aggregate_val_metric(metric_arr)
        if self._is_main_proc():
            save_path, _ = self._save_states()
            self._maintain_topk_checkpoint(valid_metric, save_path)
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
        else:
            self.patience -= 1
        if self.sched_freq == 'val_epoch':
            self.scheduler.step(valid_metric)
        self.last_valid_metric = valid_metric
        # write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            if self._is_main_proc():
                print_log(f'{name}: {value}')
            # writer is None => using wandb => using global step (avoid overwritting)
            self.log(name, value, self.global_step if self.writer is None else self.epoch)
        self.writer_buffer = {}
        self._valid_epoch_end(device)
        self.model.train()
    
    def _valid_epoch_end(self, device):
        return

    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        # maintain topk
        if topk >= 0:
            worst_pos = -1
            while len(self.topk_ckpt_map) > topk + 1:
                last_ckpt_path = self.topk_ckpt_map[worst_pos][1]
                match = re.search(r'epoch(\d+)_step(\d+).ckpt$', last_ckpt_path)
                if match.group(1) == str(self.epoch) and match.group(2) == str(self.global_step):
                    # do not delete the current ckpt
                    worst_pos -= 1
                    if -worst_pos > len(self.topk_ckpt_map):
                        break
                    continue
                if os.path.exists(last_ckpt_path):
                    os.remove(last_ckpt_path)
                if os.path.exists(last_ckpt_path + '.states'):
                    os.remove(last_ckpt_path + '.states')
                self.topk_ckpt_map.pop(worst_pos)
            # update saved topk information
            if os.path.exists(ckpt_path):
                states = torch.load(ckpt_path + '.states', weights_only=False)
                states['attrs']['topk_ckpt_map'] = self.topk_ckpt_map
                torch.save(states, ckpt_path + '.states')

        # save map
        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def _modify_writer(self):
        return

    def _check_train_finished(self):
        finish_flag = False
        if self.config.max_epoch is not None:
            finish_flag = finish_flag or (self.epoch >= self.config.max_epoch)
        if self.config.max_step is not None:
            finish_flag = finish_flag or (self.global_step >= self.config.max_step)
        return finish_flag

    def train(self, device_ids, local_rank):
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            if self.config.logger == 'tensorboard':
                print_log(f'Using tensorboard to record the training procedure')
                self.writer = SummaryWriter(self.config.save_dir) # using tensorboard
            elif self.config.logger == 'wandb':
                print_log(f'Using wandb to record the training procedure')
                if wandb is None:
                    raise ImportError('logger is set to wandb, but package wandb is not installed')
                assert hasattr(self.config, 'proj_name'), 'proj_name must be specified in trainer.config for using wandb'
                wandb.init(
                    entity=getattr(self.config, 'team', None),
                    project=self.config.proj_name,
                    group=self.config.save_dir.rstrip(os.path.sep).split(os.path.sep)[-2],
                    name=self.config.save_dir.rstrip(os.path.sep).split(os.path.sep)[-1],
                    config=self.save_config,
                )
                atexit.register(wandb.finish) # for wandb safe exit
                self.writer = None  # using wandb.log 
            elif self.config.logger == 'swanlab':
                print_log(f'Using swanlab to record the training procedure')
                if swanlab is None:
                    raise ImportError('logger is set to swanlab, but package swanlab is not installed')
                assert hasattr(self.config, 'proj_name'), 'proj_name must be specified in trainer.config for using swanlab'
                swanlab.init(
                    workspace=getattr(self.config, 'team', None),
                    project=self.config.proj_name,
                    experiment_name=self.config.save_dir.rstrip(os.path.sep).split(os.path.sep)[-2],
                    config=self.save_config
                )
                # atexit.register(swanlab.finish)
                self.writer = None
            self._modify_writer()
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.yaml'), 'w') as fout:
                yaml.safe_dump(self.save_config, fout)
            if self.config.enable_mixed_precision:
                print_log(f'Mixed precision enabled, with dtype = {self.half_precision_type}')
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        # device = torch.device('cpu')
        self.model.to(device)
        move_optimizer_to_device(self.optimizer, device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=self.config.find_unused_parameters,
                gradient_as_bucket_view=True,  # perf: skip grad->bucket copies (~1.5%)
            )
        else:
            print_log(f'training on {device_ids}')
        while not self._check_train_finished():
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            if (self.config.val_freq_mode == 'epoch') and ((self.epoch + 1) % self.config.val_freq == 0):
                print_log(f'validating ...') if self._is_main_proc() else 1
                self._valid_epoch(device)
            self.epoch += 1
            if self.patience <= 0:
                break

    def log(self, name, value, step, val=False, batch_size=1):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if val:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].extend([value] * batch_size)
            else:
                if self.writer is None:
                    if self.config.logger == 'wandb': wandb.log({ name: value }, step=step)    # wandb
                    elif self.config.logger == 'swanlab': swanlab.log( {name: value }, step=step)
                    else: raise ValueError(f'Local writer is None but the logger name {self.config.logger} not recognized.')
                else: self.writer.add_scalar(name, value, step)         # tensorboard

    # define optimizer
    def get_optimizer(self):
        opt_cfg = deepcopy(self.config.optimizer)
        cls = getattr(torch.optim, opt_cfg.pop('class'))
        if cls is torch.optim.AdamW and os.environ.get('MEET_FUSED_ADAMW', '1') == '1':
            # perf: fused adamw kernel on DCU (~8% on b64); MEET_FUSED_ADAMW=0 to disable
            opt_cfg.setdefault('fused', True)
        optimizer = cls(self.model.parameters(), **opt_cfg)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        if not hasattr(self.config, 'scheduler'):
            return None
        sched_cfg = deepcopy(self.config.scheduler)
        cls = getattr(torch.optim.lr_scheduler, sched_cfg.pop('class'))
        freq = sched_cfg.pop('frequency')
        return {
            'scheduler': cls(optimizer, **sched_cfg),
            'frequency': freq # batch/epoch/val_epoch
        }

    ########## Overload these functions below ##########
    # train step, note that batch should be dict/list/tuple/instance. Objects with .to(device) attribute will be automatically moved to the same device as the model
    def train_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        return loss

    # validation step
    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss
