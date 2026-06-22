#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_mean

from utils import register as R
from utils.gnn_utils import length_to_batch_id

from .atomvae_trainer import AtomVAETrainer
import os
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


@R.register('EMATrainer')
class EMATrainer(AtomVAETrainer): # autoregressive autoencoder trainer

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, criterion, config, save_config)
        self.ema_decay = config.get('ema_decay', 0.999)
        self.ema_start = config.get('ema_start', 0)
        self.ema_avg_fn = get_ema_multi_avg_fn(decay=self.ema_decay)
        self.ema_model = AveragedModel(self.model, multi_avg_fn=self.ema_avg_fn)
        self.use_ema_model = config.get('use_ema_model', True)
        self.original_model_state = None  # to store original model parameters during validation when using ema model
        self.ema_model.requires_grad_(False)

    def _train_step_end(self, device):
        if self.global_step >= self.ema_start:
            with torch.no_grad():
                self.ema_model.update_parameters(self.model)
        return super()._train_step_end(device)
    
    def _valid_epoch_begin(self, device):
        if self.use_ema_model:
            # self.original_model_state = self.model.state_dict()
            self.original_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            if self.global_step > self.ema_start:
                # self.model.load_state_dict(self.ema_model.state_dict(), strict=False)
                self.model.load_state_dict({k: v.to(device) for k, v in self.ema_model.state_dict().items()}, strict=False)
        return super()._valid_epoch_begin(device)
    
    def _valid_epoch_end(self, device):
        if self.use_ema_model:
            # self.model.load_state_dict(self.original_model_state)
            self.model.load_state_dict({k: v.to(device) for k, v in self.original_model_state.items()})
            self.original_model_state = None
        return super()._valid_epoch_end(device)

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
            'scheduler_state': None if self.scheduler is None else self.scheduler.state_dict(),
            'original_model_state': self.original_model_state,
            'ema_model_state': self.ema_model.state_dict() if self.ema_model is not None else None
        }, states_save_path)

        if hasattr(self.config, 'save_freq') and self.config.save_freq > 0 and self.global_step % self.config.save_freq == 0:
            freq_save_path = os.path.join(self.model_dir, f'freq_step{self.global_step}.ckpt')
            torch.save(module_to_save, freq_save_path)

        return save_path, states_save_path