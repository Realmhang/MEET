#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import torch
from torch_scatter import scatter_mean

from data.data_struct import VAEModelInput
from utils import register as R
from utils.gnn_utils import length_to_batch_id

from .abs_trainer import Trainer


@R.register('AtomVAETrainer')
class AtomVAETrainer(Trainer): # autoregressive autoencoder trainer

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion

    def train_step(self, batch: VAEModelInput, batch_idx: int):
        if self.config.warmup != 0:
            batch.warmup_progress = min(self.global_step / self.config.warmup, 1.0)
            self.log('warmup_progress', batch.warmup_progress, batch_idx, val=False)
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch: VAEModelInput, batch_idx: int):
        loss = self.share_step(batch, batch_idx, val=True)
        return loss.detach()

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    ########## Override end ##########
    def share_step(self, batch: VAEModelInput, batch_idx: int, val: bool=False):
        loss_dict = self.model(batch)
        
        
        # torch.save(self.model.module, f'problematic_{self.local_rank}.ckpt')
        # torch.save(batch, f'problematic_batch_{self.local_rank}.ckpt')
        # if self._is_main_proc():
        #     for key in loss_dict: print(key, loss_dict[key])
        
        
        if self.is_oom_return(loss_dict):
            return loss_dict
        loss = loss_dict['total']# loss_dict['X'] + loss_dict['S'] + loss_dict['A']

        log_type = 'Validation' if val else 'Train'

        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)
            self.log('epoch', self.epoch, batch_idx, val)
            self.log('batch_size', batch.lengths.shape[0], batch_idx, val)
        return loss

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

        if hasattr(self.config, 'save_freq') and self.config.save_freq > 0 and self.global_step % self.config.save_freq == 0:
            freq_save_path = os.path.join(self.model_dir, f'freq_step{self.global_step}.ckpt')
            torch.save(module_to_save, freq_save_path)

        return save_path, states_save_path