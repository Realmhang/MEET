#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn

from ..modules.generative.base_model import IterativeGenerativeModel
from ..modules.generative.path import XtInfo, ProbabilityPath


class MultiModalGenerator(IterativeGenerativeModel):
    """
    Simple diffusion model for generating H, V, and X
    """
    def __init__(self, backbone: nn.Module, paths: dict[str, ProbabilityPath], model_targets: dict[str, str], **kwargs):
        super().__init__(backbone, paths, model_targets, **kwargs)
    
    def preprocess_data(self, batch):
        '''
            batch: <key, torch.Tensor>, e.g.
            {
                'H': (N, dh)
                'V': (N, 3, dv)
                'X': (N, 3),
                'position_ids': (N)
                'batch_ids': (N)
                'generate_mask': (N)
            }
        '''
        generate_mask = batch['generate_mask']
        ligand_idx = batch['ligand_idx']  # int64 [L]; capture-safe (bool mask indexing syncs)
        x0 = { 'H': batch['H'].index_select(0, ligand_idx), 'X': batch['X'].index_select(0, ligand_idx) }
        batch_ids = batch['batch_ids'].index_select(0, ligand_idx)
        batch_size = batch_ids.max() + 1
        context_info = batch
        return x0, batch_size, batch_ids, context_info  # only the ligand part, with original data in context_info

    def postprocess_data(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None):
        # denoised H, X
        return {
            'H': xt_infos['H'].xt,
            'X': xt_infos['X'].xt,
        }

    def model_prediction(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None) -> dict[str, torch.Tensor]:
        H_input = context_info['H'].clone()
        X_input = context_info['X'].clone()
        ligand_idx = context_info['ligand_idx']  # capture-safe scatter
        H_input.index_copy_(0, ligand_idx, xt_infos['H'].xt.to(H_input.dtype))
        X_input.index_copy_(0, ligand_idx, xt_infos['X'].xt.to(X_input.dtype))
        H, X = self.backbone(
            H_input, X_input,
            context_info['position_ids'],
            context_info['batch_ids'],
            t[context_info['batch_ids']]
        )
        return {
            'H': H.index_select(0, ligand_idx),
            'X': X.index_select(0, ligand_idx)
        }