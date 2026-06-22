#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_sum

from data.data_struct import VAEModelInput

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalPositionEmbedding
from utils.gnn_utils import length_to_batch_id

from ..AtomVAE.model_bond import AtomVAEBond
from ..AtomVAE.configs import VAESampleConfig
from ..modules.generative import make_path, make_sampler_partial
from ..modules.generative.solver import Solver
from .generator import MultiModalGenerator
from .score_net import ScoreNet
from .configs import LatentGeneratorSampleConfig



@R.register('LatentGeneratorHX')
class LatentGeneratorHX(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            hidden_size,
            net_type,
            net_opts,
            paths={
                'H': 'CosineDDPM',
                'X': 'CosineDDPM',
            },
            model_targets={
                'H': 'x0',
                'X': 'x0'
            },
            std=10.0,   # normalizing scale for X
            loss_weights={
                'H': 1.0,
                'X': 1.0
            },
            latent_deterministic=True,  # whether to add noise on the latent H/V/X for supervision
            score_net_hv_return_delta=False
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: AtomVAEBond = torch.load(autoencoder_ckpt, map_location='cpu', weights_only=False)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        model = ScoreNet(
            h_input_size=self.autoencoder.h_latent_size,
            hidden_size=hidden_size,
            net_type=net_type,
            net_opts=net_opts,
            hv_return_delta=score_net_hv_return_delta,
            x_return_delta=model_targets['X'] != 'x0'
        )

        self.generator = MultiModalGenerator(
            backbone=model,
            paths={
                'H': make_path(paths['H']),
                'X': make_path(paths['X']),
            },
            model_targets=model_targets
        )
        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.std = std
        self.loss_weights = loss_weights

    @oom_decorator
    def forward(self, inputs: VAEModelInput):

        # encode latent_H_0 (N*d1), latent_V_0 (N*d2*3), and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            encoded = self.autoencoder.encode(
                inputs.X, inputs.S, inputs.A, inputs.bonds, inputs.chain_ids, inputs.ligand_mask, inputs.block_lengths, inputs.lengths,
                deterministic=self.latent_deterministic or not self.training
            ) # [Nblock, d1], [Nblock, 3, d2], [Nblock, 3]
            Zh, Zx = encoded[:2] # [Nblock, d1], [Nblock, 3, d2], [Nblock, 3]
            if Zx.dtype != Zh.dtype: Zx = Zx.to(Zh.dtype)   # Zx is still fp32 in bf16 mode
            del encoded

            batch_ids = length_to_batch_id(inputs.lengths)

        # normalize
        Zx, centers = self._normalize_position(Zx, batch_ids, ~inputs.ligand_mask)
        # Zh = torch.randn(inputs.S.shape[0], self.autoencoder.h_latent_size, device=inputs.S.device)
        # Zx = torch.randn(inputs.S.shape[0], 3, device=inputs.S.device)
        # batch_ids = length_to_batch_id(inputs.lengths)

        
        _, loss_dict = self.generator({
            'H': Zh,
            'X': Zx,
            'position_ids': inputs.position_ids,
            'batch_ids': batch_ids,
            'generate_mask': inputs.ligand_mask
        })

        # loss
        loss = 0
        for key in self.loss_weights: loss = loss + loss_dict[key] * self.loss_weights[key]
        loss_dict['total'] = loss

        return loss_dict
    
    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def generate(
            self,
            inputs: VAEModelInput,
            configs: LatentGeneratorSampleConfig,
        ):

        # ensure there is no data leakage
        batch_ids = length_to_batch_id(inputs.lengths)
        block_ids = length_to_batch_id(inputs.block_lengths)
        inputs.S[inputs.ligand_mask] = 0
        inputs.A[inputs.ligand_mask[block_ids]] = 0
        inputs.X[inputs.ligand_mask[block_ids]] = 0

        # encoding context
        encoded = self.autoencoder.encode(
            inputs.X, inputs.S, inputs.A, inputs.bonds, inputs.chain_ids, inputs.ligand_mask, inputs.block_lengths, inputs.lengths,
            deterministic=True
        )
        Zh, Zx = encoded[:2] # [Nblock, d1], [Nblock, 3, d2], [Nblock, 3]
        
        # normalize
        Zx, centers = self._normalize_position(Zx, batch_ids, ~inputs.ligand_mask)

        # TODO: sample
        solver = Solver(
            model = self.generator,
            sampler_partials = {
                'H': make_sampler_partial(configs.h_solver_type, **configs.h_solver_args),
                'X': make_sampler_partial(configs.x_solver_type, **configs.x_solver_args),
            },
            num_steps = configs.num_steps
        )

        final_samples, intermediates = solver.sample({
            'H': Zh,
            'X': Zx,
            'position_ids': inputs.position_ids,
            'batch_ids': batch_ids,
            'generate_mask': inputs.ligand_mask
        }, records = 'none')

        generate_mask = inputs.ligand_mask
        Zh[generate_mask] = final_samples['H'].to(Zh.dtype)
        Zx[generate_mask] = final_samples['X'].to(Zx.dtype)

        # unnormalize
        Zx = self._unnormalize_position(Zx, centers, batch_ids)

        return self.autoencoder.generate(
            inputs = inputs,
            configs = VAESampleConfig(
                n_iter = configs.vae_decoder_n_iter,
                latents = [Zh, Zx]
            )
        )
