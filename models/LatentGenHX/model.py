#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

import os
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




def _pad_latent_batch(H, X, position_ids, batch_ids, generate_mask, n_peptides, bucket=128):
    """Append ONE fake peptide (id = n_peptides) so that total tokens and ligand
    token counts are multiples of `bucket`. Real peptides' block-diagonal
    attention is unaffected; the fake peptide contributes no loss (masked)."""
    N = H.shape[0]
    L = int(generate_mask.sum().item())
    if os.environ.get('MEET_BUCKET_PAD_LIG', '1') == '0':
        L = -(-L // bucket) * bucket  # diagnostic: no fake ligand tokens
    lbucket = int(os.environ.get('MEET_BUCKET_PAD_L', '0')) or bucket
    Lp = -(-L // lbucket) * lbucket
    extra_lig = Lp - L
    Np = -(-(N + extra_lig) // bucket) * bucket
    extra = Np - N
    if 0 < extra <= extra_lig:
        # need padding but no ligand tokens in the fake peptide: t[batch_ids]
        # in model_prediction would index out of bounds (batch_size too small).
        # force at least one fake ligand token (shift one token from receptor side).
        extra_lig = extra
    if extra > 0 and extra_lig == 0:
        extra_lig = 1
        extra = max(extra, 1)
    batch = {
        'H': H, 'X': X, 'position_ids': position_ids,
        'batch_ids': batch_ids, 'generate_mask': generate_mask,
        'real_sample_mask': None,
    }
    if extra <= 0:
        return batch
    dev = H.device
    # split padding into EXACTLY TWO fake peptides: keeps the sequence count
    # (grid dim B) static for graph capture, bounds each fake's length, and
    # cuts the fake-attention quadratic cost ~4x vs one giant fake sequence.
    half = extra // 2
    l1, l2 = half, extra - half  # each <= 512 for the supported bucket range
    n1 = max(extra_lig, 0)       # all fake ligand tokens live in fake #1
    # non-degenerate fake values (all-zero latents can produce NaN activations
    # in vector norms, which poisons backward via NaN*0 gradients)
    fake_h = torch.randn(extra, H.shape[-1], device=dev, dtype=H.dtype) * 0.02
    fake_x = torch.randn(extra, X.shape[-1], device=dev, dtype=X.dtype) * 0.02
    fake_pos = torch.zeros(extra, dtype=position_ids.dtype, device=dev)
    ids1 = torch.full((l1,), n_peptides, dtype=batch_ids.dtype, device=dev)
    ids2 = torch.full((l2,), n_peptides + 1, dtype=batch_ids.dtype, device=dev)
    fake_mask = torch.zeros(extra, dtype=torch.bool, device=dev)
    fake_mask[:min(n1, l1)] = True  # fake #1's ligand tokens
    batch['H'] = torch.cat([H, fake_h], dim=0)
    batch['X'] = torch.cat([X, fake_x], dim=0)
    batch['position_ids'] = torch.cat([position_ids, fake_pos], dim=0)
    batch['batch_ids'] = torch.cat([batch_ids, ids1, ids2], dim=0)
    batch['generate_mask'] = torch.cat([generate_mask, fake_mask], dim=0)
    # per-sample loss mask over ligand-carrying samples (dim_size = B_real+1):
    # fake #1 (id n_peptides) is the only fake with ligand tokens -> mask it.
    rsm = torch.ones(n_peptides + 1, device=dev, dtype=H.dtype)
    rsm[-1] = 0.0
    batch['real_sample_mask'] = rsm
    batch['fake_seqlens'] = (l1, l2)
    return batch


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

    def _prepare_gen_batch(self, inputs: VAEModelInput):
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

        
        _bucket = int(os.environ.get('MEET_BUCKET_PAD', '0'))
        if self.training and _bucket > 0:
            gen_batch = _pad_latent_batch(Zh, Zx, inputs.position_ids, batch_ids,
                                          inputs.ligand_mask, inputs.lengths.numel(), _bucket)
        else:
            gen_batch = {
                'H': Zh,
                'X': Zx,
                'position_ids': inputs.position_ids,
                'batch_ids': batch_ids,
                'generate_mask': inputs.ligand_mask,
            }

        gen_batch['ligand_idx'] = gen_batch['generate_mask'].nonzero(as_tuple=False).squeeze(-1)
        seqlens = inputs.lengths.detach().cpu().tolist()
        Np = gen_batch['H'].shape[0]
        Nreal = sum(seqlens)
        if Np > Nreal and 'fake_seqlens' in gen_batch:
            seqlens = seqlens + list(gen_batch['fake_seqlens'])  # exactly two fakes
        elif Np > Nreal:
            seqlens = seqlens + [Np - Nreal]
        gen_batch['seqlens'] = seqlens
        return gen_batch

    @oom_decorator
    def forward(self, inputs: VAEModelInput):
        gen_batch = self._prepare_gen_batch(inputs)
        _, loss_dict = self.generator(gen_batch)

        # loss
        loss = 0
        for key in self.loss_weights: loss = loss + loss_dict[key] * self.loss_weights[key]
        loss_dict['total'] = loss

        return loss_dict

    def _meet_graph_step(self, inputs, scaler=None, autocast_dtype=None, ddp=None):
        """Graph-captured train step: returns {'total': static loss tensor}.
        Forward + scaled backward (incl. DDP allreduce) run inside one replay."""
        if not hasattr(self, '_graph_driver'):
            from models.LatentGenHX.graph_step import _GraphedGen
            self._graph_driver = _GraphedGen(self, scaler, autocast_dtype, ddp)
        if autocast_dtype is not None:
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                gen_batch = self._prepare_gen_batch(inputs)
                loss = self._graph_driver.step(gen_batch)
        else:
            gen_batch = self._prepare_gen_batch(inputs)
            loss = self._graph_driver.step(gen_batch)
        return loss  # tensor: trainer expects a scalar loss (isnan/backward)
    
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
