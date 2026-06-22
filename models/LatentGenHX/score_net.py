#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from utils.nn_utils import SinusoidalTimeEmbeddings
from utils.nn_utils import SinusoidalPositionEmbedding

from ..modules.MEET.meet import MEETWrapper


def create_net(hidden_size, net_type, net_opts):
    if net_opts is None: net_opts = {}
    if net_type == 'MEET':
        required_keys = ['n_heads', 'n_layers', 'adaLN_zero', 'backend']
        missing_keys = [key for key in required_keys if key not in net_opts]
        assert len(missing_keys) == 0, f'keys {missing_keys} are missing, which are required for MEET'
        return MEETWrapper(
            d_model=hidden_size,
            **net_opts
        )
    else:
        raise NotImplementedError(f'Network type {net_type} is not implemented')


class ScoreNet(nn.Module):
    def __init__(self, h_input_size, hidden_size, net_type='EPT', net_opts=None, hv_return_delta=False, x_return_delta=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.h_input_linear = nn.Linear(h_input_size + hidden_size * 2, hidden_size)
        self.time_embedding = SinusoidalTimeEmbeddings(hidden_size)
        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.h_output_linear = nn.Linear(hidden_size, h_input_size)
        self.x_output_linear = nn.Linear(hidden_size, 1, bias=False)

        self.net = create_net(hidden_size, net_type, net_opts)

        self.hv_return_delta = hv_return_delta  # return H_out - H_in
        self.x_return_delta = x_return_delta

    def forward(self, H_noisy, X_noisy, position_ids, batch_ids, t):
        """
        Args:
            H_noisy: (N, d1)
            X_noisy: (N, 3)
            batch_ids: (N)
            t: (N), range from 0 to 1
            generate_mask: (N)
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 3)
        """
        t_embed = self.time_embedding(t)    # [N, hidden_size]
        pos_embed = self.position_encoding(position_ids)
        H_input = self.h_input_linear(torch.cat([H_noisy, t_embed, pos_embed], dim=-1))

        # generate V_input
        V_input = None

        H_out, V_mid = self.net(H_input, V_input, X_noisy, batch_ids)   # [N, hidden_size], [N, 3, hidden_size]
        H_out = self.h_output_linear(H_out)
        X_add = self.x_output_linear(V_mid).squeeze(-1)    # [N, 3]
        if self.x_return_delta: X_out = X_add   # from SE(3) changed to SO(3)
        else: X_out = X_noisy + X_add

        if self.hv_return_delta:
            H_out = H_out - H_noisy

        return H_out, X_out
