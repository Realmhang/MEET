#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
try:
    import xformers
    import xformers.ops as xops
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask
except ImportError:
    # DCU/HIP: no xformers build; use mathematically-equivalent SDPA shim
    from .xformers_compat import BlockDiagonalMask
    from .xformers_compat import memory_efficient_attention as _mea_compat
    class _XopsShim:
        @staticmethod
        def memory_efficient_attention(*args, **kwargs):
            return _mea_compat(*args, **kwargs)
    xops = _XopsShim()

from .utils import ScalarNorm, VectorNorm, init_normal, MLP, cross_feats, QKV_Prep, swiglu

try:
    from xformers.ops.fmha import _set_use_fa3
    _set_use_fa3(False)
    print('Disable FlashAttention 3, which causes illegal memory access in this project')
except NameError:
    pass  # xformers not installed (DCU shim path); nothing to disable
except Exception as e:
    print(f'Failed to disable FlashAttention 3: {str(e)}')


ACTFNDICT = {
    'relu': F.relu,
    'gelu': F.gelu,
    'silu': F.silu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
    'swiglu': swiglu,
}


_VAE_FUSED_PREP = __import__('os').environ.get('MEET_VAE_FUSED_PREP', '0') == '1'


class InitLayer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.d_hidden = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_head = self.d_hidden // self.n_heads
        self.backend = model_config['backend']
        assert self.backend == 'xformers', f"backend {self.backend} not supported; only xformers is kept"

        self.use_atom_embedding = model_config.get('use_atom_embedding', True)
        if self.use_atom_embedding:
            self.atom_embedding = nn.Embedding(model_config['atom_types'], self.d_hidden)
        self.o_proj = nn.Linear(self.n_heads, self.d_hidden, bias=False)

        self.att_norm = model_config['att_norm']
        self.scaler_qk = nn.Linear(self.d_hidden, self.d_hidden * 2, bias=model_config['include_bias'])
        self.scaler_q_norm = ScalarNorm(model_config['norm_type'], self.d_head, model_config['norm_with_affine']) if self.att_norm else nn.Identity()
        self.scaler_k_norm = ScalarNorm(model_config['norm_type'], self.d_head, model_config['norm_with_affine']) if self.att_norm else nn.Identity()
        self.factor = 1 / math.sqrt(self.d_head)
        self.dist_scale = nn.Parameter(torch.ones((1, self.n_heads, 1)) / math.sqrt(self.factor))
        self.init_std = model_config['init_std']
        self.init_cutoff_factor = model_config['init_cutoff_factor']

    def reset_parameters(self):
        init_normal(self.scaler_qk, self.init_std, self.init_cutoff_factor)
        if hasattr(self.scaler_q_norm, 'reset_parameters'):
            self.scaler_q_norm.reset_parameters()
        if hasattr(self.scaler_k_norm, 'reset_parameters'):
            self.scaler_k_norm.reset_parameters()
        init_normal(self.o_proj, self.init_std, self.init_cutoff_factor)

    def _prepare_qk(self, H, coord):
        H_qk = self.scaler_qk(H)
        H_q, H_k = H_qk.chunk(2, dim=-1)
        H_q = H_q.unflatten(-1, [self.n_heads, self.d_head])
        H_k = H_k.unflatten(-1, [self.n_heads, self.d_head])
        H_q = self.scaler_q_norm(H_q)
        H_k = self.scaler_k_norm(H_k)

        if _VAE_FUSED_PREP and not torch.is_grad_enabled() and coord.is_cuda:
            # frozen-VAE fast path: ONE triton kernel builds q/k/v (dist feats +
            # cat + pad + v expand). Bit-exact vs the eager chain below (unit
            # tested); the trainable LDM InitLayer path (grad enabled) is unchanged.
            from .triton_vae_prep import fused_vae_prep
            return fused_vae_prep(H_q, H_k, coord, self.dist_scale, self.n_heads)

        x, y, z = coord.chunk(3, dim=-1)
        ones = torch.ones_like(x)
        dist_q = self.dist_scale * torch.cat([x**2, ones, -2 * x, y**2, ones, -2 * y, z**2, ones, -2 * z], dim=-1).unsqueeze(-2)
        dist_k = self.dist_scale * torch.cat([ones, x**2, x, ones, y**2, y, ones, z**2, z], dim=-1).unsqueeze(-2)
        dist_q = F.pad(dist_q, (0, 7), value=0)
        dist_k = F.pad(dist_k, (0, 7), value=0)
        return torch.cat([H_q, -dist_q], dim=-1).contiguous(), torch.cat([H_k, dist_k], dim=-1).contiguous()

    def forward(self, atom_feat, coords, seqlen):
        if self.use_atom_embedding:
            atom_feat = self.atom_embedding(atom_feat)
        p_drop = self.model_config['dropout'] if self.training else 0.0
        block_diagonal_mask = BlockDiagonalMask.from_seqlens(seqlen, device=coords.device)

        atom_feat = atom_feat.unsqueeze(0)
        coords = coords.unsqueeze(0)
        _pq = self._prepare_qk(atom_feat.squeeze(0), coords.squeeze(0))
        if len(_pq) == 3:
            attn_q, attn_k, v = _pq  # fused no-grad path already built v
        else:
            attn_q, attn_k = _pq
            v = coords.unsqueeze(-2).expand(-1, -1, self.n_heads, -1)
            v = F.pad(v, (0, 5), value=0)
        attn_out = xops.memory_efficient_attention(
            attn_q.unsqueeze(0).to(v.dtype), attn_k.unsqueeze(0).to(v.dtype), v,
            scale=self.factor, attn_bias=block_diagonal_mask, p=p_drop,
        )
        vector_feat = (attn_out[..., :3] - coords.unsqueeze(-2)).transpose(-2, -1)
        vector_feat = self.o_proj(vector_feat)
        return (atom_feat.squeeze(0), vector_feat.squeeze(0)), block_diagonal_mask


class MEETAttnLayer(nn.Module):
    def __init__(self, model_config, layer_idx=-1):
        super().__init__()
        self.layer_idx = layer_idx
        self.model_config = model_config
        self.d_hidden = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_head = self.d_hidden // self.n_heads
        self.backend = model_config['backend']
        assert self.backend == 'xformers', f"backend {self.backend} not supported; only xformers is kept"
        self.qkv_prep = QKV_Prep(self.model_config)
        self.scaler_o = nn.Linear(self.d_hidden + 4 * self.n_heads, self.d_hidden, bias=model_config['include_bias'])
        self.vector_o = nn.Linear(self.d_hidden + 4 * self.n_heads, self.d_hidden, bias=False)
        self.init_std = model_config['init_std']
        self.init_cutoff_factor = model_config['init_cutoff_factor']

    def reset_parameters(self):
        init_normal(self.scaler_o, self.init_std, self.init_cutoff_factor)
        init_normal(self.vector_o, self.init_std, self.init_cutoff_factor)
        self.qkv_prep.reset_parameters()

    def forward(self, feats, coords, seqlen_info):
        p_drop = self.model_config['dropout'] if self.training else 0.0
        H, V = feats
        attn_q, attn_k, attn_v = self.qkv_prep(H, V, coords)
        attn_out = xops.memory_efficient_attention(
            attn_q.unsqueeze(0).to(attn_v.dtype), attn_k.unsqueeze(0).to(attn_v.dtype), attn_v.unsqueeze(0).to(attn_v.dtype),
            scale=self.qkv_prep.factor, attn_bias=seqlen_info, p=p_drop,
        ).squeeze(0)
        H_res, V_res = attn_out.split([self.d_head + 4, self.d_head * 3 + 12], dim=-1)
        H_res = H_res.flatten(-2)
        V_res = V_res.unflatten(-1, [3, self.d_head + 4]).transpose(-2, -3).flatten(-2)
        return self.scaler_o(H_res), self.vector_o(V_res)


class MEETFFNLayer(nn.Module):
    def __init__(self, model_config, se3=False):
        super().__init__()
        self.se3 = se3
        self.d_hidden = model_config['d_model']
        self.h_act_fn_name = model_config['h_act_fn']
        self.v_act_fn_name = model_config['v_act_fn']
        self.mlp_ratio = model_config['mlp_ratio']
        self.include_bias = model_config['include_bias']
        self.init_std = model_config['init_std']
        self.init_cutoff_factor = model_config['init_cutoff_factor']

        self.h_in = 2 * self.d_hidden
        self.v_in = 2 * self.d_hidden if self.se3 else self.d_hidden
        if self.d_hidden * self.mlp_ratio > 128:
            self.hidden_features = int(self.d_hidden * self.mlp_ratio) // 128 * 128
        else:
            self.hidden_features = int(self.d_hidden * self.mlp_ratio) // 8 * 8

        self.v_fc1 = nn.Linear(self.v_in, self.hidden_features + self.d_hidden, bias=False)
        if self.h_act_fn_name.endswith('glu'):
            self.h_fc1 = nn.Linear(self.h_in, self.hidden_features * 3, bias=self.include_bias)
        else:
            self.h_fc1 = nn.Linear(self.h_in, self.hidden_features * 2, bias=self.include_bias)
        self.h_act_fn = ACTFNDICT[self.h_act_fn_name]
        self.v_act_fn = ACTFNDICT[self.v_act_fn_name]
        self.v_fc2 = nn.Linear(self.hidden_features, self.d_hidden, bias=False)
        self.h_fc2 = nn.Linear(self.hidden_features, self.d_hidden, bias=self.include_bias)
        self.drop = nn.Dropout(model_config['dropout'])

    def reset_parameters(self):
        init_normal(self.h_fc1, self.init_std, self.init_cutoff_factor)
        init_normal(self.v_fc1, self.init_std, self.init_cutoff_factor)
        init_normal(self.v_fc2, self.init_std, self.init_cutoff_factor)
        init_normal(self.h_fc2, self.init_std, self.init_cutoff_factor)

    def forward(self, feats):
        H, V = feats
        if self.se3:
            V = cross_feats(V)
        V_proj = self.v_fc1(V)
        V1, V2 = V_proj[..., :self.d_hidden], V_proj[..., self.d_hidden:]
        H = torch.cat([H, V1.norm(dim=-2)], dim=-1)
        H_proj = self.h_fc1(H)
        H1, H2 = H_proj[..., :self.hidden_features], H_proj[..., self.hidden_features:]
        V2 = self.v_act_fn(H1).unsqueeze(-2) * V2
        H2 = self.drop(self.h_act_fn(H2))
        return self.drop(self.h_fc2(H2)), self.v_fc2(V2)


class BondAdapter(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.d_edge = model_config['d_edge']
        self.d_hidden = model_config['d_model']
        self.use_bias = model_config['include_bias']
        edge_input_size = self.d_edge + 2 * self.d_hidden
        self.edge_mlp = MLP(
            edge_input_size, self.d_hidden / edge_input_size, 2 * self.d_hidden,
            include_bias=model_config['include_bias'], is_vector=False,
            act_fn=model_config['act_fn'],
        )

    def reset_parameters(self):
        init_normal(self.edge_mlp.fc1, 0.02, 1.0)
        nn.init.zeros_(self.edge_mlp.fc2.weight)
        if self.use_bias:
            nn.init.zeros_(self.edge_mlp.fc2.bias)

    def forward(self, feats, coords, edge_index, edge_attr):
        H, V = feats
        src_idx, dst_idx = edge_index
        edge_input = torch.cat([H[src_idx], H[dst_idx], edge_attr], dim=-1)
        dH, dV_weight = self.edge_mlp(edge_input).chunk(2, dim=-1)
        dV = dV_weight.unsqueeze(-2) * V[dst_idx]
        H_out = scatter_mean(dH, src_idx, dim=0, dim_size=H.size(0))
        V_out = scatter_mean(dV, src_idx, dim=0, dim_size=V.size(0))
        return H_out, V_out


class VNSubLayerWrapper(nn.Module):
    def __init__(self, sub_layer, model_config):
        super().__init__()
        self.sub_layer = sub_layer
        self.norm_position = model_config['norm_position']
        self.adaLN_zero = model_config['adaLN_zero']
        self.use_vector_layernorm = model_config.get('use_vector_layernorm', False)
        if self.adaLN_zero:
            assert self.norm_position == 'pre', 'adaLN_zero only supports pre normalization'
            self.scalar_norm = ScalarNorm(model_config['norm_type'], model_config['d_model'], 'adaptive', 'adaptive')
            if self.use_vector_layernorm:
                self.vector_norm = VectorNorm(model_config['norm_type'], model_config['d_model'], 'adaptive', 'adaptive')
        else:
            self.scalar_norm = ScalarNorm(model_config['norm_type'], model_config['d_model'], model_config['norm_with_affine'])
            if self.use_vector_layernorm:
                self.vector_norm = VectorNorm(model_config['norm_type'], model_config['d_model'], model_config['norm_with_affine'])

    def reset_parameters(self):
        self.scalar_norm.reset_parameters()
        if self.use_vector_layernorm:
            self.vector_norm.reset_parameters()
        if hasattr(self.sub_layer, 'reset_parameters'):
            self.sub_layer.reset_parameters()

    def forward(self, feats, conditions=None, **kwargs):
        H, V = feats
        assert (conditions is not None) == self.adaLN_zero, 'conditions must be provided for adaLN_zero'
        if conditions is None:
            scale_H, bias_H, gate_H, scale_V, bias_V, gate_V = None, None, 1.0, None, None, 1.0
        else:
            scale_H, bias_H, gate_H, scale_V, bias_V, gate_V = conditions
        H0, V0 = H, V
        if self.norm_position == 'pre':
            H = self.scalar_norm(H0, scale_H, bias_H)
            if hasattr(self, 'vector_norm'):
                V = self.vector_norm(V0, scale_V, bias_V)
        H, V = self.sub_layer((H, V), **kwargs)
        if self.norm_position == 'reordered':
            H = self.scalar_norm(H, scale_H, bias_H)
            if hasattr(self, 'vector_norm'):
                V = self.vector_norm(V, scale_V, bias_V)
        H = gate_H * H + H0
        V = gate_V * V + V0
        if self.norm_position == 'post':
            H = self.scalar_norm(H, scale_H, bias_H)
            if hasattr(self, 'vector_norm'):
                V = self.vector_norm(V, scale_V, bias_V)
        return H, V


class FinalLayer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.adaLN_zero = model_config['adaLN_zero']
        self.use_bias = model_config['include_bias']
        self.use_final_linear = model_config.get('use_final_linear', True)
        if self.adaLN_zero:
            self.scalar_norm = ScalarNorm(model_config['norm_type'], model_config['d_model'], 'adaptive', 'adaptive')
            self.vector_norm = VectorNorm(model_config['norm_type'], model_config['d_model'], 'adaptive', 'adaptive')
        else:
            self.scalar_norm = ScalarNorm(model_config['norm_type'], model_config['d_model'], model_config['norm_with_affine'])
            self.vector_norm = VectorNorm(model_config['norm_type'], model_config['d_model'], model_config['norm_with_affine'])
        if self.use_final_linear:
            self.h_output = nn.Linear(model_config['d_model'], model_config['d_model'], bias=self.use_bias)
            self.v_output = nn.Linear(model_config['d_model'], model_config['d_model'], bias=False)
        self.init_std = model_config['init_std']
        self.init_cutoff_factor = model_config['init_cutoff_factor']

    def reset_parameters(self):
        self.scalar_norm.reset_parameters()
        self.vector_norm.reset_parameters()
        if self.use_final_linear:
            init_normal(self.h_output, self.init_std, self.init_cutoff_factor)
            init_normal(self.v_output, self.init_std, self.init_cutoff_factor)

    def forward(self, feats, conditions=None):
        H, V = feats
        assert (conditions is not None) == self.adaLN_zero, 'conditions must be provided for adaLN_zero'
        if conditions is None:
            scale_H, bias_H, scale_V, bias_V = None, None, None, None
        else:
            scale_H, bias_H, _, scale_V, bias_V, _ = conditions
        H = self.scalar_norm(H, scale_H, bias_H)
        V = self.vector_norm(V, scale_V, bias_V)
        if not self.use_final_linear:
            return H, V
        return self.h_output(H), self.v_output(V)


class MEETLayer(nn.Module):
    def __init__(self, model_config, layer_idx=-1):
        super().__init__()
        self.self_attn = MEETAttnLayer(model_config, layer_idx=layer_idx)
        self.blend_ffn = MEETFFNLayer(model_config, se3=True)
        self.att_wrapper = VNSubLayerWrapper(self.self_attn, model_config)
        self.ffn_wrapper = VNSubLayerWrapper(self.blend_ffn, model_config)

    def reset_parameters(self):
        self.att_wrapper.reset_parameters()
        self.ffn_wrapper.reset_parameters()

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def forward(self, feats, coords, seqlen_info, conditions=None):
        feats = self.att_wrapper(feats, conditions=conditions, coords=coords, seqlen_info=seqlen_info)
        return self.ffn_wrapper(feats, conditions=conditions)


class MEETBondLayer(MEETLayer):
    def __init__(self, model_config, layer_idx=-1):
        super().__init__(model_config, layer_idx=layer_idx)
        self.edge_adaptor = BondAdapter(model_config)
        self.edge_wrapper = VNSubLayerWrapper(self.edge_adaptor, model_config)

    def reset_parameters(self):
        super().reset_parameters()
        self.edge_wrapper.reset_parameters()

    @torch.compile(mode="max-autotune-no-cudagraphs")
    def forward(self, feats, coords, seqlen_info, edge_index, edge_attr, conditions=None):
        feats = self.edge_wrapper(feats, conditions=conditions, coords=coords, edge_index=edge_index, edge_attr=edge_attr)
        return super().forward(feats, coords, seqlen_info, conditions=conditions)
