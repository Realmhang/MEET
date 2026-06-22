import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


def batch_ids_to_seqlen(batch_ids):
    return scatter_sum(torch.ones_like(batch_ids), batch_ids)


def init_normal(module: Union[nn.Linear, nn.Embedding], std: float, init_cutoff_factor: Optional[float] = None):
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        nn.init.normal_(module.weight, mean=0.0, std=std)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


class ScalarNorm(nn.Module):
    def __init__(self, norm_type, num_features, elementwise_affine, elementwise_shift='none'):
        super().__init__()
        assert elementwise_affine in ['learnable', 'adaptive', 'none']
        assert elementwise_shift in ['learnable', 'adaptive', 'none']
        self.elementwise_affine = elementwise_affine
        self.elementwise_shift = elementwise_shift
        self.scale = nn.Parameter(torch.zeros(num_features)) if elementwise_affine == 'learnable' else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if elementwise_shift == 'learnable' else None
        if norm_type == 'layer_nobias':
            self.ln = nn.LayerNorm(num_features, elementwise_affine=False, bias=False)
        elif norm_type == 'rms':
            self.ln = nn.RMSNorm(num_features, elementwise_affine=False)
        else:
            raise NotImplementedError(f'Norm type {norm_type} is not implemented.')

    def forward(self, x, scale=None, bias=None):
        out = self.ln(x)
        if self.elementwise_affine == 'learnable':
            out = out * (1 + self.scale)
        elif self.elementwise_affine == 'adaptive':
            out = out * (1 + scale)
        if self.elementwise_shift == 'learnable':
            out = out + self.bias
        elif self.elementwise_shift == 'adaptive':
            out = out + bias
        return out

    def reset_parameters(self):
        self.ln.reset_parameters()
        if self.scale is not None:
            nn.init.zeros_(self.scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class VectorNorm(nn.Module):
    def __init__(self, norm_type, num_features, elementwise_affine, elementwise_shift='none'):
        super().__init__()
        assert elementwise_affine in ['learnable', 'adaptive', 'none']
        assert elementwise_shift in ['learnable', 'adaptive', 'none']
        self.elementwise_affine = elementwise_affine
        self.elementwise_shift = elementwise_shift
        self.scale = nn.Parameter(torch.zeros(num_features)) if elementwise_affine == 'learnable' else None
        self.bias = nn.Parameter(torch.zeros(num_features)) if elementwise_shift == 'learnable' else None
        if norm_type == 'layer_nobias':
            self.ln = nn.LayerNorm([3, num_features], elementwise_affine=False, bias=False, eps=0.1)
        elif norm_type == 'rms':
            self.ln = nn.RMSNorm([3, num_features], elementwise_affine=False, eps=0.1)
        else:
            raise NotImplementedError(f'Norm type {norm_type} is not implemented.')

    def forward(self, x, scale=None, bias=None):
        out = self.ln(x)
        if self.elementwise_affine == 'learnable':
            out = out * (1 + self.scale)
        elif self.elementwise_affine == 'adaptive':
            out = out * (1 + scale)
        if self.elementwise_shift == 'learnable':
            out = out + self.bias
        elif self.elementwise_shift == 'adaptive':
            out = out + bias
        return out

    def reset_parameters(self):
        self.ln.reset_parameters()
        if self.scale is not None:
            nn.init.zeros_(self.scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio, out_features, act_fn, include_bias, is_vector, drop=0.0):
        super().__init__()
        if is_vector:
            raise NotImplementedError('Vector MLP support was removed with non-MEET paths')
        hidden_features = int(in_features * mlp_ratio)
        hidden_features = max(8, hidden_features // 8 * 8)
        act_fn_dict = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'tanh': F.tanh,
            'sigmoid': F.sigmoid,
            'swiglu': swiglu,
        }
        self.fc1 = nn.Linear(in_features, hidden_features * 2 if act_fn.endswith('glu') else hidden_features, bias=include_bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=include_bias)
        self.act_fn = act_fn_dict[act_fn]
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def cross_feats(V):
    D = V.shape[-1]
    channel_swap = V[..., list(range(1, D)) + [0]]
    return torch.cat([V, torch.cross(V, channel_swap, dim=-2)], dim=-1).contiguous()


class QKV_Prep(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        assert model_config['backend'] == 'xformers', f"backend {model_config['backend']} not supported; only xformers is kept"
        self.d_hidden = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.d_head = self.d_hidden // self.n_heads
        self.att_norm = model_config['att_norm']
        self.scaler_qkv = nn.Linear(self.d_hidden, self.d_hidden * 3 + 4 * self.n_heads, bias=model_config['include_bias'])
        self.vector_qkv = nn.Linear(self.d_hidden, self.d_hidden * 3 + 4 * self.n_heads, bias=False)
        self.scaler_q_norm = ScalarNorm(model_config['norm_type'], self.d_head, model_config['norm_with_affine']) if self.att_norm else nn.Identity()
        self.scaler_k_norm = ScalarNorm(model_config['norm_type'], self.d_head, model_config['norm_with_affine']) if self.att_norm else nn.Identity()
        self.vector_q_norm = VectorNorm(model_config['norm_type'], self.d_head, model_config['norm_with_affine']) if self.att_norm else nn.Identity()
        self.vector_k_norm = VectorNorm(model_config['norm_type'], self.d_head, model_config['norm_with_affine']) if self.att_norm else nn.Identity()
        self.factor = 1 / math.sqrt(2 * self.d_head)
        self.dist_scale = nn.Parameter(torch.ones((1, self.n_heads, 1)) / math.sqrt(self.factor))
        self.init_std = model_config['init_std']
        self.init_cutoff_factor = model_config['init_cutoff_factor']

    def reset_parameters(self):
        init_normal(self.scaler_qkv, self.init_std, self.init_cutoff_factor)
        init_normal(self.vector_qkv, self.init_std, self.init_cutoff_factor)
        if self.att_norm:
            self.scaler_q_norm.reset_parameters()
            self.scaler_k_norm.reset_parameters()
            self.vector_q_norm.reset_parameters()
            self.vector_k_norm.reset_parameters()

    def forward(self, H, V, coord):
        H_qkv = self.scaler_qkv(H)
        H_q, H_k, H_v = H_qkv.split([self.d_hidden, self.d_hidden, self.d_hidden + 4 * self.n_heads], dim=-1)
        H_q = self.scaler_q_norm(H_q.unflatten(-1, [self.n_heads, self.d_head]))
        H_k = self.scaler_k_norm(H_k.unflatten(-1, [self.n_heads, self.d_head]))
        H_v = H_v.unflatten(-1, [self.n_heads, self.d_head + 4])

        V_qkv = self.vector_qkv(V)
        V_q, V_k, V_v = V_qkv.split([self.d_hidden, self.d_hidden, self.d_hidden + 4 * self.n_heads], dim=-1)
        V_q = V_q.unflatten(-1, [self.n_heads, self.d_head]).transpose(-2, -3)
        V_k = V_k.unflatten(-1, [self.n_heads, self.d_head]).transpose(-2, -3)
        V_v = V_v.unflatten(-1, [self.n_heads, self.d_head + 4]).transpose(-2, -3)
        V_q = self.vector_q_norm(V_q).flatten(start_dim=-2) / math.sqrt(3)
        V_k = self.vector_k_norm(V_k).flatten(start_dim=-2) / math.sqrt(3)
        V_v = V_v.flatten(start_dim=-2)

        x, y, z = coord.chunk(3, dim=-1)
        ones = torch.ones_like(x)
        dist_q = self.dist_scale * torch.cat([x**2, ones, -2 * x, y**2, ones, -2 * y, z**2, ones, -2 * z], dim=-1).unsqueeze(-2)
        dist_k = self.dist_scale * torch.cat([ones, x**2, x, ones, y**2, y, ones, z**2, z], dim=-1).unsqueeze(-2)
        dist_q = F.pad(dist_q, (0, 7), value=0)
        dist_k = F.pad(dist_k, (0, 7), value=0)
        attn_q = torch.cat([H_q, V_q, -dist_q], dim=-1).contiguous()
        attn_k = torch.cat([H_k, V_k, dist_k], dim=-1).contiguous()
        attn_v = torch.cat([H_v, V_v], dim=-1).contiguous()
        return attn_q, attn_k, attn_v
