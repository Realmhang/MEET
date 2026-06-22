#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from .model.meet_layers import InitLayer, MEETLayer, FinalLayer, MEETBondLayer
from .model.utils import batch_ids_to_seqlen, init_normal

class MEET(torch.nn.Module):
    def __init__(
            self, 
            H_only=False,
            **update_config
        ):
        super().__init__()


        model_config = MEET.default_config()
        model_config.update(update_config)
        self.init_std = model_config['init_std']
        self.init_cutoff_factor = model_config['init_cutoff_factor']

        self.init_features = InitLayer(model_config)

        self.n_layers = model_config['n_layers']
        self.d_edge = model_config['d_edge']
        self.edge_mode = model_config.get('edge_mode', 'none')
        self.use_final_layer = model_config['use_final_layer']
        self.use_edge_feat = [False for _ in range(self.n_layers)]
        if self.edge_mode == 'first':
            self.attn_layers = nn.ModuleList()
            self.attn_layers.append(MEETBondLayer(model_config))
            for _ in range(1, self.n_layers):
                self.attn_layers.append(MEETLayer(model_config))
            self.use_edge_feat[0] = True
        elif self.edge_mode == 'all':
            self.attn_layers = nn.ModuleList([MEETBondLayer(model_config) for _ in range(self.n_layers)])
            self.use_edge_feat = [True for _ in range(self.n_layers)]
        else:
            self.attn_layers = nn.ModuleList([MEETLayer(model_config) for _ in range(self.n_layers)])

        self.split_outputs = model_config.get('split_outputs', False)
        if self.split_outputs:
            self.h_output_layer = MEETLayer(model_config)
            self.v_output_layer = MEETLayer(model_config)
            if self.use_final_layer:
                self.h_final_layer = FinalLayer(model_config)
                self.v_final_layer = FinalLayer(model_config)


        elif self.use_final_layer:
            self.final_layer = FinalLayer(model_config)

        self.H_only = H_only
        if self.H_only:
            self.output = nn.Linear(model_config['d_model']*2, model_config['d_model'])

        self.use_adaLN = model_config.get('adaLN_zero', False)
        if self.use_adaLN:
            self.condition_linear = nn.Linear(model_config['d_model'], 6 * model_config['d_model'])
        self.d_model = model_config['d_model']
        self.reset_parameters()
    @staticmethod
    def default_config():
        
        default_dict = {
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 16,
            'd_edge': 32,
            'use_atom_embedding': False,
            'atom_types': 10,
            'include_bias': True,
            'att_norm': True,
            'norm_position': 'pre',
            'norm_type': 'rms',
            'norm_with_affine': 'none',
            'adaLN_zero': True,
            'init_std': 0.02,  # Added for reset_parameters
            'init_cutoff_factor': 3.0,  # Added for reset_parameters
            'mlp_ratio': 3.0, # use 3.0 for blendffn v2, 0.67 for blendffn v1
            'act_fn': 'swiglu',
            'h_act_fn': 'swiglu', # for blendffn v2
            'v_act_fn': 'silu',  # for blendffn v2
            'dropout': 0.0,
            'mean_len': 512,
            'backend': 'xformers',
            'use_vector_layernorm': True,
            'use_vector_neuron': False,
            'edge_mode': 'none', # 'none', 'first', 'all'
            'use_final_layer': False,
            'split_outputs': False,
            'use_final_linear': True
        }

        return default_dict
    
    def reset_parameters(self):
        self.init_features.reset_parameters()
        for layer in self.attn_layers:
            layer.reset_parameters()
        if self.use_final_layer:
            if hasattr(self, 'split_outputs') and self.split_outputs:
                self.h_final_layer.reset_parameters()
                self.v_final_layer.reset_parameters()
            else:
                self.final_layer.reset_parameters()
        if self.use_adaLN:
            nn.init.zeros_(self.condition_linear.weight)
            nn.init.zeros_(self.condition_linear.bias)
        if self.H_only:
            init_normal(self.output, self.init_std, self.init_cutoff_factor)


    def get_seqlen_list(self, batch_ids, ligand_mask=None):
        if ligand_mask is None:
            lengths = batch_ids_to_seqlen(batch_ids)
        else:
            receptor_batch_ids = batch_ids[~ligand_mask]
            ligand_batch_ids = batch_ids[ligand_mask]
            receptor_lengths = batch_ids_to_seqlen(receptor_batch_ids)
            ligand_lengths = batch_ids_to_seqlen(ligand_batch_ids)
            lengths = torch.cat([receptor_lengths, ligand_lengths], dim=0)
        return lengths.detach().cpu().tolist()

    def rearrange_index(self, N, device, ligand_mask = None):
        forward_ids, backward_ids = torch.arange(N, device=device), torch.arange(N, device = device)
        if ligand_mask is not None:
            receptor_ids = forward_ids[~ligand_mask]
            ligand_ids = forward_ids[ligand_mask]
            forward_ids = torch.cat([receptor_ids, ligand_ids], dim=0)
            backward_ids_tmp = torch.empty_like(backward_ids)
            backward_ids_tmp[forward_ids] = backward_ids
            backward_ids = backward_ids_tmp
        return forward_ids, backward_ids
        

    def forward(self, atom_feat, coords, batch_ids, ligand_mask=None, edge_index = None, edge_attr = None, condition=None):

        if condition is not None:
            condition = self.condition_linear(condition)
            scale_H, bias_H, gate_H, scale_V, bias_V, gate_V = torch.chunk(condition, 6, dim=1)
            gate_V = gate_V.unsqueeze(-2)
            condition = (scale_H, bias_H, gate_H, scale_V, bias_V, gate_V)
        seqlen = self.get_seqlen_list(batch_ids, ligand_mask)
        N, device = atom_feat.shape[0], atom_feat.device
        forward_ids, backward_ids = self.rearrange_index(N, device, ligand_mask)
        atom_feat, coords = atom_feat[forward_ids], coords[forward_ids]

        if edge_index is not None:
            # edge_index = edge_index[:, forward_ids]
            edge_index = torch.stack([forward_ids[edge_index[0]], forward_ids[edge_index[1]]], dim=0)
        feats, seqlen_info = self.init_features(atom_feat, coords, seqlen)
        for i in range(self.n_layers):
            if hasattr(self, 'use_edge_feat') and self.use_edge_feat[i]:
                feats = self.attn_layers[i](feats, coords, seqlen_info, edge_index=edge_index, edge_attr=edge_attr, conditions=condition)
            else:
                feats = self.attn_layers[i](feats, coords, seqlen_info, conditions=condition)
        if hasattr(self, 'split_outputs') and self.split_outputs:
            feats_h = self.h_output_layer(feats, coords, seqlen_info, conditions=condition)
            feats_v = self.v_output_layer(feats, coords, seqlen_info, conditions=condition)
            if self.use_final_layer:
                feats_h = self.h_final_layer(feats_h,conditions=condition)
                feats_v = self.v_final_layer(feats_v,conditions=condition)
            H_o_from_h, V_o_from_h = feats_h
            H_o_from_v, V_o_from_v = feats_v
            H_o = H_o_from_h + H_o_from_v * 0.0
            V_o = V_o_from_h * 0.0 + V_o_from_v
            feats = (H_o, V_o)
        elif self.use_final_layer:
            feats = self.final_layer(feats,conditions=condition)
        H_o, V_o = feats
        if self.H_only:
            H_o = self.output(torch.cat([H_o, torch.norm(V_o, dim=-2)], dim=-1))
        return H_o[backward_ids], V_o[backward_ids], coords[backward_ids]
    

class MEETWrapper(torch.nn.Module):

    def __init__(
            self, 
            **update_config
        ):
        super().__init__()
        self.meet = MEET(**update_config)

    def forward(self, atom_feat, vector_feat, coords, batch_ids, ligand_mask=None, edge_index = None, edge_attr = None, condition=None):
        H, V, X = self.meet(atom_feat, coords, batch_ids, ligand_mask = ligand_mask, edge_index = edge_index, edge_attr = edge_attr, condition = condition)
        return H, V
