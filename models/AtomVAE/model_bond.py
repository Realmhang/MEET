#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_sum
from torch_cluster import radius_graph, knn_graph

from data.bioparse import VOCAB, const, Atom37_NAME_LIST
from data.data_struct import VAEModelInput, VAEModelOutput
from utils.nn_utils import SinusoidalPositionEmbedding, kl_loss, expand_like, SinusoidalTimeEmbeddings
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean, scatter_sort, bipartite_knn_graph
import utils.register as R
from utils.oom_decorator import oom_decorator

from .map import block_to_atom_map, Map
from .configs import VAESampleConfig

from ..modules.nn import BlockEmbedding, MLP, init_embedding
from ..modules.create_net import create_net
from ..modules.metrics import batch_accu


def _random_mask(batch_ids, sigma=0.15):
    '''
        Get random mask position, with mask ratio 68% within 1 sigma, 95% within 2 sigma
    '''
    w = torch.empty(batch_ids.max() + 1, device=batch_ids.device) # [batch_size]
    # 68% within 1sigma (0.15), 95% within 2sigma (0.30)
    if sigma < 1e-5: # eps, near zero
        mask_ratio = w * 0.0
    else:
        mask_ratio = torch.abs(nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-1.0, b=1.0))
    mask = torch.rand_like(batch_ids, dtype=torch.float) < mask_ratio[batch_ids]
    return mask


def _local_distance_loss(X_gt, X_t, t, batch_ids, block_ids, atom_coord_generate_mask, dist_th=6.0, t_th=0.25):
    with torch.no_grad():
        row, col = radius_graph(X_gt, r=dist_th, batch=batch_ids[block_ids], loop=False)

        # at least one end is in generation part, and don't repeat the same pair
        # select_mask = (generate_mask[block_ids[row]] | generate_mask[block_ids[col]]) & (row < col)
        select_mask = (atom_coord_generate_mask[row] | atom_coord_generate_mask[col]) & (row < col)
        row, col = row[select_mask], col[select_mask]

        dist = torch.norm(X_gt[row] - X_gt[col], dim=-1) # [E]
        # # get distance within 6.0 A
        # select_mask = (dist < dist_th)
        # row, col, dist = row[select_mask], col[select_mask], dist[select_mask]

    # MSE
    dist_t = torch.norm(X_t[row] - X_t[col], dim=-1)
    loss = F.smooth_l1_loss(dist, dist_t, reduction='none')
    
    # only add loss on t steps below 0.25 (near data state)
    loss = loss[(t[row] < t_th)] # t[row] should be equal to t[col] since row and col are in the same batch
    # return loss.mean() if len(loss) else 0
    return loss.mean() if len(loss) else torch.tensor(0., device=loss.device)


def _bond_length_loss(X_gt, X_t, t, bonds, atom_coord_generate_mask, t_th=0.25):
    with torch.no_grad():
        # generate_mask = generate_mask[block_ids]
        # bond_mask = generate_mask[bonds[:, 0]] & generate_mask[bonds[:, 1]]
        bond_mask = atom_coord_generate_mask[bonds[:, 0]] & atom_coord_generate_mask[bonds[:, 1]]
        bonds = bonds[bond_mask]
        row, col = bonds[:, 0], bonds[:, 1]
        dist = torch.norm(X_gt[row] - X_gt[col], dim=-1)

    # MSE
    dist_t = torch.norm(X_t[row] - X_t[col], dim=-1)
    loss = F.smooth_l1_loss(dist, dist_t, reduction='none')
    
    # only add loss on t steps below 0.25 (near data state)
    loss = loss[(t[row] < t_th)] # t[row] should be equal to t[col] since row and col are in the same batch
    # return loss.mean() if len(loss) else 0
    return loss.mean() if len(loss) else torch.tensor(0., device=loss.device)


@R.register('AtomVAE_bond')
class AtomVAEBond(nn.Module):
    def __init__(
            self,
            encoder_type: str,
            block_type_decoder_type: str,
            struct_decoder_type: str,
            embed_size,
            edge_size,
            hidden_size,
            h_latent_size,
            x_latent_size: int = 1,         # should be 1. Multi-channel is not yet supported
            h_type_latent_size: int = 0,    # not used
            h_latent_std: float = 1.0,
            x_latent_std: float = 1.0,
            num_block_type = VOCAB.get_num_block_type(),
            num_atom_type = VOCAB.get_num_atom_type(),
            num_atom_name = len(Atom37_NAME_LIST),
            max_num_atoms = const.aa_max_n_atoms,
            k_neighbors = 9,    # intra knn and inter knn, a total of 2K neighbors
            encoder_opt = {},
            block_type_decoder_opt = {},
            struct_decoder_opt = {},
            loss_weights = {
                'Zh_kl_loss': 0.3,
                'Zx_kl_loss': 0.5,
                'atom_coord_loss': 1.0,
                'block_type_loss': 1.0,
                'contrastive_loss': 0.3,
                'local_distance_loss': 0.5,
                'bond_loss': 0.0,
                'bond_length_loss': 0.0
            },
            # auxiliary parameters
            fm_coord_prior_std=1.0,        # prior for the flow matching in the decoder
            zx_noise_scale=0.0,
            zh_noise_scale=0.0,
            pocket_mask_ratio=0.05,     # cannot be zero when kl_on_pocket=False, otherwise the latent state of the pocket will explode
            decode_mask_ratio=0.0,
            kl_on_pocket=False,         # whether to exert kl regularization also on pocket nodes
            discrete_timestep=False,
            default_num_steps=10,
            disable_block_embedding=True,
            disable_name_embedding=False,
            disable_chain_embedding=True,
            fm_bond_pred_n_rbf=32,
            fm_bond_pred_rbf_cutoff=3.5,    # meaningless to consider longer distances
            split_h=False,
            label_smoothing=0.0,
            use_gt_zx = False
        ) -> None:
        super().__init__()
        self.encoder = create_net(encoder_type, hidden_size, edge_size,False, encoder_opt)
        self.block_type_decoder = create_net(block_type_decoder_type, hidden_size, edge_size,True, block_type_decoder_opt)
        self.struct_decoder = create_net(struct_decoder_type, hidden_size, edge_size,False, struct_decoder_opt)

        self.embedding = BlockEmbedding(num_block_type, num_atom_type, embed_size, disable_block_embedding)
        # self.ctx_embedding = nn.Embedding(2, embed_size) # [context, generation]
        # self.spatial_edge_embedding = init_embedding(2, edge_size) # nn.Embedding(2, edge_size) # [intra, inter]
        self.bond_edge_embedding = init_embedding(5, edge_size) # nn.Embedding(5, edge_size) # [None, single, double, triple, aromatic]
        # self.block_spatial_edge_embedding = init_embedding(2, edge_size) # nn.Embedding(2, edge_size) # [intra, inter]
        self.enc_embed2hidden = nn.Linear(embed_size, hidden_size)
        self.dec_atom_embedding = init_embedding(num_atom_type, hidden_size) # nn.Embedding(num_atom_type, hidden_size)
        # self.dec_name_embedding = init_embedding(num_atom_name, hidden_size) # nn.Embedding(num_atom_name, hidden_size)
        self.dec_pos_embedding = SinusoidalPositionEmbedding(hidden_size)
        self.h_type_latent_size = h_type_latent_size if h_type_latent_size > 0 else h_latent_size
        self.dec_seq_latenth2hidden = nn.Linear(self.h_type_latent_size + hidden_size, hidden_size)   # H + position ids
        self.dec_time_embedding = SinusoidalTimeEmbeddings(hidden_size)
        self.split_h = split_h if h_type_latent_size > 0 else False
        self.dec_struct_input_linear_h = nn.Linear(hidden_size * (3 if disable_chain_embedding else 4) + h_latent_size - (self.h_type_latent_size if self.split_h else 0), hidden_size) # atom, atom37_name, time, position, latent
        self.disable_chain_embedding = disable_chain_embedding
        if not self.disable_chain_embedding:
            self.dec_chain_embedding = init_embedding(2, hidden_size)
        # For KL divergence
        self.Wh_mu = nn.Linear(hidden_size, h_latent_size)
        self.Wh_log_var = nn.Linear(hidden_size, h_latent_size)
        self.Wx_log_var = nn.Linear(hidden_size, x_latent_size) # has to be isotropic gaussian to maintain equivariance
        self.h_latent_std = h_latent_std
        self.x_latent_std = x_latent_std
        self.use_gt_zx = use_gt_zx
        # For output
        self.block_type_mlp = MLP(
            input_size=hidden_size, # H, V, X
            hidden_size=hidden_size,
            output_size=num_block_type,
            n_layers=3,
            dropout=0.1
        )
        # self.bond_type_mlp = MLP(
        #     input_size=hidden_size * 2 + fm_bond_pred_n_rbf,
        #     hidden_size=hidden_size,
        #     output_size=num_bond_type,
        #     n_layers=3,
        #     dropout=0.1
        # )

        self.max_num_atoms = max_num_atoms
        self.k_neighbors = k_neighbors
        self.loss_weights = loss_weights
        self.fm_coord_prior_std = fm_coord_prior_std
        self.zx_noise_scale = zx_noise_scale
        self.zh_noise_scale = zh_noise_scale
        self.pocket_mask_ratio = pocket_mask_ratio
        self.decode_mask_ratio = decode_mask_ratio
        self.kl_on_pocket = kl_on_pocket
        self.discrete_timestep = discrete_timestep
        self.default_num_steps = default_num_steps
        self.fm_bond_pred_n_rbf = fm_bond_pred_n_rbf
        self.fm_bond_pred_rbf_cutoff = fm_bond_pred_rbf_cutoff
        self.label_smoothing = label_smoothing

    @property
    def h_latent_size(self):
        return self.Wh_mu.weight.shape[0]

    @property
    def hidden_size(self):
        return self.enc_embed2hidden.weight.shape[0]
    
    @oom_decorator
    def forward(self, inputs: VAEModelInput):
        # backup ground truth
        X_gt, S_gt, A_gt = inputs.X.clone(), inputs.S.clone(), inputs.A.clone()
        block_ids = length_to_batch_id(inputs.block_lengths)
        batch_ids = length_to_batch_id(inputs.lengths)
        batch_size = inputs.lengths.shape[0]
        precision = inputs.X.dtype

        # encoding
        non_recon_kl_ratio = 0.01   # avoid exploding latent variable on context
        hkl_mask = torch.logical_or(inputs.block_type_generate_mask, torch.rand_like(inputs.block_type_generate_mask.float()) < non_recon_kl_ratio)
        block_coord_generate_mask = scatter_sum(inputs.atom_coord_generate_mask, block_ids, dim=0) > 0
        xkl_mask = torch.logical_or(block_coord_generate_mask, torch.rand_like(block_coord_generate_mask.float()) < non_recon_kl_ratio)
        Zh, Zx, Zx_mu, signed_Zx_log_var, Zh_kl_loss, Zx_kl_loss, bind_site_repr, ligand_repr = self.encode(
            inputs.X, inputs.S, inputs.A, inputs.bonds, inputs.chain_ids, inputs.ligand_mask, inputs.block_lengths, inputs.lengths,
            hkl_mask=hkl_mask, xkl_mask=xkl_mask
        ) # [Nblock, d_latent], [Nblock, 3], [1], [1]
        # Zx_log_var = -torch.abs(signed_Zx_log_var).expand_as(Zx_mu)

        if self.training: # add noise on Zx to improve robustness
            # noise = torch.randn_like(Zx) * self.zx_noise_scale
            # noise = torch.where(expand_like(block_coord_generate_mask, Zx), noise, torch.zeros_like(noise))
            Zx = Zx + torch.randn_like(Zx) * self.zx_noise_scale
            # noise = torch.randn_like(Zh) * self.zh_noise_scale
            Zh = Zh + torch.randn_like(Zh) * self.zh_noise_scale
            # noise = torch.where(expand_like(, Zh), noise, torch.zeros_like(noise))

        # # random mask some Zh for contextual prediction
        # Zh = self._random_mask(Zh, generate_mask, batch_ids)

        # decode block types from latent graph
        pred_block_logits = self.decode_block_type(Zh, Zx, inputs.chain_ids, batch_ids, inputs.ligand_mask, inputs.position_ids)

        # sample a timestep and decode structure
        if self.discrete_timestep:
            t = torch.randint(1, self.default_num_steps + 1, size=(batch_size,), device=inputs.X.device)[batch_ids][block_ids]
            t = t.to(precision) / self.default_num_steps
        else:
            t = torch.rand(batch_size, device=inputs.X.device)[batch_ids][block_ids]
        # sample an initial state
        X_t, vector_field = self._sample_from_prior(inputs.X, Zx, block_ids, inputs.atom_coord_generate_mask, t)
        row, col, bond_type = inputs.bonds.T
        # # get bond edges
        bond_edges = torch.stack([torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)], dim=0)
        bond_edge_types = torch.cat([bond_type, bond_type], dim=0)
        # # to embedding
        bond_edge_attr = self.bond_edge_embedding(bond_edge_types)
        # decode structure
        pred_vector = self.decode_structure(Zh, X_t, inputs.A, inputs.position_ids, bond_edges, bond_edge_attr, inputs.chain_ids, batch_ids, block_ids, inputs.ligand_mask, t)
        X_final = X_t + pred_vector * t[:, None] # for local neighborhood

        # loss
        # contrastive_loss_mask = scatter_sum(~inputs.ligand_mask, batch_ids, dim=0) > 0 # has target
        atom_coord_loss = F.mse_loss(
            pred_vector[inputs.atom_coord_generate_mask],
            vector_field[inputs.atom_coord_generate_mask], reduction='none'
        ).sum(-1)
        ligand_generate_mask = inputs.ligand_mask[block_ids][inputs.atom_coord_generate_mask]
        if hasattr(self, 'label_smoothing') and self.training:
            label_smoothing = self.label_smoothing
        else:
            label_smoothing = 0.0
        loss_dict = {
            'Zh_kl_loss': Zh_kl_loss,
            'Zx_kl_loss': Zx_kl_loss,
            'block_type_loss': F.cross_entropy(
                input=pred_block_logits[inputs.block_type_generate_mask],
                target=S_gt[inputs.block_type_generate_mask],
                label_smoothing=label_smoothing
            ),
            'atom_coord_loss': atom_coord_loss.mean(), # sum the xyz dimension and average between atoms
            'atom_coord_pocket_loss': atom_coord_loss[~ligand_generate_mask].mean(),
            'atom_coord_ligand_loss': atom_coord_loss[ligand_generate_mask].mean(),
            'local_distance_loss': _local_distance_loss(X_gt, X_final, t, batch_ids, block_ids, inputs.atom_coord_generate_mask),
            'bond_length_loss': _bond_length_loss(X_gt, X_final, t, inputs.bonds, inputs.atom_coord_generate_mask)
        }

        total = 0
        for name in self.loss_weights:
            weight = self.loss_weights[name]
            if 'kl_loss' in name: weight *= min(inputs.warmup_progress, 1.0)
            total = total + loss_dict[name] * weight
        loss_dict['total'] = total
        
        # some problem with this batch, will induce all nan in gradient, so manually set the loss to inf to skip this batch
        if torch.any(torch.isnan(pred_block_logits)) or torch.any(torch.isnan(pred_vector)): loss_dict['total'] = loss_dict['total'] / 0

        # for evaluation
        with torch.no_grad():
            loss_dict.update({
                'block_type_accu': batch_accu(
                    pred_block_logits[inputs.block_type_generate_mask],
                    S_gt[inputs.block_type_generate_mask],
                    batch_ids[inputs.block_type_generate_mask],
                    reduction='mean'
                ),
            })
            # record deviation of Zx and centers
            block_centers = scatter_mean(X_gt, block_ids, dim=0, dim_size=block_ids.max() + 1) # [Nblock, 3]
            zx_rmsd = ((block_centers - Zx_mu) ** 2).sum(-1) # [Nblock]
            loss_dict.update({
                'pocket_zx_center_rmsd': torch.sqrt(scatter_mean(zx_rmsd[~inputs.ligand_mask], batch_ids[~inputs.ligand_mask], dim=0)).mean(),
                'ligand_zx_center_rmsd': torch.sqrt(scatter_mean(zx_rmsd[inputs.ligand_mask], batch_ids[inputs.ligand_mask], dim=0)).mean(),
            })
            # record norm of Zh
            zh_norm = torch.norm(Zh, dim=-1) # [Nblock]
            loss_dict.update({
                'pocket_zh_norm': zh_norm[~inputs.ligand_mask].mean(),
                'ligand_zh_norm': zh_norm[inputs.ligand_mask].mean()
            })
            # record std of Zh and Zx
            # zx_std = torch.exp(-torch.abs(signed_Zx_log_var) / 2) # sigma
            zx_std = torch.exp(signed_Zx_log_var / 2) # sigma
            loss_dict.update({
                'pocket_zx_std': zx_std[~inputs.ligand_mask].mean(),
                'ligand_zx_std': zx_std[inputs.ligand_mask].mean()
            })
        # print(loss_dict)
        return loss_dict
    
    def encode(self, X, S, A, bonds, chain_ids, ligand_mask, block_lengths, lengths, deterministic=False, hkl_mask=None, xkl_mask=None):

        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)

        H = self.embedding(S, A, block_ids)
        H = self.enc_embed2hidden(H) # [Natom, hidden_size]

        # edges from spatial coordinates
        # spatial_edges, spatial_edge_types = self.get_edges(batch_ids[block_ids], chain_ids[block_ids], X, ligand_mask[block_ids], allow_lig_to_tgt=False, allow_tgt_to_lig=False)
        # spatial_edge_attr = self.spatial_edge_embedding(spatial_edge_types)

        # make bonds bidirectional
        bond_row, bond_col, bond_type = bonds.T
        bond_edges = torch.stack([  # bi-directional
            torch.cat([bond_row, bond_col], dim=0),
            torch.cat([bond_col, bond_row], dim=0)
        ], dim=0)   # [2, E]
        bond_edge_types = torch.cat([bond_type, bond_type], dim=0)
        bond_edge_attr = self.bond_edge_embedding(bond_edge_types)

        # merge two type of edges
        # edges, edge_attr = torch.cat([spatial_edges, bond_edges], dim=-1), torch.cat([spatial_edge_attr, bond_edge_attr], dim=0)

        # prepare initial V
        # V = scatter_mean(X[edges[0]] - X[edges[1]], edges[0], dim=0)    # [Natom, 3]
        # V = V.unsqueeze(-1).expand(*V.shape, H.shape[-1])               # [Natom, 3, d]
        # Zh_atom, Zv_atom, Zx_atom = self.encoder(
        #     H, V, X, edges, edge_attr,
        # ) # [Natom, hidden_size], [Natom, 3, hidden_size] [Natom, 3]
        Zh_atom, Zv_atom, Zx_atom = self.encoder(
            H, X, batch_ids[block_ids], ligand_mask = ligand_mask[block_ids], edge_index=bond_edges, edge_attr=bond_edge_attr
        ) # [Natom, hidden_size], [Natom, 3, hidden_size] [Natom, 3]
        Zh = std_conserve_scatter_mean(Zh_atom, block_ids, dim=0) # [Nblock, hidden_size]
        Zv = std_conserve_scatter_mean(Zv_atom, block_ids, dim=0) # [Nblock, 3, hidden_size]
        Zx = scatter_mean(Zx_atom, block_ids, dim=0) # [Nblock, c, 3], not use std_conserve_scatter_mean for equivariance
        Zx = Zx + Zv.mean(dim=-1)
        # assert not torch.any(torch.isnan(Zh)), 'encoding Zh nan'
        # print(Zh[generate_mask][0], Zx[generate_mask][0])

        # # contrastive learning between Target and Ligand
        # batch_size = batch_ids.max() + 1
        # tgt_repr = scatter_mean(Zh[~ligand_mask], batch_ids[~ligand_mask], dim=0, dim_size=batch_size) # [bs, hidden_size]
        # lig_repr = scatter_mean(Zh[ligand_mask], batch_ids[ligand_mask], dim=0, dim_size=batch_size) # [bs, hidden_size]
        tgt_repr, lig_repr = None, None

        # rsample: for coordinates, prior at the center of each block? or do not add constraints?
        Zx_prior_mu = scatter_mean(X, block_ids, dim=0) # [Nblock, 3]
        # Zx_prior_mu = None # do not add constraints


        if hasattr(self, 'use_gt_zx') and self.use_gt_zx:
            Zx = Zx_prior_mu + Zx * 0.0
            # Zx_prior_mu = None
        Zx_mu, signed_Zx_log_var = Zx.clone(), self.Wx_log_var(Zh)
        Zh, Zx, Zh_kl_loss, Zx_kl_loss = self.rsample(Zh, Zx, Zx_prior_mu, deterministic, hkl_mask, xkl_mask)

        return Zh, Zx, Zx_mu, signed_Zx_log_var, Zh_kl_loss, Zx_kl_loss, tgt_repr, lig_repr 
    
    def decode_block_type(self, Zh, Zx, chain_ids, batch_ids, ligand_mask, position_ids):
        '''
            Args:
                Zh: [Nblock, h_latent_size]
                Zv: [Nblock, 3, v_latent_size]
                Zx: [Nblock, 3]
                chain_ids: [Nblock]
                batch_ids: [Nblock]
                position_ids: [Nblock]
            Returns:
                pred_block_logits: [Nblock, n_block_type]
        '''
        # edges, edge_type = self.get_edges(batch_ids, chain_ids, Zx, ligand_mask, True, True)   # allow target-ligand inter edges
        # edge_attr = self.block_spatial_edge_embedding(edge_type)
        if hasattr(self, 'h_type_latent_size') and self.h_type_latent_size > 0:
            Zh = Zh[..., :self.h_type_latent_size]
        H = self.dec_seq_latenth2hidden(torch.cat([Zh, self.dec_pos_embedding(position_ids)], dim=-1))
        # V = scatter_mean(Zx[edges[0]] - Zx[edges[1]], edges[0], dim=0)    # [Natom, 3]
        # V = V.unsqueeze(-1).expand(*V.shape, H.shape[-1])               # [Natom, 3, d]
        # print(f'Zh max: {Zh.max()}')
        # print(f'Zv max: {Zv.max()}')
        # print(f'Zx max: {Zx.max()}')
        # assert not torch.any(torch.isnan(H)), 'H'
        # assert not torch.any(torch.isnan(V)), 'V'
        block_h, block_v, block_x = self.block_type_decoder(H, Zx, batch_ids)
        # assert not torch.any(torch.isnan(block_h)), 'block_h'
        # assert not torch.any(torch.isinf(block_h)), 'block_h'
        # assert not torch.any(torch.isnan(block_v)), 'block_v'
        # assert not torch.any(torch.isinf(block_v)), 'block_v'
        # assert not torch.any(torch.isnan(block_x)), 'block_x'
        # assert not torch.any(torch.isinf(block_x)), 'block_x'
        # print(f'max block x: {block_x.max()}')
        # row, col = edges
        # radial = torch.cat([
        #     (block_v[row] * block_v[col]).sum(-2),  # [E, hidden_size]
        #     # (block_x[row] - block_x[col]) ** 2,     # [E, 3]
        #     ((block_x[row] - block_x[col]) ** 2).sum(-1, keepdim=True),     # [E, 1]
        # ], dim=-1)
        # assert not torch.any(torch.isinf((block_v[row] * block_v[col]).sum(-2))), 'cross v inner dot'
        # assert not torch.any(torch.isinf(((block_x[row] - block_x[col]) ** 2).sum(-1, keepdim=True))), 'cross x delta'
        # assert not torch.any(torch.isnan(radial)), 'radial'
        # assert not torch.any(torch.isinf(radial)), 'radial'
        pred_block_logits = self.block_type_mlp(block_h)
        # pred_block_logits = self.block_type_mlp(torch.cat([block_h, scatter_mean(radial, row, dim=0, dim_size=block_h.shape[0])], dim=-1))
        # assert not torch.any(torch.isnan(pred_block_logits)), 'pred_block_logits'
        return pred_block_logits

    def decode_structure(self, Zh, X, A, position_ids, bond_edges, bond_edge_attr, chain_ids, batch_ids, block_ids, ligand_mask, t):
        '''
            Args:
                Zh: [Nblock, latent_size]
                X: [Natom, 3]
                A: [Natom]
                position_ids: [Nblock], only work for proteins/peptides. For small molecules, they are the same
                topo_edges: [2, Etopo]
                topo_edge_attr: [Etopo, edge_size]
                chain_ids: [Nblock]
                batch_ids: [Nblock]
                block_ids: [Natom]
                t: [Natom]
        '''
        # decode atom-level structures
        # edges from spatial coordinates
        # spatial_edges, spatial_edge_types = self.get_edges(batch_ids[block_ids], chain_ids[block_ids], X, ligand_mask[block_ids], allow_lig_to_tgt=True, allow_tgt_to_lig=True)
        # spatial_edge_attr = self.spatial_edge_embedding(spatial_edge_types)

        # merge two type of edges
        # edges, edge_attr = torch.cat([spatial_edges, bond_edges], dim=-1), torch.cat([spatial_edge_attr, bond_edge_attr], dim=0)
        # SE(3)-invariant input from latent h
        if hasattr(self, 'split_h') and self.split_h:
            Zh = Zh[..., self.h_type_latent_size:]

        if self.disable_chain_embedding:
            H = self.dec_struct_input_linear_h(torch.cat([
                self.dec_atom_embedding(A), self.dec_time_embedding(t), self.dec_pos_embedding(position_ids[block_ids]), Zh[block_ids]
                # self.dec_atom_embedding(A), self.dec_time_embedding(t), Zh[block_ids]
            ], dim=-1)) # [Natom, hidden_size]
        else:
            chain_type = ligand_mask[block_ids].long()
            H = self.dec_struct_input_linear_h(torch.cat([
                self.dec_atom_embedding(A), self.dec_chain_embedding(chain_type), self.dec_time_embedding(t), self.dec_pos_embedding(position_ids[block_ids]), Zh[block_ids]
                # self.dec_atom_embedding(A), self.dec_time_embedding(t), Zh[block_ids]
            ], dim=-1)) # [Natom, hidden_size]
        # initialize V
        # V = scatter_mean(X[edges[0]] - X[edges[1]], edges[0], dim=0)    # [Natom, 3]
        # V = V.unsqueeze(-1).expand(*V.shape, H.shape[-1])               # [Natom, 3, d]
        H_t, V_t, X_next = self.struct_decoder(H, X, batch_ids[block_ids], edge_index = bond_edges, edge_attr = bond_edge_attr) # [Natom', hidden_size], [Natom', 3]
        
        pred_vector = (H_t.unsqueeze(-2) * V_t + (X_next - X).unsqueeze(-1)).mean(-1) # [Natom', 3]
        return pred_vector
    
    def _random_mask(self, Zh, generate_mask, batch_ids):
        Zh = Zh.clone()

        mask = _random_mask(batch_ids, sigma=self.decode_mask_ratio)
        mask = mask & generate_mask

        Zh[mask] = self.mask_embedding
        return Zh

    def _sample_from_prior(self, X, Zx_mu, block_ids, atom_coord_generate_mask, t):
        atom_generate_mask = expand_like(atom_coord_generate_mask, X)
        Zx_mu = Zx_mu[block_ids]

        # sample random noise (directly use gaussian)
        noise = torch.randn_like(X) * self.fm_coord_prior_std
        
        # sample each atom from block prior (only the generation part)
        X_init = torch.where(atom_generate_mask, Zx_mu + noise, X)

        # vector field
        vector_field = X - X_init

        # state at timestep t (0.0 is the data, 1.0 is the prior)
        X_t = torch.where(atom_generate_mask, X_init + vector_field * (1 - t)[..., None], X)

        return X_t, vector_field
    
    @torch.no_grad()
    def _get_inter_block_nbh(self, X_t, batch_ids, block_ids, topo_generate_mask, dist_th):
        row, col = radius_graph(X_t, r=dist_th, batch=batch_ids[block_ids], loop=False)
        # inter-block and at least one end is in topo generation part, and row < col to avoid repeated bonds
        select_mask = (block_ids[row] != block_ids[col]) & (topo_generate_mask[block_ids[row]] | topo_generate_mask[block_ids[col]]) * (row < col)
        row, col = row[select_mask], col[select_mask]
        return torch.stack([row, col], dim=0) # [2, E]
        
    @torch.no_grad()
    def _get_bond_to_pred(self, X_t, gt_bonds, batch_ids, block_ids, generate_mask, neg_dist_th=3.5, neg_to_pos_ratio=1.0, given_gt=None):

        if given_gt is None:
            # get ground truth
            gt_row, gt_col, gt_type = gt_bonds[:, 0], gt_bonds[:, 1], gt_bonds[:, 2]
            # inter-block and at least one end is in generation part
            select_mask = (block_ids[gt_row] != block_ids[gt_col]) & (generate_mask[block_ids[gt_row]] | generate_mask[block_ids[gt_col]])
            gt_row, gt_col, gt_type = gt_row[select_mask], gt_col[select_mask], gt_type[select_mask]
        else:
            gt_row, gt_col = given_gt[0]
            gt_type = given_gt[1]

        # local neighborhood for negative bonds
        row, col = self._get_inter_block_nbh(X_t, batch_ids, block_ids, generate_mask, neg_dist_th)

        # negative sampling from local neighborhood (low possibility to coincide with postive bonds)
        if len(row) == 0: ratio = 0.1
        else: ratio = len(gt_row) / len(row) * neg_to_pos_ratio # neg:pos ~ 2:1
        select_mask = torch.rand_like(row, dtype=torch.float) < ratio
        row, col = row[select_mask], col[select_mask]
        neg_type = torch.zeros_like(row, dtype=torch.long)

        bonds_to_pred = torch.stack([
            torch.cat([gt_row, row], dim=0),
            torch.cat([gt_col, col], dim=0),
        ], dim=0)
        labels = torch.cat([gt_type, neg_type])

        return bonds_to_pred, labels
    
    @torch.no_grad()
    def get_edges(self, atom_batch_ids, segment_ids, Z, ligand_mask, allow_lig_to_tgt, allow_tgt_to_lig):
        
        intra_edges = knn_graph(Z, self.k_neighbors, atom_batch_ids * 2 + ligand_mask.long(), loop=False, flow='target_to_source')
        if (not allow_lig_to_tgt) and (not allow_tgt_to_lig):
            return intra_edges, torch.zeros_like(intra_edges[0])
        inter_edges = bipartite_knn_graph(Z, self.k_neighbors, atom_batch_ids, ligand_mask)
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        # 0 for intra-chain or intra-molecule edges, 1 for inter-chain or inter-molecule
        edge_type = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        return edges, edge_type
    
    @torch.no_grad()
    def _get_bond_edges(self, bonds, block_ids, generate_mask):
        '''
        Only used in training
            bonds: [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            block_ids: [N]
            generate_mask: [Nblock]
        '''
        row, col, bond_type = bonds.T
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0) # bidirectional
        bond_type = torch.cat([bond_type, bond_type], dim=-1) #[2*Nbonds, edge_size]

        # # TODO: for block-level: two blocks are connected if there are two atoms having chemical bonds
        # # TODO: if this is enabled, there must also be an block edge prediction module
        # inter_block = block_ids[row] != block_ids[col]
        # row_block, col_block = block_ids[row][inter_block], block_ids[col][inter_block]
        # block_topo_edges = torch.unique(torch.stack([row_block, col_block], dim=0), dim=1) # [2, Eblock]

        # for atom-level chemical bonds: only give intra-block ones for generated part, but all for the fixed part
        # intra_block = block_ids[row] == block_ids[col]
        # row, col, bond_type = row[intra_block], col[intra_block], bond_type[intra_block]
        row_block, col_block = block_ids[row], block_ids[col]
        select_mask = (row_block == col_block) | ((~generate_mask[row_block]) & (~generate_mask[col_block]))
        row, col, bond_type = row[select_mask], col[select_mask], bond_type[select_mask]
        bond_edges = torch.stack([row, col], dim=0) # [2, E]

        return bond_edges, bond_type
    
    @torch.no_grad()
    def _unmask_inter_bond_edges(self, bonds, batch_ids, block_ids, generate_mask):
        atom_batch_ids = batch_ids[block_ids]
        row, col, bond_type = bonds.T

        # get inter-block bonds
        row_block, col_block = block_ids[row], block_ids[col]
        select_mask = (row_block != col_block) & (generate_mask[row_block] | generate_mask[col_block])
        row, col, bond_type = row[select_mask], col[select_mask], bond_type[select_mask]

        # # sample some to provide as contexts, others for prediction
        # unmask_ratio = torch.rand(batch_ids.max() + 1, device=bonds.device)
        # select_mask = torch.rand_like(atom_batch_ids[row], dtype=torch.float) < unmask_ratio[atom_batch_ids[row]]
        # 50% cases for structure prediction, others for design
        unmask = torch.rand(batch_ids.max() + 1, device=bonds.device) < 0.5
        select_mask = unmask[atom_batch_ids[row]]

        # bi-directional
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        bond_type = torch.cat([bond_type, bond_type], dim=0)
        select_mask = torch.cat([select_mask, select_mask], dim=0)

        return torch.stack([row, col], dim=0), bond_type, select_mask

    def rsample(self, Zh, Zx, Zx_prior_mu=None, deterministic=False, hkl_mask=None, xkl_mask=None):
        '''
            Zh: [Nblock, dh]
            Zv: [Nblock, 3, dv]
            Zx: [Nblock, dx, 3]
            Zx_prior_mu: [Nblock, 3]
        '''

        # data_size = Zh.shape[0]

        # invariant latent features
        Zh_mu = self.Wh_mu(Zh)
        Zh_log_var = -torch.abs(self.Wh_log_var(Zh)) #Following Mueller et al., z_log_var is log(\sigma^2))
        if hkl_mask is None: Zh_kl_loss = kl_loss(Zh_mu, Zh_log_var, prior_std=self.h_latent_std, average=True)
        else: Zh_kl_loss = kl_loss(Zh_mu[hkl_mask], Zh_log_var[hkl_mask], prior_std=self.h_latent_std, average=True)
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)

        # SE(3) equivariant latent features
        if Zx_prior_mu is None:
            Zx_sampled, Zx_kl_loss = Zx + torch.randn_like(Zx), 0 # fix as standard gaussian
        else:
            # Zx_mu_delta = Zx - Zx_prior_mu # [Nblock, c, 3], if perfectly from prior, the expectation should be zero
            # Zx_log_var = -torch.abs(self.Wx_log_var(Zh)).expand_as(Zx)
            Zx_log_var = self.Wx_log_var(Zh).expand_as(Zx)
            # Zx_kl_loss = -0.5 * torch.sum((1.0 + Zx_log_var - Zx_mu_delta * Zx_mu_delta - torch.exp(Zx_log_var))) / (data_size * Zx.shape[-1])
            if xkl_mask is None: Zx_kl_loss = kl_loss(Zx, Zx_log_var, Zx_prior_mu, self.x_latent_std, average=True)
            else: Zx_kl_loss = kl_loss(Zx[xkl_mask], Zx_log_var[xkl_mask], Zx_prior_mu[xkl_mask], self.x_latent_std, average=True)
            Zx_sampled = Zx if deterministic else Zx + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx)
        
        return Zh_sampled, Zx_sampled, Zh_kl_loss, Zx_kl_loss

    def _init_atoms(self, pred_block_type, X, A, atom_names, bonds, Zx_mu, block_ids, block_type_generate_mask, atom_coord_generate_mask):

        # get bonds of context
        ctx_block = (~block_type_generate_mask[block_ids[bonds[:, 0]]]) & (~block_type_generate_mask[block_ids[bonds[:, 1]]])
        bonds = bonds[ctx_block] # [Ectx, 3]

        # extract context
        atom_ctx_mask = ~block_type_generate_mask[block_ids]
        ctx_X, ctx_A, ctx_block_ids = X[atom_ctx_mask], A[atom_ctx_mask], block_ids[atom_ctx_mask]
        ctx_atom_names = atom_names[atom_ctx_mask]

        # initialize coordinates in the context which are required to be generated
        ctx_atom_coord_generate_mask = atom_coord_generate_mask[atom_ctx_mask]
        ctx_X = torch.where(
            expand_like(ctx_atom_coord_generate_mask, ctx_X),
            Zx_mu[ctx_block_ids] + torch.randn_like(Zx_mu[ctx_block_ids]) * self.fm_coord_prior_std,
            ctx_X
        )

        # order mapping of context bonds
        ctx_atom_order_map = -torch.ones_like(block_ids, dtype=torch.long) # -1 will be non-context atoms
        ctx_atom_order_map[atom_ctx_mask] = torch.arange(atom_ctx_mask.long().sum(), device=atom_ctx_mask.device)
        bonds = torch.stack([
            ctx_atom_order_map[bonds[:, 0]], ctx_atom_order_map[bonds[:, 1]], bonds[:, 2]
        ], dim=-1) # [E, 3]

        # generate atoms/bonds based on predicted blocks
        gen_A, gen_atom_names, gen_block_ids, gen_bonds = block_to_atom_map(pred_block_type[block_type_generate_mask], torch.nonzero(block_type_generate_mask).squeeze(-1), return_atom_names=True)
        gen_X = Zx_mu[gen_block_ids] + torch.randn_like(Zx_mu[gen_block_ids]) * self.fm_coord_prior_std
        gen_atom_coord_generate_mask = torch.ones_like(gen_block_ids, dtype=bool)

        # concat context and generated part
        # atoms
        X, A, block_ids = torch.cat([ctx_X, gen_X], dim=0), torch.cat([ctx_A, gen_A], dim=0), torch.cat([ctx_block_ids, gen_block_ids], dim=0)
        atom_names = torch.cat([ctx_atom_names, gen_atom_names], dim=0)
        atom_coord_generate_mask = torch.cat([ctx_atom_coord_generate_mask, gen_atom_coord_generate_mask], dim=0)
        # bonds
        ctx_row, ctx_col, ctx_bond_type = bonds[:, 0], bonds[:, 1], bonds[:, 2]
        gen_row, gen_col, gen_bond_type = gen_bonds[:, 0] + len(ctx_A), gen_bonds[:, 1] + len(ctx_A), gen_bonds[:, 2]
        bonds = torch.stack([
            torch.cat([ctx_row, ctx_col, gen_row, gen_col], dim=0), # bidirectional
            torch.cat([ctx_col, ctx_row, gen_col, gen_row], dim=0), # bidirectional
            torch.cat([ctx_bond_type, ctx_bond_type, gen_bond_type, gen_bond_type], dim=0)
        ], dim=-1) # [Ectx*2 + Egen*2, 3]
        
        # sorting
        block_ids, perm = scatter_sort(block_ids, block_ids, dim=0)
        X, A, atom_coord_generate_mask = X[perm], A[perm], atom_coord_generate_mask[perm]
        atom_names = atom_names[perm]
        # atom order mapping
        atom_order_map = torch.ones_like(A, dtype=torch.long)
        atom_order_map[perm] = torch.arange(len(A), device=A.device)
        bonds = torch.stack([
            atom_order_map[bonds[:, 0]],
            atom_order_map[bonds[:, 1]],
            bonds[:, 2]
        ], dim=-1)
        return X, A, atom_names, block_ids, bonds, atom_coord_generate_mask

    def _get_pep_bonds(self, pred_block_type, position_ids, block_ids, chain_ids, block_type_generate_mask, bidirectional=True):
        '''
            Generate peptide bonds between adjacent residues which are generated (fixed part already have their peptide bonds)
        '''
        data_map = Map()
        block_lengths = scatter_sum(torch.ones_like(block_ids), block_ids, dim=0)
        offsets = F.pad(block_lengths.cumsum(dim=0)[:-1], pad=(1, 0), value=0)
        indexes = torch.arange(0, position_ids.shape[0], device=offsets.device)
        left_res, right_res = indexes[:-1], indexes[1:]
        is_aa = data_map.is_aa(pred_block_type)
        # WARNING: the cappings are specialized, which might cause trouble in corner cases (e.g. mis-pred NME/ACE at incorrect positions)
        # N-terminal capping (ACE)
        ace_index = left_res[data_map.is_ace_capping(pred_block_type)[:-1] & is_aa[1:]] # ACE=>AA
        ace_row, ace_col = (offsets[ace_index] + data_map.ace_c_order), (offsets[ace_index + 1] + data_map.term_n_order)
        # C-terminal capping (NME)
        nme_index = right_res[data_map.is_nme_capping(pred_block_type)[1:] & is_aa[:-1]] # AA=>NME
        nme_row, nme_col = (offsets[nme_index] + data_map.nme_n_order), (offsets[nme_index - 1] + data_map.term_c_order)
        # normal peptide bonds
        adj_mask = is_aa[left_res] & is_aa[right_res] & \
                   block_type_generate_mask[left_res] & block_type_generate_mask[right_res] & \
                   (chain_ids[left_res] == chain_ids[right_res]) & \
                   ((position_ids[right_res] - position_ids[left_res]) == 1)
        left_res, right_res = left_res[adj_mask], right_res[adj_mask]
        # from C of left_res to N of right_res
        row, col = (offsets[left_res] + data_map.term_c_order), (offsets[right_res] + data_map.term_n_order)
        # concat
        row, col = torch.cat([row, ace_row, nme_row], dim=0), torch.cat([col, ace_col, nme_col], dim=0)

        bond_type = torch.ones_like(row) * data_map.single_bond_type
        # bidirectional
        if bidirectional:
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            bond_type = torch.cat([bond_type, bond_type], dim=0)
        return torch.stack([row, col, bond_type], dim=-1)

    def _get_capping_bonds(self, pred_block_type, position_ids, block_ids, chain_ids, block_type_generate_mask, bidirectional=True):
        '''
            Generate capping bonds between adjacent residues which are generated
        '''
        data_map = Map()
        block_lengths = scatter_sum(torch.ones_like(block_ids), block_ids, dim=0)
        offsets = F.pad(block_lengths.cumsum(dim=0)[:-1], pad=(1, 0), value=0)
        indexes = torch.arange(0, position_ids.shape[0], device=offsets.device)
        left_res, right_res = indexes[:-1], indexes[1:]
        is_aa = data_map.is_aa(pred_block_type)
        adj_mask = is_aa[left_res] & is_aa[right_res] & \
                   block_type_generate_mask[left_res] & block_type_generate_mask[right_res] & \
                   (chain_ids[left_res] == chain_ids[right_res]) & \
                   ((position_ids[right_res] - position_ids[left_res]) == 1)
        left_res, right_res = left_res[adj_mask], right_res[adj_mask]
        # from C of left_res to N of right_res
        row, col = (offsets[left_res] + data_map.term_c_order), (offsets[right_res] + data_map.term_n_order)
        bond_type = torch.ones_like(row) * data_map.single_bond_type
        # bidirectional
        if bidirectional:
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
            bond_type = torch.cat([bond_type, bond_type], dim=0)
        return torch.stack([row, col, bond_type], dim=-1)

    def generate(
        self,
        inputs: VAEModelInput,
        configs: VAESampleConfig
    ) -> VAEModelOutput:
        
        # if self.discrete_timestep: assert n_iter == self.default_num_steps
        if configs.latents is not None: Zh, Zx = configs.latents
        else:   # reconstruction test
            Zh, Zx, _, _, _, _, _, _ = self.encode(
                inputs.X, inputs.S, inputs.A, inputs.bonds, inputs.chain_ids, inputs.ligand_mask, inputs.block_lengths, inputs.lengths, deterministic=True
            ) # [Nblock, h_latent_size], [Nblock, 3, v_latent_size], [Nblock, 3]
        
        if configs.manual_noise_h_std is not None:
            noise = torch.where(expand_like(inputs.ligand_mask, Zh), torch.randn_like(Zh) * configs.manual_noise_h_std, torch.zeros_like(Zh))
            Zh = Zh + noise
        if configs.manual_noise_x_std is not None:
            noise = torch.where(expand_like(inputs.ligand_mask, Zx), torch.randn_like(Zx) * configs.manual_noise_x_std, torch.zeros_like(Zx))
            Zx = Zx + noise

        precision = Zx.dtype

        
        block_ids = length_to_batch_id(inputs.block_lengths)
        batch_ids = length_to_batch_id(inputs.lengths)
        # if resample_mode: Zh = self._random_mask(Zh, generate_mask, batch_ids)

        # if topo_generate_mask is None: topo_generate_mask = generate_mask
        # start_t = 0.5
        if inputs.block_type_generate_mask.sum() != 0:
            # decode block types from latent graph
            pred_block_logits = self.decode_block_type(Zh, Zx, inputs.chain_ids, batch_ids, inputs.ligand_mask, inputs.position_ids)
            # force the first is ACE, the last is NME, and in the middles are normal amino acids
            # WARNING: assume position ids is increasing for peptides
            is_ace = F.pad((~inputs.ligand_mask[:-1]) & inputs.ligand_mask[1:], pad=(1, 0), value=False)
            is_nme = F.pad((~inputs.ligand_mask[1:]) & inputs.ligand_mask[:-1], pad=(0, 1), value=True)
            is_aa = ~(is_ace | is_nme)
            non_aa_mask = ~torch.tensor(VOCAB.aa_mask, dtype=torch.bool, device=is_aa.device)
            pred_block_logits = pred_block_logits.masked_fill(non_aa_mask[None, :] & is_aa[:, None], float('-inf'))
            pred_block_prob = torch.softmax(pred_block_logits, dim=-1) # [Nblock, num_block_type]
            # if resample_mode: pred_block_type = torch.multinomial(pred_block_prob, num_samples=1).squeeze(-1)
            # else: prob, pred_block_type = torch.max(pred_block_prob, dim=-1) # [Nblock]
            prob, pred_block_type = torch.max(pred_block_prob, dim=-1) # [Nblock]
            pred_block_type[~inputs.block_type_generate_mask] = inputs.S[~inputs.block_type_generate_mask]
            pred_block_type[is_nme] = Map().nme_idx
            pred_block_type[is_ace] = Map().ace_idx
        
            # initialize (append atoms and sample coordinates)
            X_t, A, atom_names, block_ids, bonds, atom_coord_generate_mask = self._init_atoms(
                pred_block_type, inputs.X, inputs.A, inputs.atom_names, inputs.bonds, Zx, block_ids, inputs.block_type_generate_mask, inputs.atom_coord_generate_mask)
            # add peptide bonds between adjacent residues
            pep_bonds = self._get_pep_bonds(pred_block_type, inputs.position_ids, block_ids, inputs.chain_ids, inputs.block_type_generate_mask)
            bonds = torch.cat([bonds, pep_bonds], dim=0)
        else: # purely structure prediction
            pred_block_type = inputs.S.clone()
            # only need to initialize atoms
            random_X = Zx[block_ids] + torch.randn_like(Zx[block_ids]) * self.fm_coord_prior_std
            X_t = torch.where(expand_like(inputs.atom_coord_generate_mask, inputs.X), random_X, inputs.X)
            # bidirectional
            _row, _col, _type = inputs.bonds[:, 0], inputs.bonds[:, 1], inputs.bonds[:, 2]
            bonds = torch.stack([
                torch.cat([_row, _col], dim=0),
                torch.cat([_col, _row], dim=0),
                torch.cat([_type, _type], dim=0)
            ], dim=1)
            A, atom_names, atom_coord_generate_mask = inputs.A, inputs.atom_names, inputs.atom_coord_generate_mask
        
        # concat context bonds and generated bonds
        bond_edges, bond_edge_types = bonds[:, :2].T, bonds[:, 2]
        bond_edge_attr = self.bond_edge_embedding(bond_edge_types)

        # iterative
        X_init = X_t.clone()
        all_vectors, span = [], 1.0 / configs.n_iter
        X_gen_mask = expand_like(atom_coord_generate_mask, X_t)

        added_bond_edges, added_bond_attr = None, None

        for i in range(configs.n_iter):
            t = (1.0 - i * span) * torch.ones_like(block_ids, dtype=precision)
            # to embedding
            # H_t, V_t, X_next = self.decode_structure(
            pred_vector = self.decode_structure(
                Zh, X_t, A, inputs.position_ids, 
                bond_edges if added_bond_edges is None else torch.cat([bond_edges, added_bond_edges], dim=-1),
                bond_edge_attr if added_bond_attr is None else torch.cat([bond_edge_attr, added_bond_attr], dim=0),
                inputs.chain_ids, batch_ids, block_ids, inputs.ligand_mask, t
            )
            pred_vector = torch.where(
                X_gen_mask,
                # X_next - X_t, # + self._bond_length_guidance((1.0 - i * span), H_t, X_t, batch_ids, block_ids, generate_mask),
                # X_next - X_t, # + _inter_clash_guidance((1.0 - i * span), A, X_t, batch_ids, block_ids, chain_ids, generate_mask),
                pred_vector,
                torch.zeros_like(X_t))
            X_t = X_t + pred_vector * span # update
            all_vectors.append(pred_vector)


        X = X_t

        if configs.return_tensor:
            # return Zh, Zv, Zx, H_t, V_t, X
            return Zh, Zx, X
        # VLB for iterative process (the smaller, the better)
        ll = ((X_t - X_init).unsqueeze(0) - torch.stack(all_vectors, dim=0)) ** 2 # [T, Natom, 3]
        ll = ll.sum(-1).mean(0) # [Natom]

        # disable bond prediction
        row = torch.zeros([0], dtype=torch.long, device=X.device)
        col, bond_type, bond_prob = row.clone(), row.clone(), row.clone().to(precision)

        return VAEModelOutput(
            X = X,
            S = pred_block_type,
            A = A,
            bonds = bonds,
            predicted_bonds = torch.stack([row, col, bond_type], dim=-1),
            predicted_bond_probs = bond_prob,
            block_lengths = scatter_sum(torch.ones_like(block_ids), block_ids, dim=0),
            block_type_generate_mask = inputs.block_type_generate_mask,
            lengths = scatter_sum(torch.ones_like(batch_ids), batch_ids, dim=0),
            Zh = Zh,
            Zv = Zx.unsqueeze(-1),
            Zx = Zx,
            ref_cplx = inputs.ref_cplx,
            model_indexes = inputs.model_indexes
        )
