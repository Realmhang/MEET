#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from data.bioparse import Atom, Block, Molecule, Complex, Bond, BondType, VOCAB, const, Atom37
from data.bioparse.utils import recur_index, index_to_numerical_index, is_aa
from utils.gnn_utils import length_to_batch_id

import pdb

########## Definition of different data structures ##########

@dataclass
class VAEModelInput:
    X: torch.Tensor                             # [Natom, 3], float, atom coordinates     
    S: torch.Tensor                             # [Nblock], int, block types
    A: torch.Tensor                             # [Natom], int, atom types
    bonds: torch.Tensor                         # [Nbonds, 3], int, chemical bonds, src-dst-type (single: 1, double: 2, triple: 3) (single-directional)
    position_ids: torch.Tensor                  # [Nblock], int, block position ids
    chain_ids: torch.Tensor                     # [Nblock], int, split different chains
    ligand_mask: torch.Tensor                   # [Nblock], bool, 1 for ligand, 0 for pocket
    block_type_generate_mask: torch.Tensor      # [Nblock], bool, 1 for generation, 0 for context
    atom_coord_generate_mask: torch.Tensor      # [Natom], bool, 1 for generated position, 0 for fixed position (block_type generated part should always be 1)
    block_lengths: torch.Tensor                 # [Nblock], int, number of atoms in each block
    lengths: Union[int, torch.Tensor]           # For single data, it is a single int. For batch data, [batch_size], int, number of blocks
    atom_names: torch.Tensor                    # [Natom], int, atom37 names
    warmup_progress: int = 1.0                  # increasing KL from 0% (0.0) to 100% (1.0) for training stability
    kwargs: dict = field(default_factory=dict)  # other key-value arguments

    # for recording necessary information (single-complex / batch-list_of_complex)
    ref_cplx: Union[Complex, List[Complex]] = None
    model_indexes: Union[List[tuple], List[List[tuple]]] = None

    @classmethod
    def batchify(cls: "VAEModelInput", items: List["VAEModelInput"]) -> "VAEModelInput":
        bonds_list = [item.bonds for item in items]
        offset = 0
        for i, bonds in enumerate(bonds_list):
            bonds[:, :2] = bonds[:, :2] + offset # src/dst
            offset += len(items[i].A)
        return cls(
            X = torch.cat([item.X for item in items], dim=0),
            S = torch.cat([item.S for item in items], dim=0),
            A = torch.cat([item.A for item in items], dim=0),
            bonds = torch.cat(bonds_list, dim=0),
            position_ids = torch.cat([item.position_ids for item in items], dim=0),
            chain_ids = torch.cat([item.chain_ids for item in items], dim=0),
            ligand_mask = torch.cat([item.ligand_mask for item in items], dim=0),
            block_type_generate_mask = torch.cat([item.block_type_generate_mask for item in items], dim=0),
            atom_coord_generate_mask = torch.cat([item.atom_coord_generate_mask for item in items], dim=0),
            block_lengths = torch.cat([item.block_lengths for item in items], dim=0),
            lengths = torch.tensor([item.lengths for item in items], dtype=torch.long),
            atom_names = torch.cat([item.atom_names for item in items], dim=0),
            ref_cplx = [item.ref_cplx for item in items],
            model_indexes = [item.model_indexes for item in items]
        )

    @classmethod
    def from_complex(
            cls: "VAEModelInput",
            cplx: Complex,
            tgt_chains: List[str],
            lig_chains: List[str],
            pocket_block_ids: Optional[List[tuple]] = None,
            ligand_block_ids: Optional[List[tuple]] = None,
            include_raw_data: bool = False
        ) -> "VAEModelInput":

        if pocket_block_ids is None:
            tgt_block_ids = []
            for c in tgt_chains: tgt_block_ids.extend([(c, block.id) for block in cplx[c]])
        else: tgt_block_ids = pocket_block_ids

        if ligand_block_ids is None:
            lig_block_ids = []
            for c in lig_chains: lig_block_ids.extend([(c, block.id) for block in cplx[c]])
        else: lig_block_ids = ligand_block_ids


        # split blocks by chain
        chain2blocks, chain2block_ids = {}, {}
        for _id in tgt_block_ids + lig_block_ids:
            chain = _id[0]
            if chain not in chain2blocks:
                chain2blocks[chain] = []
                chain2block_ids[chain] = []
            chain2blocks[chain].append(recur_index(cplx, _id))
            chain2block_ids[chain].append(_id)

        data = blocks_to_data(*chain2blocks.values())

        # mapping from atom indexes to data indexes (0, 1, 2, ...)
        atom_id2data_id = {}
        for chain in chain2block_ids:
            for block, prefix_id in zip(chain2blocks[chain], chain2block_ids[chain]):
                for atom in block:
                    atom_id = prefix_id + (atom.id,) # custom index
                    atom_id2data_id[index_to_numerical_index(cplx, atom_id)] = len(atom_id2data_id)
        # bonds
        bonds = []
        for bond in cplx.bonds:
            if bond.index1 not in atom_id2data_id or bond.index2 not in atom_id2data_id:
                continue
            bonds.append((
                atom_id2data_id[bond.index1], # src
                atom_id2data_id[bond.index2], # end
                bond.bond_type.value          # bond type
            ))
        data['bonds'] = torch.tensor(bonds, dtype=torch.long) # [E, 3]
        if len(data['bonds']) == 0: # shape: [], will cause error for collation
            data['bonds'] = torch.zeros((0, 3), dtype=torch.long)

        ligand_mask = torch.tensor([0 for _ in tgt_block_ids] + [1 for _ in lig_block_ids], dtype=torch.bool)

        # atom coord generate mask: pocket is also flexible except for those connecting with the remaining part of target protein
        # here a "terminal" block is defined as with position ids only continuous at one side (e.g. None, 2, 3, where 2 is a terminal block)
        # a terminal block should be fixed since it connects with the remaining part of proteins which are not seen in this model
        not_term_block = []
        num_tgt_blocks, position_ids, chain_ids = len(tgt_block_ids), data['position_ids'], data['chain_ids']
        for i, (p, c) in enumerate(zip(position_ids, chain_ids)):
            if i < num_tgt_blocks:
                if (i == 0) or (i == num_tgt_blocks - 1): not_term_block.append(0)
                elif (i > 0) and ((chain_ids[i - 1] != c) or ((chain_ids[i - 1] == c) and (position_ids[i - 1] < p - 1))): not_term_block.append(0)
                elif (i < num_tgt_blocks - 1) and ((chain_ids[i + 1] != c) or ((chain_ids[i + 1] == c) and (position_ids[i + 1] > p + 1))): not_term_block.append(0)
                else: not_term_block.append(1)
            else: not_term_block.append(1)  # in ligand part. all blocks should be flexible

        atom_coord_generate_mask = []
        for not_term, block_len in zip(not_term_block, data['block_lengths']):
            atom_coord_generate_mask.extend([not_term for _ in range(block_len)])
        atom_coord_generate_mask = torch.tensor(atom_coord_generate_mask, dtype=torch.bool)

        return cls(
            X = data['X'],
            S = data['S'],
            A = data['A'],
            bonds = data['bonds'],
            position_ids = data['position_ids'],
            chain_ids = data['chain_ids'],
            ligand_mask = ligand_mask,
            block_type_generate_mask = ligand_mask, # by default, the block types of the ligand should be generated while others should be fixed
            atom_coord_generate_mask = atom_coord_generate_mask,
            block_lengths = data['block_lengths'],
            lengths = data['lengths'],
            atom_names = data['atom_names'],
            # raw data
            ref_cplx = deepcopy(cplx) if include_raw_data else None,
            model_indexes = deepcopy(tgt_block_ids + lig_block_ids) if include_raw_data else None
        )

    def to(self, device):
        for key in self.__dict__.keys():
            val = getattr(self, key)
            if not hasattr(val, 'to'): continue
            setattr(self, key, val.to(device))
        return self


@dataclass
class VAEModelOutput:
    X: torch.Tensor                             # [Natom, 3], float, atom coordinates     
    S: torch.Tensor                             # [Nblock], int, block types
    A: torch.Tensor                             # [Natom], int, atom types
    bonds: torch.Tensor                         # [Nbonds, 3], int, chemical bonds, src-dst-type (single: 1, double: 2, triple: 3) (single-directional)
    predicted_bonds: torch.Tensor               # [Npred, 3], int, predicted connections between blocks
    predicted_bond_probs: torch.Tensor          # [Npred], float, probability of the predicted bonds
    block_lengths: torch.Tensor                 # [Nblock], int, number of atoms in each block
    block_type_generate_mask: torch.Tensor      # [Nblock], bool, used for deciding the atom names, either copy from the original block or use canonical atoms in the VOCAB
    lengths: Union[int, torch.Tensor]           # For single data, it is a single int. For batch data, [batch_size], int, number of blocks
    # latent states, which might be used for analysis
    Zh: torch.Tensor                            # [Nblock, h_latent_size]
    Zv: torch.Tensor                            # [Nblock, 3, v_latent_size]
    Zx: torch.Tensor                            # [Nblock, 3]
    kwargs: dict = field(default_factory=dict)  # other key-value arguments
    
    # for recording necessary information (single-complex / batch-list_of_complex)
    ref_cplx: Union[Complex, List[Complex]] = None
    model_indexes: Union[List[tuple], List[List[tuple]]] = None

    def split_batch(self) -> List["VAEModelOutput"]:
        items = []
        batch_ids = length_to_batch_id(self.lengths)
        block_ids = length_to_batch_id(self.block_lengths)
        atom_idx_offset = 0
        for i, l in enumerate(self.lengths):
            item_block_mask = (batch_ids == i)
            item_atom_mask = (batch_ids[block_ids] == i)
            item_bonds = self.bonds[item_atom_mask[self.bonds[:, 0]]]
            item_bonds[:, :2] = item_bonds[:, :2] - atom_idx_offset
            item_pred_bond_mask = item_atom_mask[self.predicted_bonds[:, 0]] if self.predicted_bonds.shape[0] > 0 else torch.zeros((self.predicted_bonds.shape[0],), dtype=bool, device=self.predicted_bonds.device)
            item_pred_bonds = self.predicted_bonds[item_pred_bond_mask]
            item_pred_bonds[:, :2] = item_pred_bonds[:, :2] - atom_idx_offset
            items.append(VAEModelOutput(
                X=self.X[item_atom_mask],
                S=self.S[item_block_mask],
                A=self.A[item_atom_mask],
                bonds=item_bonds,
                predicted_bonds=item_pred_bonds,
                predicted_bond_probs=self.predicted_bond_probs[item_pred_bond_mask],
                block_lengths=self.block_lengths[item_block_mask],
                block_type_generate_mask=self.block_type_generate_mask[item_block_mask],
                lengths=l.item(),
                Zh=self.Zh[item_block_mask],
                Zv=self.Zv[item_block_mask],
                Zx=self.Zx[item_block_mask],
                kwargs=self.kwargs,
                ref_cplx=self.ref_cplx[i] if self.ref_cplx is not None else None,
                model_indexes=self.model_indexes[i] if self.model_indexes is not None else None
            ))
            atom_idx_offset += item_atom_mask.sum()
        return items

    def to_complex(self):
        assert (not isinstance(self.ref_cplx, list)), 'Please call split_batch to get a list of single complexes first'
        assert isinstance(self.ref_cplx, Complex)
        block_atoms, block_x = [[] for _ in range(self.lengths)], [[] for _ in range(self.lengths)]
        block_ids = length_to_batch_id(self.block_lengths)
        for i, block_i in enumerate(block_ids):
            block_atoms[block_i].append(self.A[i])
            block_x[block_i].append(self.X[i].tolist())
        id2block_i = { _id: i for i, _id in enumerate(self.model_indexes) }
        block_i2id = { i: _id for i, _id in enumerate(self.model_indexes) }
        atom_i2id, block_i_atom_cnt = {}, {}
        for i, block_i in enumerate(block_ids):
            block_i = block_i.item()
            if block_i not in block_i_atom_cnt: block_i_atom_cnt[block_i] = 0
            atom_i2id[i] = (block_i2id[block_i], block_i_atom_cnt[block_i])
            block_i_atom_cnt[block_i] += 1
        atom_cnt = 0

        # atoms and coordinates
        molecules, bonds = [], []
        for mol in self.ref_cplx:
            new_blocks = []
            for block in mol:
                _id = (mol.id, block.id)
                if _id not in self.model_indexes:
                    new_block = deepcopy(block)
                    for atom in new_block:
                        atom_cnt += 1
                        atom.id = atom_cnt
                else:
                    model_block_i = id2block_i[_id]
                    if self.block_type_generate_mask[model_block_i]:
                        block_name = VOCAB.idx_to_abrv(self.S[model_block_i])
                        atom_names = VOCAB.abrv_to_atoms(block_name)
                    else:
                        block_name = block.name
                        atom_names = [atom.name for atom in block]
                    atoms = []
                    for i, name in enumerate(atom_names):
                        atom_cnt += 1
                        atoms.append(Atom(
                            name=name,
                            coordinate=block_x[model_block_i][i],
                            element=VOCAB.idx_to_atom(block_atoms[model_block_i][i]),
                            id=atom_cnt,
                            properties={'occupancy': 1.0}
                        ))
                    if block_name == 'CN': props = { 'original_name': 'NME' }
                    elif block_name == 'CC=O': props = { 'original_name': 'ACE' }
                    else: props = None
                    new_block = Block(
                        name=block_name,
                        atoms=atoms,
                        id=_id[1],
                        properties=props
                        # properties= TODO: properties?
                    )
                new_blocks.append(new_block)
            molecules.append(Molecule(name=mol.name, blocks=new_blocks, id=mol.id, properties=mol.properties))
        
        # bonds
        def numerical2index(cplx, numerical):
            mol = cplx[numerical[0]]
            block = mol[numerical[1]]
            return (mol.id, block.id)

        # retained bonds
        for bond in self.ref_cplx.bonds:
            src_id, dst_id = numerical2index(self.ref_cplx, bond.index1), numerical2index(self.ref_cplx, bond.index2)
            if ((src_id in id2block_i) and self.block_type_generate_mask[id2block_i[src_id]]) or \
               ((dst_id in id2block_i) and self.block_type_generate_mask[id2block_i[dst_id]]): continue
            bonds.append(deepcopy(bond))
        # added bonds from generated blocks
        mask = self.block_type_generate_mask[block_ids[self.bonds[:, 0]]]
        for bond_tuple in self.bonds[mask]: # [src, dst, bond type]
            src_id, src_atom_pos = atom_i2id[bond_tuple[0].item()]
            dst_id, dst_atom_pos = atom_i2id[bond_tuple[1].item()]
            bond = Bond(
                index1=index_to_numerical_index(self.ref_cplx, src_id) + (src_atom_pos,),
                index2=index_to_numerical_index(self.ref_cplx, dst_id) + (dst_atom_pos,),
                bond_type=BondType(bond_tuple[2].item())
            )
            bonds.append(bond)
        # predicted bonds
        for bond_tuple, bond_prob in zip(self.predicted_bonds, self.predicted_bond_probs):
            src_id, src_atom_pos = atom_i2id[bond_tuple[0].item()]
            dst_id, dst_atom_pos = atom_i2id[bond_tuple[1].item()]
            bond = Bond(
                index1=index_to_numerical_index(self.ref_cplx, src_id) + (src_atom_pos,),
                index2=index_to_numerical_index(self.ref_cplx, dst_id) + (dst_atom_pos,),
                bond_type=BondType(bond_tuple[2].item())
            )
            bonds.append(bond)

        return Complex(
            name=self.ref_cplx.name,
            molecules=molecules,
            bonds=bonds,
            properties=deepcopy(self.ref_cplx.properties)
        )


########## Data utils ##########

def blocks_to_data(*blocks_list: List[List[Block]]):
    '''
    an example of the returned data
    {
        'X': [Natom, 3],
        'S': [Nblock],
        'A': [Natom],
        # 'atom_order': [Natom] order of atoms within each block
        'atom_names': [Natom] atom37 names,
        'position_ids': [Nblock],
        'chain_ids': [Nblock],
        'is_aa': [Nblock]
        'block_lengths': [Nblock],
        'lengths': [1]
    }
    '''
    X, S, A, atom_names, position_ids, chain_ids, block_lengths, is_amino_acid = [], [], [], [], [], [], [], []
    for i, blocks in enumerate(blocks_list):
        insert_offset = 0 # for insertion codes
        if len(blocks) == 0:
            continue
        for block in blocks:
            # atom level variables
            atom_cnt = 0
            # if block.name in const.AA_GEOMETRY: # natural amino acid
            #     canonical_order = { atom_name: _i for _i, atom_name in enumerate(const.backbone_atoms + const.sidechain_atoms[VOCAB.abrv_to_symbol(block.name)]) }
            # else: canonical_order = {} # no order
            for atom in block:
                if atom.get_element() == 'H': continue # do not model hydrogen
                A.append(VOCAB.atom_to_idx(atom.get_element()))
                X.append(atom.get_coord())
                # atom_order.append(canonical_order.get(atom.name, atom_cnt))
                atom_name = 'CA' if atom.name == 'CH3' else atom.name
                atom_name = 'UNK' if atom_name not in Atom37.__members__ else atom_name
                atom_names.append(Atom37[atom_name].value)
                atom_cnt += 1
            if atom_cnt == 0: continue

            # block level variables
            S.append(VOCAB.abrv_to_idx(block.name))
            if block.id[1] != '' and 'original_name' not in block.properties:
                insert_offset += 1  # has insertion code and is not fragment
            position_ids.append(block.id[0] + insert_offset)    # distinguish blocks with different insertion code
            chain_ids.append(i)
            is_amino_acid.append(is_aa(block))
            block_lengths.append(atom_cnt)
            
    data = {
        'X': torch.tensor(X, dtype=torch.float),                        # [Natom, 3]
        'S': torch.tensor(S, dtype=torch.long),                         # [Nblock], block type
        'A': torch.tensor(A, dtype=torch.long),                         # [Natom]
        'position_ids': torch.tensor(position_ids, dtype=torch.long),   # [Nblock]
        'chain_ids': torch.tensor(chain_ids, dtype=torch.long),         # [Nblock]
        'block_lengths': torch.tensor(block_lengths, dtype=torch.long), # [Nblock]
        'lengths': len(S),
        'atom_names': torch.tensor(atom_names, dtype=torch.long)        # [Natom], int, atom37 names
    }

    return data