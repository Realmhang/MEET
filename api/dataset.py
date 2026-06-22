#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import List
from copy import deepcopy

import torch

from data.data_struct import VAEModelInput
from data.bioparse.hierarchy import Molecule
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.utils import recur_index, assign_new_chain_id
from data.bioparse.interface import compute_interacting_pairs
from utils.logger import print_log
import utils.register as R


@R.register('FileDataset')
class FileDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths: List[str], tgt_chains: List[List[str]], lig_chains: List[List[str]]=None, hotspots: List[List[tuple]]=None, pocket_dist_th: float=20.0):
        super().__init__()
        self.file_paths = file_paths
        self.tgt_chains = tgt_chains
        assert not (lig_chains is None and hotspots is None), 'Either reference compound (lig_chains) or hotspots should be provided!'
        if (lig_chains is not None) and (hotspots is not None):
            print_log('Both lig_chains and hotspots are provided! Default using lig_chains to define the binding site!', level='WARN')
        self.lig_chains = [None for _ in file_paths] if lig_chains is None else lig_chains
        # hotspots for distinguishing pockets when reference compound is not available
        self.hotspots = [None for _ in file_paths] if hotspots is None else hotspots
        self.pocket_dist_th = pocket_dist_th
        self.cplxs, self.pocket_block_ids = [], []

        for i, (path, tgt, lig, htspt) in enumerate(zip(self.file_paths, self.tgt_chains, self.lig_chains, self.hotspots)):
            cplx, pocket_block_ids, lig = self._process_file(path, tgt, lig, htspt)
            self.cplxs.append(cplx)
            self.pocket_block_ids.append(pocket_block_ids)

    def get_id(self, idx):
        return os.path.splitext(os.path.basename(self.file_paths[idx]))[0]

    def __getitem__(self, idx):
        cplx, pocket_block_ids = self.cplxs[idx], self.pocket_block_ids[idx]
        
        # you may need to modify the complex here in the future

        data = VAEModelInput.from_complex(
            cplx, self.tgt_chains[idx], self.lig_chains[idx], pocket_block_ids, include_raw_data=True
        )
        # peptide position ids start from 1
        pep_position_ids = data.position_ids[data.ligand_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data.position_ids[data.ligand_mask] = pep_position_ids
        return data

    def __len__(self):
        return len(self.cplxs)
    
    def collate_fn(self, batch):
        return VAEModelInput.batchify(batch)
    
    def _process_file(self, file_path, tgt_chains, lig_chains, hotspots):
        '''
        hotspots: in the format of tuples (chain id, seq number, insert code), e.g. [('A', 113, ''), ('A', 114, 'A')]
        '''
        if file_path.endswith('.pdb'):
            cplx = pdb_to_complex(file_path, tgt_chains + ('' if lig_chains is None else lig_chains))
        elif file_path.endswith('.cif'):
            cplx = mmcif_to_complex(file_path, tgt_chains + ('' if lig_chains is None else lig_chains))
        else: raise ValueError(f'File format {file_path} not recognized. Supported: .pdb, .cif')
        if lig_chains is not None:
            frag_lig_chains = []
            for mol in cplx:
                for c in lig_chains:
                    if c + '_' in mol.id: frag_lig_chains.append(mol.id) # small molecules fragmented
            chain_ids_for_pocket_detection = list(lig_chains) + frag_lig_chains
        else:  # hotspots should not be None
            residues = []
            for chain_id, seq_number, icode in hotspots:
                try: residues.append(recur_index(cplx, (chain_id, (int(seq_number), icode))))
                except KeyError: raise KeyError(f'Hotspot {(chain_id, seq_number, icode)} does not exist in the provided protein file')
            new_chain_id = assign_new_chain_id(tgt_chains)
            # add these residues as dummy ligand
            dummy_ligand = Molecule(new_chain_id, [deepcopy(res) for res in residues], id=new_chain_id)
            cplx.id2idx[new_chain_id] = len(cplx.molecules)
            cplx.molecules.append(dummy_ligand)
            chain_ids_for_pocket_detection = [new_chain_id]
            lig_chains = new_chain_id
        
        pairs = compute_interacting_pairs(cplx, chain_ids_for_pocket_detection, tgt_chains, dist_th=self.pocket_dist_th, efficient=True, use_cb=True)   # a large enough threshold to incorporate all residues
        pocket_block_ids = [i for _, i, _, _, _ in pairs]
            
        return cplx, pocket_block_ids, lig_chains
    

if __name__ == '__main__':
    import sys
    dataset = FileDataset([sys.argv[1]], [sys.argv[2]], [sys.argv[3]]) # e.g. xxx.pdb AB CD
    print(dataset[0])