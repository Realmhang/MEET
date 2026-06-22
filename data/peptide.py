#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

import numpy as np

from utils import register as R
from functools import cache

from .resample import ClusterResampler, IdentityMapping
from .base import BaseDataset
from .file_loader import MolType


@R.register('PeptideDataset')
class PeptideDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            min_pocket_dist: float = 10.0,
            max_pocket_dist: float = 20.0,
            length_type: str = 'atom',
            in_memory: bool = False,
            include_raw_data: bool = False,
            subsample: int = -1,
            use_explicit_length: bool = False,
            length_data: Optional[str] = None,
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index, in_memory, include_raw_data)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.min_pocket_dist = min_pocket_dist
        self.max_pocket_dist = max_pocket_dist
        self.length_type = length_type
        self.subsample = subsample
        if length_data is not None and use_explicit_length:
            self.use_explicit_length = True
            self.length_data = np.load(length_data, mmap_mode='r')
        elif use_explicit_length:
            self.use_explicit_length = True
            self.length_data = np.load(f'{mmap_dir}/lengths.npy', mmap_mode='r')
        else:
            self.use_explicit_length = False

        self.dynamic_idxs = IdentityMapping(len(self))
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    def __len__(self):
        tot = super().__len__()
        if self.subsample > 0 and self.subsample < tot:
            return self.subsample
        return tot

    ########## Start of Overloading ##########

    @property
    def mol_type(self):
        return MolType.PEPTIDE

    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self.load_id(idx)

    def _get_original_length(self, idx):
        if not self.use_explicit_length:
            props = self._properties[idx]
            if self.length_type == 'atom':
                return props['pocket_num_atoms'] + props['ligand_num_atoms']
            elif self.length_type == 'block':
                return props['pocket_num_blocks'] + props['ligand_num_blocks']
            else:
                raise NotImplementedError(f'length type {self.length_type} not recognized')
        else:
            if self.length_type == 'atom':
                return self.length_data[idx, 0]
            elif self.length_type == 'block':
                return self.length_data[idx, 1]
            else:
                raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._get_original_length(idx)
    
    def get_prop(self, idx, prop_name):
        idx = self.dynamic_idxs[idx]
        return super().get_prop(idx, prop_name)

    def _get_pocket_block_ids(self, idx):   # not exposed, thus idx is the real idx
        dist_th = np.random.rand() * (self.max_pocket_dist - self.min_pocket_dist) + self.min_pocket_dist
        pocket_block_ids = []
        props = self._properties[idx]
        for _id, d in zip(props['pocket_block_id'], props['pocket_block_dist']):
            if d < dist_th: pocket_block_ids.append(_id)
        return pocket_block_ids
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        data = super().__getitem__(idx)
        # peptide position ids start from 1
        pep_position_ids = data.position_ids[data.ligand_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data.position_ids[data.ligand_mask] = pep_position_ids
        return data
    

if __name__ == '__main__':
    import sys
    np.random.seed(10)
    dataset = PeptideDataset(sys.argv[1])
    item = dataset[1]
    print(item.S[item.ligand_mask])
    # # print(len(dataset[0].A))
    # print(item.bonds.shape)