#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .bioparse.hierarchy import Complex
from .data_struct import VAEModelInput
from .mmap_dataset import MMAPDataset, HierMMAPDataset

'''
Base class
'''


class BaseDataset(HierMMAPDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            in_memory: bool = False,
            include_raw_data: bool = False
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index, in_memory)
        self.mmap_dir = mmap_dir
        self.include_raw_data = include_raw_data

    ########## Start of Overloading ##########

    def get_id(self, idx: int):
        raise NotImplementedError(f'get_id(self, idx) not implemented for {self}')

    def get_len(self, idx: int):
        raise NotImplementedError(f'get_len(self, idx) not implemented for {self}')
    
    def get_prop(self, idx, prop_name):
        props = self._properties[idx]
        return props[prop_name]

    def _get_pocket_block_ids(self, idx: int):
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in self._properties[idx]['pocket_block_id']]
        return pocket_block_ids

    @property
    def mol_type(self):
        raise NotImplementedError(f'mol_type not implemented')
    
    ########## End of Overloading ##########

    def get_raw_data(self, idx: int):
        cplx = Complex.from_tuple(super().__getitem__(idx))
        return cplx
    
    def __getitem__(self, idx: int):
        '''
        an example of the returned data
        {
            'X': [Natom, 3],
            'S': [Nblock],
            'A': [Natom],
            'bonds': [Nbond, 3]
            'position_ids': [Nblock],
            'chain_ids': [Nblock], used to distinguish different chains
            'generate_mask': [Nblock], 0 for context, 1 for generation
            'center_mask': [Nblock], 1 for used to centering the complex (e.g. pocket)
            'block_lengths': [Nblock],
            'is_aa': [Nblock]
            'lengths': [1]
        }
        '''
        cplx, props = self.get_raw_data(idx), self._properties[idx]
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in self._get_pocket_block_ids(idx)]
        return VAEModelInput.from_complex(
            cplx, props['target_chain_ids'], props['ligand_chain_ids'], pocket_block_ids, include_raw_data=self.include_raw_data
        )

    def collate_fn(self, batch):
        return VAEModelInput.batchify(batch)