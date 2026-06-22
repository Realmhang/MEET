#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from .parser.mmcif_to_complex import mmcif_to_complex
from .parser.pdb_to_complex import pdb_to_complex
from .utils import is_standard_aa
from .vocab import VOCAB


def load_cplx(path, selected_chains=None, *args, **kwargs):
    if path.endswith('.pdb'): cplx = pdb_to_complex(path, selected_chains, *args, **kwargs)
    elif path.endswith('.cif'): cplx = mmcif_to_complex(path, selected_chains, *args, **kwargs)
    else: raise ValueError(f'File format not supported: {path}')
    return cplx


def extract_seqs(path, selected_chains):
    cplx = load_cplx(path, selected_chains)
    seqs = []
    for chain in cplx:
        seqs.append(''.join([VOCAB.abrv_to_symbol(block.name) for block in chain]))
    return seqs


def extract_CA_coords(path, selected_chains):
    cplx = load_cplx(path, selected_chains)
    coords, masks = [], []
    for chain in cplx:
        chain_coords, chain_masks = [], []
        for block in chain:
            if not is_standard_aa(block.name): continue
            x = None
            for atom in block:
                if atom.name == 'CA': x = atom.get_coord()
            chain_coords.append([0, 0, 0] if x is None else x)
            chain_masks.append(False if x is None else True)
        coords.append(np.array(chain_coords, dtype=np.float64))
        masks.append(np.array(chain_masks, dtype=bool))
    return coords, masks