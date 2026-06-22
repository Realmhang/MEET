#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import tempfile
import argparse
from copy import deepcopy

from data.mmap_dataset import create_mmap
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.hierarchy import Bond, BondType
from data.bioparse.interface import compute_interacting_pairs
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index, is_aa
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process protein-peptide complexes')
    parser.add_argument('--index_file', type=str, required=True, help='test.txt')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_args()


def process_iterator(items, mmcif_out_dir):
    os.makedirs(mmcif_out_dir, exist_ok=True)
    cnt = 0

    for _id in items:
        cnt += 1
        item = items[_id]
        pdb_file, rec_chain, lig_chain = item['path'], item['rec_chain'], item['lig_chain']
        try:
            cplx = pdb_to_complex(pdb_file, selected_chains=[rec_chain, lig_chain], remove_het=True)
        except Exception as e:
            print_log(f'Parsing {pdb_file} failed: {e}')
            continue
        tmp = os.path.join(mmcif_out_dir, 'tmp.pdb')
        cplx[rec_chain].blocks = [b for b in cplx[rec_chain] if ('original_name' not in b.properties) or (b.properties['original_name'] not in ['NMA', 'ACE']) ]
        lig_blocks = cplx[lig_chain].blocks
        cplx.bonds.append(Bond(
            (cplx.id2idx[lig_chain], len(lig_blocks) - 2, 0),
            (cplx.id2idx[lig_chain], len(lig_blocks) - 1, 0),
            BondType.SINGLE
        ))
        ace_c = None
        next_aa_n = None
        for i, atom in enumerate(lig_blocks[0]):
            if atom.name == 'C': ace_c = i
        for i, atom in enumerate(lig_blocks[1]):
            if atom.name == 'N': next_aa_n = i
        cplx.bonds.append(Bond(
            (cplx.id2idx[lig_chain], 0, ace_c),
            (cplx.id2idx[lig_chain], 1, next_aa_n),
            BondType.SINGLE
        ))
        interblock_bonds = []
        for bond in cplx.bonds:
            if bond.index1[0] == cplx.id2idx[lig_chain] and bond.index2[0] == cplx.id2idx[lig_chain]:
                if bond.index1[1] != bond.index2[1]:
                    interblock_bonds.append((bond.index1, bond.index2, bond.bond_type))
        for block in lig_blocks:
            if ('original_name' in block.properties) and (block.properties['original_name'] == 'NMA'):
                block.properties['original_name'] = 'NME'
                block.name = 'NME'
                block.id = (block.id[0] + 1, block.id[1][1:])
        complex_to_pdb(cplx, tmp, selected_chains=[rec_chain, lig_chain], explict_bonds=interblock_bonds)
        cplx = pdb_to_complex(tmp, selected_chains=[rec_chain, lig_chain])
        complex_to_mmcif(cplx, os.path.join(mmcif_out_dir, _id + '.cif'), selected_chains=[rec_chain, lig_chain])
        
        target_blocks = cplx[rec_chain].blocks
        lig_blocks = cplx[lig_chain].blocks
        target_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in target_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in lig_blocks])
        if len(lig_blocks) < 6 or len(lig_blocks) > 16: continue
    
        # pocket_block_id, _ = compute_pocket(cplx, [rec_chain], [lig_chain], dist_th=pocket_th)
        pairs = compute_interacting_pairs(cplx, [lig_chain], [rec_chain], dist_th=100, efficient=True, use_cb=True)   # a large enough threshold to incorporate all residues
        pocket_block_id, pocket_block_dist = [], []
        for _, i, _, _, d in pairs:
            pocket_block_id.append(i)
            pocket_block_dist.append(d)
        pocket_blocks = [recur_index(cplx, _id) for _id in pocket_block_id]
        data = cplx.to_tuple()

        properties = {
            'pocket_num_blocks': len(pocket_block_id),
            'ligand_num_blocks': len(lig_blocks),
            'pocket_num_atoms': sum([len(block) for block in pocket_blocks]),
            'ligand_num_atoms': sum([len(block) for block in lig_blocks]),
            'target_chain_ids': [rec_chain],
            'ligand_chain_ids': [lig_chain],
            'target_sequences': [target_seq],
            'ligand_sequences': [lig_seq],
            'pocket_block_id': pocket_block_id,
            'pocket_block_dist': pocket_block_dist
        }

        yield _id, data, properties, cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get indexes
    items = {}
    root_dir = os.path.dirname(args.index_file)
    with open(args.index_file, 'r') as fin:
        lines = fin.readlines()
    for line in lines:
        line = line.strip().split('\t')
        items[line[0]] = {
            # 'path': os.path.join(root_dir, 'pdbs', line[0] + '.pdb'),
            'path': os.path.join(root_dir, 'capped', line[0].upper() + '_-_prepared.pdb'),
            'rec_chain': line[1],
            'lig_chain': line[2]
        }
    print_log(f'Found {len(items)} files')

    # 3. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(items, os.path.join(args.out_dir, 'mmcif')),
        args.out_dir, len(items))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())