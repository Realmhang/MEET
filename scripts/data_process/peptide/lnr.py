#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import tempfile
import argparse
from copy import deepcopy

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.interface import compute_interacting_pairs
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index
from utils.logger import print_log
from utils.parallel_func import parallel_func


def parse():
    parser = argparse.ArgumentParser(description='Process protein-peptide complexes')
    parser.add_argument('--index_file', type=str, required=True, help='test.txt')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_args()


def process_iterator(items):
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
        target_blocks = cplx[rec_chain].blocks
        lig_blocks = cplx[lig_chain].blocks
        target_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in target_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in lig_blocks])
        if len(lig_seq) < 4 or len(lig_seq) > 14: continue
    
        # pocket_block_id, _ = compute_pocket(cplx, [rec_chain], [lig_chain], dist_th=pocket_th)
        pairs = compute_interacting_pairs(cplx, [lig_chain], [rec_chain], dist_th=100, efficient=True, use_cb=True)   # a large enough threshold to incorporate all residues
        pocket_block_id, pocket_block_dist = [], []
        for _, i, _, _, d in pairs:
            pocket_block_id.append(i)
            pocket_block_dist.append(d)
        pocket_blocks = [recur_index(cplx, _id) for _id in pocket_block_id]
        # add two dummy blocks for NME/ACE
        dummy_block = deepcopy(lig_blocks[-1])
        dummy_block.id = (dummy_block.id[0] + 1, '')
        lig_blocks.append(dummy_block)
        dummy_block = deepcopy(lig_blocks[-1])
        dummy_block.id = (dummy_block.id[0] + 1, '')
        lig_blocks.append(dummy_block)
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
            'path': os.path.join(root_dir, 'pdbs', line[0] + '.pdb'),
            'rec_chain': line[1],
            'lig_chain': line[2]
        }
    print_log(f'Found {len(items)} files')

    # 3. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(items),
        args.out_dir, len(items))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())