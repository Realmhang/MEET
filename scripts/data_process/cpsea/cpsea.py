#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import tempfile
import argparse

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.interface import compute_interacting_pairs
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index
from utils.logger import print_log
from utils.parallel_func import parallel_func


def parse():
    parser = argparse.ArgumentParser(description='Process protein-peptide complexes')
    parser.add_argument('--data_dir', type=str, required=True, help='Database')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_args()


def split_pdb_models(pdb_str):
    lines = pdb_str.split('\n')
    model_idx = 0
    strs = {}
    buffer = []
    for line in lines:
        if line.startswith("MODEL"):
            buffer = [line + '\n']
        elif line.startswith("ENDMDL"):
            buffer.append(line + '\n')
            strs[model_idx] = buffer
            model_idx += 1
        else:
            buffer.append(line + '\n')
    return strs


def worker(pdb_id, summary):
    all_res = []
    rec_chain, lig_chain = summary['rec_chain'], summary['pep_chain']

    # temporary file to store the decompressed mmcif file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pdb')
    pdb_file = temp_file.name

    # decompress the mmcif.gz file
    try:
        with gzip.open(summary['pdb_path'], 'rb') as f_in:
            pdb_str = f_in.read().decode()
    except Exception as e:
        print_log(f'Error decompressing {summary["pdb_path"]}: {e}', level='WARN')
        temp_file.close()
        return []

    models = split_pdb_models(pdb_str)

    for model in models:
        with open(pdb_file, 'w') as fout: fout.writelines(models[model])
        cplx = pdb_to_complex(pdb_file, selected_chains=[rec_chain, lig_chain])

        target_blocks = cplx[rec_chain].blocks
        lig_blocks = cplx[lig_chain].blocks
        target_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in target_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in lig_blocks])
    
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

        all_res.append((pdb_id + f'-model{model}', data, properties))
    return all_res


def process_iterator(items):
    
    generator = parallel_func(worker, [(pdb_id, items[pdb_id]) for pdb_id in items], n_cpus=16)
    cnt = 0
    for outputs in generator:
        cnt += 1
        for _id, data, props in outputs:
            yield _id, data, props, cnt

def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get indexes
    indexes = {}
    for f in os.listdir(args.data_dir):
        if not f.endswith('.gz'): continue
        indexes[f] = {
            'rec_chain': 'R',
            'pep_chain': 'L',
            'pdb_path': os.path.join(args.data_dir, f)
        }
    print_log(f'Found {len(indexes)} files')

    # 3. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(indexes),
        args.out_dir, len(indexes))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())