#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import tempfile
import argparse

from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure, CIFBlock
from io import StringIO

from data.mmap_dataset import create_mmap
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.parser._biotite_to_complex import atom_array_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.interface import compute_interacting_pairs
from data.bioparse.hierarchy import merge_cplx
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index, BONDCAT
from utils.logger import print_log
from utils.parallel_func import parallel_func


def parse():
    parser = argparse.ArgumentParser(description='Process protein-peptide complexes')
    parser.add_argument('--data_dir', type=str, required=True, help='Database')
    parser.add_argument('--csv_path', type=str, required=True, help='metadata file path')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    parser.add_argument('--save_cif', action='store_true', help='Save cif file of the complexes')
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


def worker(_id, summary, cif_dir):
    # print(_id, summary)
    all_res = []
    rec_chain, lig_chain = summary['rec_chain'], summary['pep_chain']

    # temporary file to store the decompressed mmcif file
    #temp_file = tempfile.NamedTemporaryFile(suffix='.cif')
    #cif_file = temp_file.name

    # decompress the mmcif.gz file
    try:
        with gzip.open(summary['cif_path'], 'rb') as f_in:
            file = CIFFile.read(StringIO(f_in.read().decode('utf-8')))
            # with open(cif_file, 'wb') as f_out:
            #     f_out.write(f_in.read())
            # cif_str = f_in.read().decode()
    except Exception as e:
        print_log(f'Error decompressing {summary["pdb_path"]}: {e}', level='WARN')
        temp_file.close()
        return []

    # models = split_pdb_models(pdb_str)
    # file = CIFFile.read(cif_file)
    # print(list(file.keys()))

    for model_i, model_name in enumerate(file.keys()):
        if (model_i != 0) and (model_name != _id): continue
        block = CIFBlock.deserialize(file._blocks[model_name])
        if model_i != 0:
            block._categories['chem_comp_bond'] = BONDCAT
        model = get_structure(block, include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])
        cplx = atom_array_to_complex(model, _id, selected_chains=[rec_chain, lig_chain])
        if model_i == 0: tgt = cplx
        elif model_name == _id: lig = cplx
    
    cplx = merge_cplx(tgt, lig, name=_id)
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

    if cif_dir is not None:
        out_cif_path = os.path.join(cif_dir, f'{_id}.cif')
        complex_to_mmcif(cplx, out_cif_path)

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

    all_res.append((_id, data, properties))
    return all_res


def process_iterator(items, cif_dir):
    
    # for _id in items: worker(_id, items[_id], cif_dir)
    unordered = True if len(items) > 1000000 else False # avoid MEM accumulation
    cnt = 0
    while len(items):
        batch_max = 1000000
        inputs, del_ids = [], []
        for _id in items:
            inputs.append((_id, items[_id], cif_dir))
            del_ids.append(_id)
            batch_max -= 1
            if batch_max == 0: break

        generator = parallel_func(worker, [(_id, items[_id], cif_dir) for _id in items], n_cpus=8, unordered=unordered)
        # delete the ids
        for _id in del_ids: items.pop(_id)
        for outputs in generator:
            cnt += 1
            for _id, data, props in outputs:
                yield _id, data, props, cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get indexes
    indexes = {}
    with open(args.csv_path) as fin:
        lines = fin.readlines()
    for line in lines:
        _id = line.strip().split('\t')[0]
        if not _id.startswith('AF'): continue   # two line mistake
        cluster = '_'.join(_id.split('_')[:-1])
        for folder in os.listdir(args.data_dir):
            cif_path = os.path.join(args.data_dir, folder, f'Clusters_Found_AFDB_{folder[-1]}', cluster + '.cif.gz')
            if not os.path.exists(cif_path): continue
            indexes[_id] = {
                'rec_chain': 'R',
                'pep_chain': 'L',
                'cif_path': cif_path
            }
            break
    print_log(f'Found {len(indexes)} files')

    # 3. process pdb files into our format (mmap)
    cif_dir = os.path.join(args.out_dir, 'mmcif') if args.save_cif else None
    if cif_dir is not None: os.makedirs(cif_dir, exist_ok=True)
    create_mmap(
        process_iterator(indexes, cif_dir),
        args.out_dir, len(indexes))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())