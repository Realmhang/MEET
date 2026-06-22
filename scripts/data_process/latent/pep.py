#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import gzip
import tempfile
import argparse

import torch

from generate import to_device, prepare
from data import create_dataloader, create_dataset

from data.mmap_dataset import create_mmap
from utils.logger import print_log
from utils.parallel_func import parallel_func


def parse():
    parser = argparse.ArgumentParser(description='Encoding protein-peptide complexes into latent states')
    parser.add_argument('--config', type=str, required=True, help='Generation config')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_known_args()


def process_iterator(model, test_set, test_loader, device):
    
    cnt = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = to_device(batch, device)
            Zh, Zv, Zx, Zx_mu, signed_Zx_log_var, _, _, _, bind_site_repr, ligand_repr = model.encode(
                inputs.X, inputs.S, inputs.A, inputs.bonds, inputs.chain_ids, inputs.ligand_mask, inputs.block_lengths, inputs.lengths,
                deterministic=True
            ) # [Nblock, d_latent], [Nblock, 3], [1], [1]
            offset = 0
            for l in inputs.lengths:
                h, v, x = Zh[offset:offset+l], Zv[offset:offset+l], Zx[offset:offset+l]
                mask = inputs.ligand_mask[offset:offset+l]
                pos_ids = inputs.position_ids[offset:offset+l]
                offset += l
                data = {
                    'H': h.tolist(),
                    'V': v.tolist(),
                    'X': x.tolist(),
                    'ligand_mask': mask.tolist(),
                    'position_ids': pos_ids.tolist()
                } 
                props = { 'length': l.item() }
                _id = test_set.get_id(cnt)
                cnt += 1
                yield _id, data, props, cnt


def main(args, opt_args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get dataset
    args.save_dir = ''  # dummy
    config, model, test_set, _, device = prepare(args, opt_args)
    print_log(f'Loaded data and model')
    
    test_loader = create_dataloader(test_set, config['dataloader'])

    # 3. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(model, test_set, test_loader, device),
        args.out_dir, len(test_set))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(*parse())