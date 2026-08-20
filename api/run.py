#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from copy import deepcopy

import yaml
import torch

from generate import Recorder, to_device, generate_wrapper, load_config as load_model_config
from data.bioparse import VOCAB
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.random_seed import setup_seed
from utils.checkpoint import load_model
import utils.register as R

from .dataset import *


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to a training config used to construct the model; '
                             'the checkpoint state_dict is then loaded into the fresh model.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpus', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    
    parser.add_argument('--seed', type=int, default=None, help='seed')
    return parser.parse_known_args()


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    model_cfg = load_model_config(args.model_config)['model']
    model = load_model(args.ckpt, model_config=model_cfg)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load dataset and dataloader
    test_set = R.construct(config['dataset'])
    batch_size = config.get('batch_size', 32)
    os.makedirs(args.save_dir, exist_ok=True)
    recorder = Recorder(test_set, config['n_samples'], args.save_dir)

    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = test_set.collate_fn([test_set[i] for i, _ in batch_list])
        batch = to_device(batch, device)

        with torch.no_grad():
            res = generate_wrapper(model, deepcopy(config.get('sample_opt', {})))(batch)
            res_list = res.split_batch()

        for item, (item_idx, n) in zip(res_list, batch_list):
            cplx = item.to_complex()
            _id = test_set.get_id(item_idx)
            item_save_dir = os.path.join(args.save_dir, _id)
            os.makedirs(item_save_dir, exist_ok=True)
            complex_to_mmcif(cplx, os.path.join(item_save_dir, f'{n}.cif'))
            recorder.check_and_save({
                'id': _id,
                'gen_seq': ''.join([VOCAB.idx_to_symbol(idx.item())for idx in item.S[item.block_type_generate_mask]]),
                'target_chain_ids': test_set.tgt_chains[item_idx],
                'ligand_chain_ids': test_set.lig_chains[item_idx],
                'model_block_idx': item.model_indexes
            }, item_idx, n)


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    if args.seed is not None: setup_seed(12)
    main(args, opt_args)