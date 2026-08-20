#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import json
import argparse
from copy import deepcopy

from tqdm import tqdm
import torch

import models
from data import create_dataset
from data.bioparse import Complex, VOCAB
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from models.AtomVAE.configs import VAESampleConfig
from models.LatentGenHX.configs import LatentGeneratorSampleConfig
from utils.logger import print_log
from utils.random_seed import setup_seed
from utils.config_utils import overwrite_values
from utils.checkpoint import load_model

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


def _get_by_path(config, path):
    value = config
    for key in path.split('.'):
        value = value[key]
    return value


def _resolve_refs(config, root=None):
    if root is None:
        root = config
    if isinstance(config, dict):
        return {key: _resolve_refs(value, root) for key, value in config.items()}
    if isinstance(config, list):
        return [_resolve_refs(value, root) for value in config]
    if isinstance(config, str) and config.startswith('${') and config.endswith('}'):
        return _get_by_path(root, config[2:-1])
    return config


def load_config(path):
    with open(path, 'r') as fin:
        text = fin.read()
    if OmegaConf is not None:
        config = OmegaConf.create(text)
        return OmegaConf.to_container(config, resolve=True)
    return _resolve_refs(yaml.safe_load(text))


class Recorder:
    def __init__(self, test_set, n_samples, save_dir):
        self.pbar = tqdm(total=n_samples * len(test_set))
        self.waiting_list = [(i, n) for n in range(n_samples) for i in range(len(test_set))]
        self.num_generated, self.num_failed = 0, 0
        self.fout = open(os.path.join(save_dir, 'results.jsonl'), 'w')

    def is_finished(self):
        return len(self.waiting_list) == 0

    def get_next_batch_list(self, batch_size):
        batch_list = self.waiting_list[:batch_size]
        self.waiting_list = self.waiting_list[batch_size:]
        return batch_list

    def check_and_save(self, log, item_idx, n, struct_only=False):
        self.num_generated += 1
        if log is None:
            self.num_failed += 1
            self.waiting_list.append((item_idx, n))
        else:
            log.update({
                'n': n,
            })
            self.fout.write(json.dumps(log) + '\n')
            self.fout.flush()
            self.pbar.update(1)

    def __del__(self):
        self.fout.close()


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def generate_wrapper(model, sample_opt={}):
    mixed_precision = sample_opt.get('enable_mixed_precision', False)
    sample_opt = {k: v for k, v in sample_opt.items() if k != 'enable_mixed_precision'}
    if isinstance(model, models.AtomVAEBond):
        def wrapper(batch):
            if mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    ret = model.generate(batch, VAESampleConfig(**sample_opt))
            else:
                ret = model.generate(batch, VAESampleConfig(**sample_opt))
            return ret
    elif isinstance(model, models.LatentGeneratorHX):
        def wrapper(batch):
            if mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    ret = model.generate(batch, LatentGeneratorSampleConfig(**sample_opt))
            else:
                ret = model.generate(batch, LatentGeneratorSampleConfig(**sample_opt))
            return ret
    else:
        raise NotImplementedError(f'Wrapper for {type(model)} not implemented')
    return wrapper


def prepare(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    mode = config.get('sample_opt', {}).get('mode', 'codesign')
    struct_only = mode == 'fixseq'
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
    print(f'Using checkpoint {b_ckpt}')
    # Construct a fresh model from the training config and load the checkpoint
    # state_dict into it. This avoids relying on the pickled model object, which
    # can break on forward when the class definition gained new attributes
    # since the checkpoint was saved (e.g. older FinalLayer without
    # use_final_linear).
    model_cfg = load_config(args.model_config)['model']
    print(f'Constructing model from config {args.model_config} and loading state_dict')
    model = load_model(b_ckpt, model_config=model_cfg)
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    
    # save path
    if args.save_dir is None:
        save_dir = os.path.join(ckpt_dir, 'results')
    else:
        save_dir = args.save_dir

    return config, model, test_set, save_dir, device


def main(args, opt_args):
    config, model, test_set, save_dir, device = prepare(args, opt_args)
    
    ref_save_dir = os.path.join(save_dir, 'references')
    os.makedirs(ref_save_dir, exist_ok=True)
    # cand_save_dir = os.path.join(save_dir, 'candidates')
    # tmp_cand_save_dir = os.path.join(save_dir, 'tmp_candidates')
    # for directory in [ref_save_dir, cand_save_dir, tmp_cand_save_dir]:
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    n_samples = config.get('n_samples', 1)
    n_cycles = config.get('n_cycles', 0)
    batch_size = config['dataloader']['batch_size']
    pocket_dist_defs = config.get('pocket_dist_defs', [None])
    
    for ith_def, pocket_dist in enumerate(pocket_dist_defs):
        print_log(f'Pocket definition: {pocket_dist}')

        # change the pocket distance definition
        if pocket_dist is not None:
            test_set.min_pocket_dist = pocket_dist
            test_set.max_pocket_dist = pocket_dist + 1e-6

        # prepare
        cand_save_dir = os.path.join(save_dir, f'pocket_dist_{pocket_dist}', 'candidates')
        os.makedirs(cand_save_dir, exist_ok=True)
        recorder = Recorder(test_set, n_samples, os.path.join(save_dir, f'pocket_dist_{pocket_dist}'))

        # start generation
        while not recorder.is_finished():
            batch_list = recorder.get_next_batch_list(batch_size)
            batch = [test_set[i] for i, _ in batch_list]
            batch = test_set.collate_fn(batch)
            batch = to_device(batch, device)

            with torch.no_grad():
                res = generate_wrapper(model, deepcopy(config.get('sample_opt', {})))(batch)
                res_list = res.split_batch()

            for item, (item_idx, n) in zip(res_list, batch_list):
                cplx: Complex = item.to_complex()
                _id = test_set.get_id(item_idx).replace('.pdb', '').replace('.gz', '')
                item_save_dir = os.path.join(cand_save_dir, _id)
                if n == 0: # the first candidate of this complex, we should save the reference structure
                    if ith_def == 0:    # save the candidate
                        ref_save_path = os.path.join(ref_save_dir, f'{_id}.cif')
                        complex_to_mmcif(item.ref_cplx, ref_save_path)
                    os.makedirs(item_save_dir, exist_ok=True)
                complex_to_mmcif(cplx, os.path.join(item_save_dir, f'{n}.cif'))
                seq = ''
                for idx in item.S[item.block_type_generate_mask]:
                    symbol = VOCAB.idx_to_symbol(idx.item())
                    if symbol.startswith('f'): continue
                    seq += symbol
                recorder.check_and_save({
                    'id': _id,
                    'gen_seq': seq,
                    'target_chain_ids': test_set.get_prop(item_idx, 'target_chain_ids'),
                    'ligand_chain_ids': test_set.get_prop(item_idx, 'ligand_chain_ids'),
                    'model_block_idx': item.model_indexes
                }, item_idx, n)

        # vae_batch_list = []
        # for S, X, A, ll, bonds, intra_bonds, (item_idx, n) in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list):
        #     cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
        #     summary: Summary = deepcopy(test_set.get_summary(item_idx))
        #     # revise id
        #     format_id(summary)
        #     summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
        #     if n == 0: # the first round
        #         os.makedirs(os.path.dirname(summary.ref_pdb), exist_ok=True)
        #         complex_to_pdb(cplx, summary.ref_pdb, summary.target_chain_ids + summary.ligand_chain_ids)
        #         os.makedirs(os.path.join(cand_save_dir, summary.id), exist_ok=True)
        #         os.makedirs(os.path.join(tmp_cand_save_dir, summary.id), exist_ok=True)
        #         complex_to_pdb(cplx, os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'), summary.target_chain_ids)
        #     if n_cycles == 0: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
        #     else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}.pdb')
        #     log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, check_validity=False)
        #     if n_cycles == 0: recorder.check_and_save(log, item_idx, n, struct_only)
        #     else:
        #         vae_batch_list.append(
        #             _get_item(
        #                 os.path.join(tmp_cand_save_dir, summary.id, f'pocket.pdb'),
        #                 save_path.rstrip('.pdb') + '.sdf',
        #                 summary.target_chain_ids
        #             )
        #         )

        # for cyc_i in range(n_cycles):
        #     print_log(f'Cycle: {cyc_i}', level='DEBUG')
        #     final_cycle = cyc_i == n_cycles - 1
        #     batch = test_set.collate_fn(vae_batch_list)
        #     batch = to_device(batch, device)
        #     vae_batch_list = []
        #     model_autoencoder = getattr(model, 'autoencoder', model)
        #     with torch.no_grad():
        #         if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
        #         batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(model_autoencoder, deepcopy(config.get('sample_opt', {})))(batch)
        #     for S, X, A, ll, bonds, intra_bonds, (item_idx, n) in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list):
        #         cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
        #         summary: Summary = deepcopy(test_set.get_summary(item_idx))
        #         # revise id
        #         format_id(summary)
        #         summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
        #         if final_cycle: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
        #         else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}_cyc{cyc_i}.pdb')
        #         # get expect atom number
        #         if hasattr(test_set, 'get_expected_atom_num'):
        #             expect_atom_num = test_set.get_expected_atom_num(item_idx)
        #         else: expect_atom_num = None
        #         log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, check_validity=final_cycle, expect_atom_num=expect_atom_num)
        #         if final_cycle: recorder.check_and_save(log, item_idx, n, struct_only)
        #         else:
        #             vae_batch_list.append(
        #                 _get_item(
        #                     os.path.join(tmp_cand_save_dir, summary.id, f'pocket.pdb'),
        #                     save_path.rstrip('.pdb') + '.sdf',
        #                     summary.target_chain_ids
        #                 )
        #             )

        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    return    

def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to a training config used to construct the model; '
                             'the checkpoint state_dict is then loaded into the fresh model.')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)
