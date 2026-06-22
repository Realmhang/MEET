#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

from evaluations.base import EvalTask, EvalRunner
from utils.random_seed import setup_seed
from utils.logger import print_log


def eval_one_dir(generate_dir, eval_config, num_cpus):
    ref_dir = os.path.join(generate_dir, '..', 'references')
    with open(os.path.join(generate_dir, 'results.jsonl'), 'r') as fin:
        lines = fin.read().strip().split('\n')
    tasks = []
    for line in lines:
        item = json.loads(line)
        tasks.append(EvalTask(
            id=f'{item["id"]}-sample{item["n"]}',
            ref_id=item['id'],
            number=item['n'],
            gen_cif_path=os.path.join(generate_dir, 'candidates', item['id'], f'{item["n"]}.cif'),
            ref_cif_path=os.path.join(ref_dir, f'{item["id"]}.cif'),
            target_chains_ids=item['target_chain_ids'],
            ligand_chains_ids=item['ligand_chain_ids'],
            model_block_idx=item['model_block_idx'],
            gen_seq=item['gen_seq']
        ))
    log_file = os.path.join(generate_dir, 'eval.log')
    runner = EvalRunner(eval_config, log_file, num_cpus)
    runner.launch(tasks)
    while not runner.check_finished():
        time.sleep(1)
    # finished
    runner.print()
    runner.save(os.path.join(generate_dir, 'eval_results.json'))


def main(args):

    print_log(f'Number of CPUs to use: {args.num_workers}')
    for subdir in os.listdir(args.root_dir):
        if subdir == 'references': continue
        generate_dir = os.path.join(args.root_dir, subdir)
        if not os.path.isdir(generate_dir): continue
        if not os.path.exists(os.path.join(generate_dir, 'results.jsonl')): continue
        print_log(f'Evaluating results from {generate_dir}...')
        eval_one_dir(generate_dir, args.eval_config, args.num_workers)


def parse():
    parser = argparse.ArgumentParser(description='calculate metrics')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the experiment')
    parser.add_argument('--eval_config', type=str, required=True, help='Evaluation config file')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use')

    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(0)
    main(parse())
