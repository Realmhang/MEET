#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Split peptide data')
    parser.add_argument('--data_dir', type=str, required=True, help='processed directory')
    parser.add_argument('--valid_num', type=float, default=1000)
    parser.add_argument('--test_num', type=float, default=1000)
    parser.add_argument('--train_num', type=float, default=100000)
    return parser.parse_args()


def read_index(csv_file):
    complete_lines = []
    with open(csv_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line: continue
            line = line.split('\t')
            if i % 2 == 1:
                complete_lines[-1].extend(line)
            else: complete_lines.append(line)
    clu2lines = {}
    for line in complete_lines:
        clu = '_'.join(line[0].split('_')[:-1])
        if clu not in clu2lines: clu2lines[clu] = []
        clu2lines[clu].append(line)

    return clu2lines


def main(args):
    clu2lines = {}
    for folder in os.listdir(args.data_dir):
        meta_csv = os.path.join(args.data_dir, folder, folder + '_metadata.csv')
        if not os.path.exists(meta_csv): continue
        clu2lines.update(read_index(meta_csv))
    
    clus = list(clu2lines.keys())
    np.random.shuffle(clus)

    out_dir = os.path.join(args.data_dir, 'splits')
    os.makedirs(out_dir, exist_ok=True)

    split = {
        'train': clus[:args.train_num],
        'valid': clus[args.train_num:args.train_num + args.valid_num],
        'test': clus[args.train_num + args.valid_num:args.train_num + args.valid_num + args.test_num],
    }
    for split_name in split:
        clus = split[split_name]
        lines = []
        for clu in clus:
            i = np.random.randint(0, len(clu2lines[clu]))
            lines.append(clu2lines[clu][i])
        with open(os.path.join(out_dir, f'{split_name}.csv'), 'w') as fout:
            for line in lines:
                fout.write('\t'.join(line) + '\n')
        print_log(f'{len(lines)} entries, {len(clus)} clusters to {split_name}')


if __name__ == '__main__':
    np.random.seed(2025)
    main(parse())
