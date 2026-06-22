#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Split peptide data')
    parser.add_argument('--processed_dir', type=str, required=True, help='processed directory')
    parser.add_argument('--valid_ratio', type=float, default=0.01)
    parser.add_argument('--test_ratio', type=float, default=0.01)
    return parser.parse_args()


def read_index(mmap_dir):
    items = {}
    index = os.path.join(mmap_dir, 'index.txt')
    with open(index, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            values = line.strip().split('\t')
            items[values[0]] = line
    return items


def main(args):
    items = read_index(args.processed_dir)
    id2cluster, cluster2ids = {}, {}
    for name in items:
        clu = name.split('.')[0]
        id2cluster[name] = clu
        if clu not in cluster2ids: cluster2ids[clu] = []
        cluster2ids[clu].append(name)
    # get val and test clusters
    clus = list(cluster2ids.keys())
    np.random.shuffle(clus)
    valid_size, test_size = int(len(clus) * args.valid_ratio), int(len(clus) * args.test_ratio)

    split = {
        'train': clus[test_size + valid_size:],
        'valid': clus[test_size:test_size + valid_size],
        'test': clus[:test_size]
    }
    for split_name in split:
        clus = split[split_name]
        index_out_file = open(os.path.join(args.processed_dir, f'{split_name}.txt'), 'w')
        cluster_out_file = open(os.path.join(args.processed_dir, f'{split_name}.cluster'), 'w')
        cnt = 0
        for c in clus:
            ids = cluster2ids[c]
            if split_name != 'train': ids = np.random.choice(ids, 1)
            for _id in ids:
                index_out_file.write(items[_id])
                cluster_out_file.write(f'{_id}\t{c}\t{len(ids)}\n')
                cnt += 1
        index_out_file.close()
        cluster_out_file.close()
        print_log(f'{cnt} entries, {len(clus)} clusters to {split_name}')


if __name__ == '__main__':
    np.random.seed(10)
    main(parse())
