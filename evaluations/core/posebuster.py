#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from pathlib import Path

import yaml
from posebusters import PoseBusters


CONFIG = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), 'posebuster_pep.yaml'
), 'r'))


def denovo_validity(pocket_file, mol_sdf):
    pred_file = Path(mol_sdf)
    true_file = Path(mol_sdf)
    cond_file = Path(pocket_file)

    buster = PoseBusters(config=CONFIG, max_workers=0)
    df = buster.bust([pred_file], true_file, cond_file)
    data = df.to_dict(orient='index') # {row1: {col1: val1, col2: val2, ...}, row2: {...}}
    is_valid = []
    for key in data:
        valid = True
        for check_type in data[key]: valid = valid and data[key][check_type]
        is_valid.append(valid)
    return is_valid, data
