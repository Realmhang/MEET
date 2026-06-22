#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import tempfile
from typing import List

import ray
import numpy as np

from data.bioparse.writer.complex_to_pdb import complex_to_pdb
import utils.register as R
from utils.parallel_func import parallel_func

from .base import BaseMetric, MetricResult, EvalTask
from .core.aar import aar
from .core.TMscore import tm_score


@R.register('SeqDiversity')
class SeqDiversity(BaseMetric):
    
    @property
    def calculate_on_group(self):
        return True

    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        return MetricResult.PASSED, input.gen_seq, {}

    def collect(self, inputs: List[EvalTask], vals: List[tuple], num_cpus: int=1):
        seqs = [val[1] for val in vals]

        id2vals = {}
        for task, val in zip(inputs, seqs):
            if task.ref_id not in id2vals: id2vals[task.ref_id] = []
            id2vals[task.ref_id].append(val)

        scores = {}

        for _id in id2vals:
            pairs = []
            for i, vi in enumerate(id2vals[_id]):
                for j, vj in enumerate(id2vals[_id]):
                    if i == j: continue
                    pairs.append((vi, vj))

            div = [(1.0 - score) for score in parallel_func(aar, pairs, num_cpus)]
            div = sum(div) / len(div)
            scores[_id] = div
        diversity = sum(scores.values()) / len(scores)
        return { 'seq_div': diversity }, scores


@R.register('StructDiversity')
class StructDiversity(BaseMetric):
    
    @property
    def calculate_on_group(self):
        return True

    @ray.remote(num_cpus=1)
    def run(self, input: EvalTask):
        # this is a group task, so run is only responsible for preparing data
        gen = input.get_gen_cplx()
        pdb_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
        complex_to_pdb(gen, pdb_file.name, selected_chains=input.ligand_chains_ids)
        return MetricResult.PASSED, pdb_file.name, {}

    def collect(self, inputs: List[EvalTask], vals: List[tuple], num_cpus: int=1):
        
        pdbs = [v[1] for v in vals]
        
        id2vals = {}
        for task, path in zip(inputs, pdbs):
            if task.ref_id not in id2vals: id2vals[task.ref_id] = []
            id2vals[task.ref_id].append(path)

        scores = {}

        for _id in id2vals:
            pairs = []
            for i, vi in enumerate(id2vals[_id]):
                for j, vj in enumerate(id2vals[_id]):
                    if i == j: continue
                    pairs.append((vi, vj))

            div = [(1.0 - score) for score in parallel_func(tm_score, pairs, num_cpus)]
            div = sum(div) / len(div)
            scores[_id] = div
        diversity = sum(scores.values()) / len(scores)
        for path in pdbs: os.remove(path)
        return { 'struct_div': diversity }, scores