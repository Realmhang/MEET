#!/usr/bin/python
# -*- coding:utf-8 -*-
from enum import Enum
from typing import List
from dataclasses import dataclass
from tqdm import tqdm

import ray
import json
import yaml
import numpy as np

from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
import utils.register as R
from utils.logger import print_log


class MetricResult(Enum):
    PASSED = 1
    FAILED = 2
    ERROR = 3


@ray.remote
class SharedDict:
    def __init__(self):
        self.data = {}

    def put(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key, None)

    def get_all(self):
        return self.data


@dataclass
class EvalTask:
    id: str
    ref_id: str
    number: int
    gen_cif_path: str
    ref_cif_path: str
    target_chains_ids: List[str]
    ligand_chains_ids: List[str]
    model_block_idx: List[tuple]
    gen_seq: str
    
    metric_results: List = None
    _shared_dict: SharedDict = None # manipulated by the EvalRunner

    # pmetric: float
    # is_antibody: bool
    # ref_metrics: dict = None
    # metrics: dict = None
    # seq: str = None
    # ca_coords: List[tuple] = None

    @ray.remote
    def _get_ref_cplx_async(self):
        cplx = ray.get(self._shared_dict.get.remote(f'{self.ref_id}_ref_cplx'))
        if cplx is None:
            try: cplx = mmcif_to_complex(self.ref_cif_path, selected_chains=self.target_chains_ids + self.ligand_chains_ids)
            except Exception as e: cplx = None
            self._shared_dict.put.remote(f'{self.ref_id}_ref_cplx', cplx)
        return cplx

    @ray.remote
    def _get_gen_cplx_async(self):
        cplx = ray.get(self._shared_dict.get.remote(str(self.id)))
        if cplx is None:
            try: cplx = mmcif_to_complex(self.gen_cif_path, selected_chains=self.target_chains_ids + self.ligand_chains_ids)
            except Exception as e: cplx = None
            self._shared_dict.put.remote(str(self.id), cplx)
        return cplx

    def get_ref_cplx(self):
        return ray.get(self._get_ref_cplx_async.remote(self))

    def get_gen_cplx(self):
        return ray.get(self._get_gen_cplx_async.remote(self))


class BaseMetric:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def calculate_on_group(self):
        return False    # calculate on single (e.g. RMSD) or group (e.g. distribution KL/JSD)

    @ray.remote
    def run(self, input: EvalTask):
        # Result (enum), score, Details
        return MetricResult.PASSED, 0, {}
    
    def print_error(self, input, err_msg):
        print_log(f'{self.name}, error occurred for input {input}: {err_msg}', level='ERROR')
    
    def print(self, tasks: List[EvalTask], vals: List[tuple], log_file=None):
        pass_cnt, fail_cnt, error_cnt = 0, 0, 0
        scores, idxs = [], []
        for i, (s, v, _) in enumerate(vals):
            if s == MetricResult.PASSED:
                pass_cnt += 1
                scores.append(v)
                idxs.append(i)
            elif s == MetricResult.FAILED: fail_cnt += 1
            elif s == MetricResult.ERROR: error_cnt += 1
        text = f'{self.name}: Passed/Failed/Error({pass_cnt}/{fail_cnt}/{error_cnt})\n'
        if len(scores) > 0:
            max_score_task = tasks[max(zip(scores, idxs), key=lambda x: x[0])[1]]
            min_score_task = tasks[min(zip(scores, idxs), key=lambda x: x[0])[1]]
            text += f'\tmean {sum(scores)/len(scores):.4f}, std {np.std(scores):.4f}, median {np.median(scores):.4f}\n' + \
                    f'\tmax {max(scores):.4f}, id {max_score_task.id}, min {min(scores):.4f} id {min_score_task.id}\n'
        for sentence in text.split('\n'): print_log(sentence)
        if log_file is not None: log_file.write(text)

    def print_group(self, vals: tuple, log_file=None):
        scores, _ = vals
        text = f'{self.name}: \n'
        if isinstance(scores, dict):
            for k in scores:
                text += f'\t{k}: {scores[k]:.4f}\n'
        else: text += f'\t{scores:.4f}'
        for sentence in text.split('\n'): print_log(sentence)
        if log_file is not None: log_file.write(text)


@ray.remote
def run_metrics(task: EvalTask, metrics: List[BaseMetric]):
    metric_results = []
    for m in metrics:
        res = m.run.remote(m, task)
        res = ray.get(res)
        metric_results.append(res)
    task.metric_results = metric_results
    return task


class EvalRunner:
    def __init__(self, eval_config: str, log_file=None, num_cpus=None, num_gpus=None, verbose=True):
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        self.shared_dict = SharedDict.remote()
        self.num_cpus = num_cpus
        config = yaml.safe_load(open(eval_config, 'r'))
        self.metrics = [R.construct(c) for c in config]
        self.log_file = log_file
        self.verbose = verbose
        if verbose: print_log(f'Initialized EvalRunner with {len(self.metrics)} metrics: {[m.name for m in self.metrics]}')

        # recording
        self.futures = []
        self.future_results = []
        self.group_metrics = {}
        self.progress_bar = None

    def launch(self, tasks: List[EvalTask]):
        if self.verbose: self.progress_bar = tqdm(tasks, ascii=True)
        for task in tasks:
            task._shared_dict = self.shared_dict
            self.futures.append(run_metrics.remote(task, self.metrics))
    
    def check_finished(self):
        '''
        Return the list of folders whose tasks have been finished
        '''
        done, pending = ray.wait(self.futures, num_returns=len(self.futures), timeout=1.0)
        self.futures = pending
        for done_id in done:
            self.future_results.append(ray.get(done_id))
        if self.progress_bar is not None:
            self.progress_bar.update(len(done))
        if len(pending) != 0: return False
        # calculate group metrics
        for i, metric in enumerate(self.metrics):
            if not metric.calculate_on_group: continue
            if metric.name in self.group_metrics: continue
            if self.verbose: print_log(f'Calculating group metric {metric.name} ...')
            scores, details = metric.collect(self.future_results, [task.metric_results[i] for task in self.future_results], num_cpus=self.num_cpus)
            self.group_metrics[metric.name] = (scores, details)
        return True

    # def get_results(self):
    #     if not self.check_finished():
    #         print_log(f'Warning: not all tasks are finished yet! {len(self.futures)} remaining', level='WARN')
    #     return self.future_results

    def print(self):
        if not self.check_finished():
            print_log(f'Warning: not all tasks are finished yet! {len(self.futures)} remaining', level='WARN')
            return
        if self.log_file is not None:
            log_file = open(self.log_file, 'w')
        else: log_file = None
        for i, metric in enumerate(self.metrics):
            if metric.calculate_on_group:
                metric.print_group(self.group_metrics[metric.name], log_file)
            else:
                metric.print(self.future_results, [task.metric_results[i] for task in self.future_results], log_file)
        if log_file is not None:
            if self.verbose: print_log(f'Log file saved to {self.log_file}')
            log_file.close()

    def save(self, path):
        if not self.check_finished():
            print_log(f'Warning: not all tasks are finished yet! {len(self.futures)} remaining', level='WARN')
            return
        results = {}
        for task in self.future_results:
            results[task.id] = {
                metric.name: task.metric_results[i][1] for i, metric in enumerate(self.metrics) if (not metric.calculate_on_group)
            }
        group_results = {
            name: { 'score': self.group_metrics[name][0], 'detail': self.group_metrics[name][1] } for name in self.group_metrics
        }
        json.dump({ 'single': results, 'group': group_results }, open(path, 'w'), indent=2)
        if self.verbose: print_log(f'Details for evaluation metrics saved to {path}')


    def __del__(self):
        ray.shutdown()

    def __len__(self):
        return len(self.futures)