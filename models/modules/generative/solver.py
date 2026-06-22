#!/usr/bin/python
# -*- coding:utf-8 -*-

from typing import Mapping
import torch

from .base_model import IterativeGenerativeModel
from functools import partial
from .sampler import BaseSampler

from tqdm import tqdm

class Solver:

    '''
        Base class for solvers
    '''

    def __init__(
        self, 
        model: IterativeGenerativeModel,
        sampler_partials: Mapping[str, partial[BaseSampler]],
        num_steps: int = 100,
        predict_x0_final: bool = True
    ):
        self.model = model
        self.samplers = {}
        for path_name in self.model.paths:
            self.samplers[path_name] = sampler_partials[path_name](self.model.paths[path_name])
        self.num_steps = num_steps
        self.device = model.device
        self.predict_x0_final = predict_x0_final

    def sample(self, batch, records = 'none'):

        '''
        Sample from the model given a batch of data.
        :param batch: the input batch data
        :param records: intermediate records to save during sampling. Options: 'none', 'samples'
        '''

        x0, batch_size, batch_ids, context_info = self.model.preprocess_data(batch)
        device = self.device

        t_steps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        xt_infos = {}
        for path_name, path in self.model.paths.items():
            shape = x0[path_name].shape
            xt_infos[path_name] = path.sample_prior(shape, device)

        xt_trajs = [xt_infos]

        for step in tqdm(range(self.num_steps)):
            t_prev = t_steps[step]
            t_next = t_steps[step + 1]
            t_prev_batch = torch.full((batch_size,), t_prev, device=device)
            t_next_batch = torch.full((batch_size,), t_next, device=device)

            xt_infos_prev = xt_trajs[-1]

            model_step = self.model.model_step(xt_infos_prev, t_prev_batch, batch_ids, context_info)
            xt_infos_next = {}

            if self.predict_x0_final and step == self.num_steps - 1:
                for path_name, path in self.model.paths.items():
                    xt_infos_next[path_name] = self.samplers[path_name].x0_prediction(model_step[path_name], t_prev_batch, batch_ids)
            else:
                for path_name, path in self.model.paths.items():
                    xt_infos_next[path_name] = self.samplers[path_name].step(model_step[path_name], t_prev_batch, t_next_batch, batch_ids)

            xt_trajs.append(xt_infos_next)

        xt_infos = xt_trajs[-1]
        final_result = self.model.postprocess_data(xt_infos, t_next_batch, batch_ids, context_info)

        if records == 'samples':
            records_to_return = {
                t.detach().cpu().item(): self.model.postprocess_data(xt, t, batch_ids, context_info)
                for t, xt in zip(t_steps, xt_trajs)
            }
        elif records == 'none':
            records_to_return = None
        else:
            raise ValueError(f"Unknown records option: {records}")

        return final_result, records_to_return
