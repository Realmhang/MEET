#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from abc import ABC, abstractmethod


class TimeScheduler(ABC):
    '''
        Base class for time schedulers
    '''
    @abstractmethod
    def sample_time_steps(self, num_samples) -> torch.Tensor:
        '''
            Get the time steps for sampling
            :param num_samples: number of time steps
            :return: a tensor of shape [num_samples] containing the time steps
        '''
        pass

class UniformTimeScheduler(TimeScheduler):
    '''
        Uniform time scheduler
    '''
    def __init__(self, t_min: float = 0.0, t_max: float = 1.0):
        self.t_min = t_min
        self.t_max = t_max

    def sample_time_steps(self, num_samples) -> torch.Tensor:
        return torch.rand(num_samples) * (self.t_max - self.t_min) + self.t_min
