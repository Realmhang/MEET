#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, List
from dataclasses import dataclass, field

import torch


@dataclass
class LatentGeneratorSampleConfig:
    h_solver_type: str                                  # e.g. DiffusionSDE, DiffusionODE
    v_solver_type: str                                  # e.g. DiffusionSDE, DiffusionODE
    x_solver_type: str                                  # e.g. DiffusionSDE, DiffusionODE
    h_solver_args: dict = field(default_factory=dict)   # arguments for solver
    v_solver_args: dict = field(default_factory=dict)   # arguments for solver
    x_solver_args: dict = field(default_factory=dict)   # arguments for solver
    num_steps: int = 100                                # number of steps for ODE/SDE solvers
    vae_decoder_n_iter: int = 10                        # number of iterations in the flow matching of the VAE
    return_tensor: bool = False                         # return tensors instead
    kwargs: dict = field(default_factory=dict)          # other key-value arguments
