#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, List
from dataclasses import dataclass, field

import torch


@dataclass
class VAESampleConfig:
    n_iter: int = 10                                # number of iterations in the flow matching
    return_tensor: bool = False                     # return tensors instead
    latents: Optional[List[torch.Tensor]] = None    # [Nblock, h_latent_size], [Nblock, 3, v_latent_size], [Nblock, 3], specify the SE(3)-invariant, SO(3)-equivariant, and SE(3)-equivariant latent variable
    manual_noise_h_std: Optional[float] = None      # manual noises on ZH to test robustness
    manual_noise_v_std: Optional[float] = None      # manual noises on ZV to test robustness
    manual_noise_x_std: Optional[float] = None      # manual noises on ZX to test robustness
    kwargs: dict = field(default_factory=dict)  # other key-value arguments
