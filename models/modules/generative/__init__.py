from .path import ProbabilityPath, LinearDDPMPath, CosineDDPMPath, VanillaFlowMatching
from .sampler import BaseSampler, DiffusionEulerMaruyamaSampler, DiffusionProbabilityFlowODESampler, FlowMatchingEulerMaruyamaSampler, AncestralSampler, DDIMSampler
from .base_model import IterativeGenerativeModel
from .solver import Solver
from .time_scheduler import TimeScheduler, UniformTimeScheduler
from .utils import XtInfo, ModelTarget, LossWeight
from functools import partial

__all__ = [
    'ProbabilityPath', 'LinearDDPMPath', 'CosineDDPMPath', 'VanillaFlowMatching',
    'BaseSampler', 'DiffusionEulerMaruyamaSampler', 'DiffusionProbabilityFlowODESampler',
    'FlowMatchingEulerMaruyamaSampler', 'AncestralSampler', 'DDIMSampler',
    'IterativeGenerativeModel', 'Solver', 'TimeScheduler', 'UniformTimeScheduler',
    'XtInfo', 'ModelTarget', 'LossWeight', 'make_path', 'make_sampler_partial',
]

def make_path(name:str, **kwargs):

    if name == 'LinearDDPM':
        return LinearDDPMPath(**kwargs)

    elif name == 'CosineDDPM':
        return CosineDDPMPath(**kwargs)

    elif name == 'LinearFlowMatching':
        return VanillaFlowMatching(**kwargs)

    else:
        raise NotImplementedError(f"Unknown Path Name: {name}")

def make_sampler_partial(name:str, **kwargs):

    if name == 'DiffusionSDE':
        return partial(
            DiffusionEulerMaruyamaSampler,
            **kwargs
        )

    elif name == 'DiffusionODE':
        return partial(
            DiffusionProbabilityFlowODESampler,
            **kwargs
        )

    elif name == 'FlowMatchingODE':
        return partial(
            FlowMatchingEulerMaruyamaSampler,
            **kwargs
        )

    elif name == 'AncestralSampler':
        return partial(
            AncestralSampler,
            **kwargs
        )
    
    elif name == 'DDIMSDE':
        return partial(
            DDIMSampler,
            **kwargs
        )

    elif name == 'DDIMODE':
        return partial(
            DDIMSampler,
            noise_scale = 0.0,
            **kwargs
        )

    else:
        raise NotImplementedError(f"Unknown Sampler Name: {name}")

    
