#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, Union
from abc import ABC, abstractmethod

import torch
import numpy as np

from .utils import XtInfo, maybe_expand_like


class ProbabilityPath(ABC):
    '''
        Define the probability path. Implemented as:
            x_t = eta_t x_0 + sigma_t eps, where x_0 is the data distribution, eps is the prior distribution
        for flow matching, eps is the x_1, while for diffusion, it is the noise used to distort x_0
        Different path should implement different scheduler of eta_t and sigma_t
    '''
    @abstractmethod
    def eta(self, t: torch.Tensor) -> torch.Tensor:
        '''
            Coefficient of x_0 for forward interpolation
        '''
        pass

    @abstractmethod
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        '''
            Coefficient of eps for forward interpolation
        '''
        pass

    @abstractmethod
    def d_eta(self, t: torch.Tensor) -> torch.Tensor:
        '''
            Derivative of eta(t) w.r.t. t, i.e., eta'(t)
        '''
        pass

    @abstractmethod
    def d_sigma(self, t: torch.Tensor) -> torch.Tensor:
        '''
            Derivative of sigma(t) w.r.t. t, i.e., sigma'(t)
        '''
        pass

    @property
    def has_markovian_sde(self) -> bool:
        '''
        Whether the probability path has an explicit markovian sde form
        '''
        return False

    def sde(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        '''
            Given x_t, return f(x_t, t) and g(t)
            dX = f(X,t)dt + g(t)dW
            Only supported when `has_markovian_sde` is set as true
        '''
        raise NotImplementedError("This probability path does not have an explicit markovian sde form")

    
    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, consider_vector_field: bool = False, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        eps = torch.randn_like(x0)
        eta_t, sigma_t = self.eta(t).to(x0.dtype), self.sigma(t).to(x0.dtype)
        eta_t, sigma_t = maybe_expand_like(eta_t, x0, batch_ids=batch_ids), maybe_expand_like(sigma_t, eps, batch_ids=batch_ids)
        xt = eta_t * x0 + sigma_t * eps
        if consider_vector_field:
            d_eta_t, d_sigma_t = self.d_eta(t), self.d_sigma(t)
            d_eta_t, d_sigma_t = maybe_expand_like(d_eta_t, x0, batch_ids=batch_ids), maybe_expand_like(d_sigma_t, eps, batch_ids=batch_ids)
            vector_field = d_eta_t * x0 + d_sigma_t * eps
        else:
            vector_field = None
        return XtInfo(xt=xt, x0=x0, eps=eps, vector_field=vector_field)

    def recover_all_info(self, xt_info: XtInfo, t: torch.Tensor, consider_vector_field: bool=False, batch_ids: Optional[torch.Tensor] = None, t_clip: float=1e-4) -> XtInfo:
        '''
            x_t = eta(t) x_0 + sigma(t) eps
            v_t = d_eta(t) x_0 + d_sigma(t) eps
            We have two equations but three unknowns (x0, eps, v_t)
            Hence given each of them, we can recover the other two
            To avoid numerical issues, we clip t into [t_clip, 1 - t_clip]
        '''
        t = torch.clamp(t, min=t_clip, max=1.0 - t_clip)
        eta_t, sigma_t = self.eta(t), self.sigma(t)
        eta_t, sigma_t = maybe_expand_like(eta_t, xt_info.xt, batch_ids=batch_ids), maybe_expand_like(sigma_t, xt_info.xt, batch_ids=batch_ids)

        if consider_vector_field and xt_info.vector_field is not None:
            vector_field = xt_info.vector_field
            d_eta_t, d_sigma_t = self.d_eta(t), self.d_sigma(t)
            d_eta_t, d_sigma_t = maybe_expand_like(d_eta_t, xt_info.xt, batch_ids=batch_ids), maybe_expand_like(d_sigma_t, xt_info.xt, batch_ids=batch_ids)
            # solve the linear equations
            # [eta(t), sigma(t)] [x0]   = [xt]
            # [d_eta(t), d_sigma(t)] [eps] = [v_t]
            det = eta_t * d_sigma_t - sigma_t * d_eta_t
            x0 = (d_sigma_t * xt_info.xt - sigma_t * vector_field) / det
            eps = (-d_eta_t * xt_info.xt + eta_t * vector_field) / det
            return XtInfo(xt=xt_info.xt, x0=x0, eps=eps, vector_field=vector_field)

        if xt_info.x0 is not None:
            x0 = xt_info.x0
            eps = (xt_info.xt - eta_t * x0) / sigma_t
        elif xt_info.eps is not None:
            eps = xt_info.eps
            x0 = (xt_info.xt - sigma_t * eps) / eta_t
        else:
            raise ValueError("When neither x0 nor eps is given in xt_info, consider_vector_field should be set to True and vector_field should be given in xt_info")

        if consider_vector_field:
            d_eta_t, d_sigma_t = self.d_eta(t), self.d_sigma(t)
            d_eta_t, d_sigma_t = maybe_expand_like(d_eta_t, xt_info.xt, batch_ids=batch_ids), maybe_expand_like(d_sigma_t, xt_info.xt, batch_ids=batch_ids)
            vector_field = d_eta_t * x0 + d_sigma_t * eps
        else:
            vector_field = None
        
        return XtInfo(xt=xt_info.xt, x0=x0, eps=eps, vector_field=vector_field)

    def score(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
            return the score function \nabla_{x_t} log p(x_t|x_0)
            for Gaussian path, we have p(x_t|x_0) = N(eta(t)x_0, sigma(t)^2 I)
            thus, score = - (x_t - eta(t)x_0) / sigma(t)^2 = - eps / sigma(t)
        '''
        sigma_t = self.sigma(t)
        sigma_t = maybe_expand_like(sigma_t, xt_info.eps, batch_ids=batch_ids)
        return - xt_info.eps / sigma_t

    def sample_prior(self, shape: torch.Size, device: torch.device) -> XtInfo:
        '''
            Sample from the prior distribution
        '''
        return XtInfo(xt = torch.randn(shape, device=device))  # standard Gaussian prior




'''
    For flow matching, we can inherit the class below and overload sigma(t)
'''

class BaseFlowMatchingPath(ProbabilityPath):
    '''
        flow matching
            x_t = eta(t) x_0 + sigma(t) x_1, where eta(t) + sigma(t) = 1.0
    '''

    def eta(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.sigma(t)

    def d_eta(self, t: torch.Tensor) -> torch.Tensor:
        return -self.d_sigma(t)

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, consider_vector_field: bool = False, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        return super().sample_xt(x0, t, batch_ids = batch_ids, consider_vector_field = True)


class VanillaFlowMatching(BaseFlowMatchingPath):
    '''
        vanilla (direct line between x0 and x1)
            x_t = (1 - t) x_0 + t x_1
    '''

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def d_sigma(self, t: torch.Tensor) -> Union[float, torch.Tensor]:
        return torch.ones_like(t)  # constant




'''
    For diffusion, we can inherit the class below and overload
'''
class BaseDiffusionPath(ProbabilityPath):
    '''
        dX = f(X,t)dt + g(t)dW from song et al.
    '''
    @property
    def has_markovian_sde(self) -> bool:
        return True



class AffineDiffusionPath(BaseDiffusionPath):
    '''
        Assume the drifting coefficient is affine, namely f(x, t) = a(t) x + b(t).
        In this way, we have X_t|X_0 ~ N(E[X_t|X_0]), Var[X_t|X_0]).
            E[X_t|X_0] = [x_0 + \int_0^t exp(-\int_0^s a(r)dr)b(s)ds]exp(\int_0^t a(s)ds)
            Var[X_t|X_0] = [\int_0^t exp(-\int_0^s 2a(r)dr)g^2(s)ds]exp(\int_0^t 2a(s)ds)
        Therefore, we need to sample x_t with:
            x_t = exp(\int_0^t a(s)ds) x_0 + \sqrt(Var[X_t|X_0]) eps + \int_0^t exp(-\int_0^s a(r)dr)b(s)ds exp(\int_0^t a(s)ds)
        but luckily, in most cases (at least known diffusion), the last bias equals zero
    '''


class DDPMPath(AffineDiffusionPath):
    r'''
        Variance preserving path, also known as DDPM https://arxiv.org/pdf/2102.09672
        For DDPM, f(x, t) = -x 1/2*\beta(t), g(t) = \sqrt(\beta(t))
        Therefore, in the affine path definition, a(t) = -1/2*\beta(t) and b(t) = 0
        Thus, given \beta(0) = 0,
            E[X_t|X_0] = exp(-0.5\int_0^t beta(s) ds) x_0
            Var[X_t|X_0] = (1 - exp(-\int_0^t beta(s) ds)) eps
        So,
            x_t = exp(-0.5\int_0^t beta(s) ds) x_0 + sqrt(1 - exp(-\int_0^t beta(s) ds)) eps
        Denote exp(-0.5\int_0^t beta(s) ds) as eta(t), we have:
            x_t = eta(t) x_0 + sqrt(1 - eta(t)^2) eps
        In this case, all DDPMs are within the same framework and only the beta(t) scheduler need to be changed
    '''

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        '''
            Beta scheduler
        '''
        pass

    @abstractmethod
    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        '''
            Integration of beta scheduler from 0 to t, i.e., \int_0^t beta(s) ds
        '''
        pass
    
    def eta(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.int_beta(t))

    def d_eta(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self.eta(t) * self.beta(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1 - self.eta(t)**2)

    def d_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.eta(t) * self.d_eta(t) / self.sigma(t)

    def sde(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        '''
            Given x_t, return f(x_t, t) and g(t)
            dX = f(X,t)dt + g(t)dW
            For DDPM, f(x, t) = -x 1/2*\beta'(t), g(t) = \sqrt(\beta'(t))
        '''
        beta_t = self.beta(t)
        beta_t = maybe_expand_like(beta_t, xt_info.xt, batch_ids=batch_ids)
        f = -0.5 * beta_t * xt_info.xt
        g = torch.sqrt(beta_t)
        g = maybe_expand_like(g, xt_info.xt, batch_ids=batch_ids)
        return f, g


class LinearDDPMPath(DDPMPath):
    '''
        DDPM with linear scheduler, which is the default in DDPM
    '''
    def __init__(self, beta_min: float=0.1, beta_max: float=20.0):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        # linear interpolation
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2


class CosineDDPMPath(DDPMPath):
    '''
        DDPM with cosine scheduler in the Improved DDPM paper: https://arxiv.org/pdf/2102.09672
    '''
    def __init__(self, s: float=0.008, t_clip = 1e-2):
        self.s = s
        self.c = np.cos(s / (1 + s) * np.pi / 2)
        self.t_clip = t_clip

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=self.t_clip, max=1.0 - self.t_clip)
        return torch.tan((t + self.s) / (1 + self.s) * np.pi / 2) * np.pi / (1 + self.s)

    def int_beta(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=self.t_clip, max=1.0 - self.t_clip)
        return -torch.log(self.c * torch.cos((t + self.s) / (1 + self.s) * np.pi / 2) ** 2) * 2
    
    def eta(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=self.t_clip, max=1.0 - self.t_clip)
        return torch.cos((t + self.s) / (1 + self.s) * np.pi / 2) / self.c
    
    def d_eta(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=self.t_clip, max=1.0 - self.t_clip)
        return -np.pi / (2 * (1 + self.s)) * torch.sin((t + self.s) / (1 + self.s) * np.pi / 2) / self.c
