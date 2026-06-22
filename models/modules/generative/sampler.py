#!/usr/bin/python
# -*- coding:utf-8 -*-

from abc import ABC, abstractmethod
from typing import Optional
import torch

from .path import ProbabilityPath
from .utils import XtInfo, maybe_expand_like

class BaseSampler(ABC):
    
    def __init__(self, path: ProbabilityPath):
        self.path = path

    @abstractmethod
    def step(self, xt_info: XtInfo, t_prev: torch.Tensor, t_next: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        '''
            One step update of the sampler
            :param xt_info: the current state information at time t
            :param t_prev: current time step
            :param t_next: next time step
            :param batch_ids: optional batch ids for variable sized data
            :return: the updated state information at time t_next
        '''
        pass

    @abstractmethod
    def x0_prediction(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        '''
            Predict x0 from the current state information at time t
            :param xt_info: the current state information at time t
            :param t: current time step
            :param batch_ids: optional batch ids for variable sized data
            :return: the predicted x0
        '''
        pass


class BaseEulerMaruyamaSampler(BaseSampler):
    '''
        The Euler-Maruyama sampler
        x_{t+dt} = x_t + drift_t dt + diffusion_t dBt
    '''

    def step(self, xt_info: XtInfo, t_prev: torch.Tensor, t_next: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        dt = t_next - t_prev
        dt = maybe_expand_like(dt, xt_info.xt, batch_ids=batch_ids)
        dBt = torch.randn_like(xt_info.xt) * torch.sqrt(torch.abs(dt))
        f_t, g_t = self.get_reversed_drift_and_diffusion(xt_info, t_prev, batch_ids)
        xt_plus_dt = xt_info.xt + f_t * dt + g_t * dBt
        return XtInfo(xt = xt_plus_dt)


    @abstractmethod
    def get_reversed_drift_and_diffusion(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class DiffusionEulerMaruyamaSampler(BaseEulerMaruyamaSampler):


    def __init__(self, path: ProbabilityPath, noise_scale: float = 1.0, marginal_concentration_factor: float = 1.0):
        '''
            :param path: the probability path with SDE form
            :param noise_scale: scale of the noise term
            :param marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. 
            Let \alpha, c denote the noise scale and the marginal concentration factor, the sampler targets p(x)^c where p(x) is the data distribution.
            drift_t = f(x_t, t) - 0.5 * c * (1 + \alpha ^ 2) * g(t) ^ 2 * \nabla_{x_t} log p(x_t|x_0)
            diffusion_t = \alpha * g(t)
        '''
        super(DiffusionEulerMaruyamaSampler, self).__init__(path)
        assert self.path.has_markovian_sde, "This sampler only supports SDEs with explicit markovian forward form"
        self.noise_scale = noise_scale
        self.marginal_concentration_factor = marginal_concentration_factor

    def get_reversed_drift_and_diffusion(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        xt_info = self.path.recover_all_info(xt_info, t, batch_ids = batch_ids)
        f_t, g_t = self.path.sde(xt_info, t, batch_ids = batch_ids)
        score_t = self.path.score(xt_info, t, batch_ids = batch_ids)
        f_t = f_t - 0.5 * (g_t ** 2) * score_t * self.marginal_concentration_factor * (1 + self.noise_scale ** 2)
        g_t = g_t * self.noise_scale
        return f_t, g_t
    
    def x0_prediction(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        xt_info = self.path.recover_all_info(xt_info, t, batch_ids = batch_ids)
        return XtInfo(xt = xt_info.x0)

class DiffusionProbabilityFlowODESampler(DiffusionEulerMaruyamaSampler):

    def __init__(self, path: ProbabilityPath, noise_scale: float = 1.0, marginal_concentration_factor: float = 1.0):

        super(DiffusionProbabilityFlowODESampler, self).__init__(path, noise_scale = 0.0, marginal_concentration_factor = marginal_concentration_factor)


class FlowMatchingEulerMaruyamaSampler(BaseEulerMaruyamaSampler):

    def __init__(self, path: ProbabilityPath):
        '''
            :param path: the probability path with ODE form
            x_{t+dt} = x_t + v_t dt
        '''
        super(FlowMatchingEulerMaruyamaSampler, self).__init__(path)

    def get_reversed_drift_and_diffusion(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        xt_info = self.path.recover_all_info(xt_info, t, batch_ids = batch_ids, consider_vector_field = True)
        f_t, g_t = xt_info.vector_field, torch.zeros_like(xt_info.vector_field)
        return f_t, g_t

    def x0_prediction(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        xt_info = self.path.recover_all_info(xt_info, t, batch_ids = batch_ids, consider_vector_field = True)
        return XtInfo(xt = xt_info.x0)

class AncestralSampler(BaseSampler):


    def __init__(self, path: ProbabilityPath, t_clip: float = 1e-4):
        '''
            The ancestral sampler for SDEs with explicit markovian forward form
            x_{t|s} = 1/\eta(t|s) * (x_t + \sigma^2(t|s) * score_t) + std * z
            where std = sqrt(\sigma(t|s) * \sigma(s) / \sigma(t|s))
            :param path: the probability path with SDE form
            :param t_clip: clip the small value of sigma(t) to avoid numerical issues
        '''
        super(AncestralSampler, self).__init__(path)
        assert self.path.has_markovian_sde, "This sampler only supports SDEs with explicit markovian forward form"
        self.t_clip = t_clip

    def step(self, xt_info: XtInfo, t_prev: torch.Tensor, t_next: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:

        # denote t, s as t_prev and t_next
        
        eta_t, eta_s, sigma_t, sigma_s = self.path.eta(t_prev), self.path.eta(t_next), self.path.sigma(t_prev), self.path.sigma(t_next)

        eta_t = maybe_expand_like(eta_t, xt_info.xt, batch_ids=batch_ids)
        eta_s = maybe_expand_like(eta_s, xt_info.xt, batch_ids=batch_ids)
        sigma_t = maybe_expand_like(sigma_t, xt_info.xt, batch_ids=batch_ids)
        sigma_s = maybe_expand_like(sigma_s, xt_info.xt, batch_ids=batch_ids)

        s_is_zero = sigma_s < self.t_clip

        sigma2_t_given_s = sigma_t**2 - sigma_s**2 * eta_t**2 / eta_s**2
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        std = sigma_t_given_s * sigma_s / sigma_t

        eta_t_given_s = eta_t / eta_s

        xt_info = self.path.recover_all_info(xt_info, t_prev, batch_ids = batch_ids)
        score_t = self.path.score(xt_info, t_prev, batch_ids = batch_ids)

        noise = torch.where(s_is_zero, torch.zeros_like(xt_info.xt), torch.randn_like(xt_info.xt))

        xt_plus_dt = (1.0 / eta_t_given_s) * (xt_info.xt + sigma2_t_given_s * score_t) + std * noise

        return XtInfo(xt = xt_plus_dt)

    def x0_prediction(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        xt_info = self.path.recover_all_info(xt_info, t, batch_ids = batch_ids)
        return XtInfo(xt = xt_info.x0)
    
class DDIMSampler(BaseSampler):

    def __init__(self, path: ProbabilityPath, noise_scale: float = 0.0, t_clip: float = 1e-4):
        '''
            The DDIM sampler for SDEs with explicit markovian forward form
            x_s = \eta(s) x_0 + \sqrt(\sigma^2(s) - \sigma^2(s|t)) * eps + \sigma(s|t) * z
            :param path: the probability path with SDE form
            :param noise_scale: the scale of the noise term. noise_scale = 0 corresponds to the deterministic sampler.
            :param t_clip: clip the small value of sigma(t) to avoid numerical issues
        '''
        super(DDIMSampler, self).__init__(path)
        assert self.path.has_markovian_sde, "This sampler only supports SDEs with explicit markovian forward form"
        self.noise_scale = noise_scale
        self.t_clip = t_clip

    def step(self, xt_info: XtInfo, t_prev: torch.Tensor, t_next: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:

        # denote t, s as t_prev and t_next
        
        eta_t, eta_s, sigma_t, sigma_s = self.path.eta(t_prev), self.path.eta(t_next), self.path.sigma(t_prev), self.path.sigma(t_next)

        eta_t = maybe_expand_like(eta_t, xt_info.xt, batch_ids=batch_ids)
        eta_s = maybe_expand_like(eta_s, xt_info.xt, batch_ids=batch_ids)
        sigma_t = maybe_expand_like(sigma_t, xt_info.xt, batch_ids=batch_ids)
        sigma_s = maybe_expand_like(sigma_s, xt_info.xt, batch_ids=batch_ids)

        s_is_zero = sigma_s < self.t_clip

        sigma_relative = self.noise_scale * (sigma_s / sigma_t) * torch.sqrt(1 - (eta_t**2) / (eta_s**2))
        sigma_relative = torch.where(s_is_zero, torch.zeros_like(sigma_relative), sigma_relative)

        score_coeff = torch.sqrt(sigma_s ** 2 - sigma_relative ** 2)

        xt_info = self.path.recover_all_info(xt_info, t_prev, batch_ids = batch_ids)
        mu_next = eta_s * xt_info.x0 + score_coeff * xt_info.eps
        noise = torch.where(s_is_zero, torch.zeros_like(xt_info.xt), torch.randn_like(xt_info.xt))
        xt_plus_dt = mu_next + sigma_relative * noise
        return XtInfo(xt = xt_plus_dt)

    def x0_prediction(self, xt_info: XtInfo, t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None) -> XtInfo:
        xt_info = self.path.recover_all_info(xt_info, t, batch_ids = batch_ids)
        return XtInfo(xt = xt_info.x0)
