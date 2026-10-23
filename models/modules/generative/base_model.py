#!/usr/bin/python
# -*- coding:utf-8 -*-

from typing import Mapping, Union, Optional, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch_scatter import scatter

from .path import ProbabilityPath
from .utils import XtInfo, ModelTarget, convert_model_prediction_to_xtinfo, field_mse_loss, LossWeight
from .time_scheduler import TimeScheduler, UniformTimeScheduler

class IterativeGenerativeModel(ABC, nn.Module):
    '''
        Base class for iterative generative models
    '''
    def __init__(
        self,
        backbone: nn.Module,
        paths: dict[str, ProbabilityPath],
        model_targets: Mapping[str, Union[ModelTarget, str]],
        time_scheduler: TimeScheduler = UniformTimeScheduler(),
        loss_weights: Optional[Mapping[str, Union[float, str, Callable[[torch.Tensor], torch.Tensor]]]] = None
    ):
        '''
            :param backbone: the neural network backbone for prediction
            :param paths: a dictionary of ProbabilityPath for each field
            :param model_targets: a dictionary specifying the target for each field
            :param time_scheduler: the time scheduler for sampling time steps
            :param loss_weights: a dictionary specifying the loss weight for each field. If None, all weights are set to 1.0
        '''

        super(IterativeGenerativeModel, self).__init__()
        self.backbone = backbone
        self.paths = paths
        self.model_targets = model_targets
        self.time_scheduler = time_scheduler
        if loss_weights is None:
            self.loss_weights = {field: LossWeight(1.0) for field in paths.keys()}
        else:
            self.loss_weights = {field: LossWeight(weight) for field, weight in loss_weights.items()}

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def preprocess_data(self, batch):
        '''
            preprocess the data for training
            :param batch: the input batch data

            :return: 
                a tuple containing:
                - x0: dict[str, torch.Tensor] the original data for each path
                - batch_size: batch size
                - batch_ids: optional batch ids for variable sized data
                - context_info: other information if needed. A simplest way is to directly return the input batch itself
        '''
        pass

    @abstractmethod
    def postprocess_data(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None):

        '''
            convert xt_infos back to whatever you like
        '''
        pass

    @abstractmethod
    def model_prediction(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None) -> dict[str, torch.Tensor]:
        '''
            Given x_t and t, predict the target for the model
            :param xt_infos: the current state information at time t
            :param t: current time step
            :param batch_ids: optional batch ids for variable sized data
            :param context_info: other information if needed
            :return: the model prediction for each path
        '''
        pass

    def model_step(self, xt_infos: dict[str, XtInfo], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None) -> dict[str, XtInfo]:
        '''
            One step model prediction
            :param xt_infos: the current state information at time t
            :param t: current time step
            :param batch_ids: optional batch ids for variable sized data
            :param context_info: other information if needed
            :return: the model prediction for the given path
        '''
        model_pred = self.model_prediction(xt_infos, t, batch_ids, context_info)

        step_results = {}
        for field in xt_infos.keys():
            model_target = self.model_targets[field]
            step_results[field] = convert_model_prediction_to_xtinfo(
                xt = xt_infos[field].xt, 
                model_prediction = model_pred[field], 
                model_target = model_target
            )

        return step_results

    def compute_loss(self, xt_infos: dict[str, XtInfo], model_predictions: dict[str, torch.Tensor], t: torch.Tensor, batch_ids: Optional[torch.Tensor] = None, context_info = None) -> torch.Tensor:
        
        '''
            Compute the loss for training
            :param xt_infos: the current state information at time t
            :param model_predictions: the model predictions for each path
            :param t: current time step
            :param batch_ids: optional batch ids for variable sized data
            :param context_info: other information if needed
            :return: the computed loss
        '''
        losses = {}
        for field in xt_infos.keys():
            model_target = self.model_targets[field]
            loss = field_mse_loss(
                prediction = model_predictions[field], 
                xt_info = xt_infos[field], 
                model_target = model_target
            )
            loss_per_field_per_sample = loss.mean(dim=tuple(range(1, loss.ndim)))
            if batch_ids is not None:
                # explicit dim_size: torch_scatter's inference does index.max().item()
                # which is a device sync (illegal during CUDA graph capture)
                _ds = int(t.shape[0]) if t is not None else None
                loss_per_field_per_sample = scatter(loss_per_field_per_sample, dim=0,
                                                    index=batch_ids, reduce='mean', dim_size=_ds)
            weight = self.loss_weights[field].get_weight(t)
            loss_per_field_per_sample = loss_per_field_per_sample * weight
            rsm = context_info.get('real_sample_mask') if isinstance(context_info, dict) else None
            if rsm is not None:
                losses[field] = (loss_per_field_per_sample * rsm).sum() / rsm.sum()
            else:
                losses[field] = loss_per_field_per_sample.mean()

        
        total_loss = sum(losses.values())
        return total_loss, losses

    def forward(self, batch):

        '''
            Forward pass for training
            :param batch: the input batch data
            :return: the computed loss
        '''
        x0_dict, batch_size, batch_ids, context_info = self.preprocess_data(batch)
        if isinstance(batch, dict) and '_static_t' in batch:
            t = batch['_static_t']  # pre-sampled outside the graph (capture-safe)
        else:
            t = self.time_scheduler.sample_time_steps(batch_size).to(self.device)

        xt_infos = {}
        for field, path in self.paths.items():
            x0 = x0_dict[field]
            _eps = batch.get(f'_static_eps_{field}') if isinstance(batch, dict) else None
            xt_infos[field] = path.sample_xt(x0, t, batch_ids = batch_ids, eps=_eps)

        model_predictions = self.model_prediction(xt_infos, t, batch_ids, context_info)

        loss, loss_details = self.compute_loss(xt_infos, model_predictions, t, batch_ids, context_info)

        return loss, loss_details
