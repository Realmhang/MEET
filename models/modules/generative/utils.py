#!/usr/bin/python
# -*- coding:utf-8 -*-
from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Mapping, Optional, Union

import torch
import torch.nn.functional as F

BUILTIN_WHITELIST = {
    "pi": math.pi,
    "e": math.e,

    "pow": pow,
    "sqrt": torch.sqrt,

    "exp": torch.exp,
    "log": torch.log,

    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,

    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,

    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,

    "abs": torch.abs,
    "round": round,
    "floor": torch.floor,
    "ceil": torch.ceil,

    "max": max,
    "min": min,
    "sum": sum,
}


@dataclass
class XtInfo:
    xt: torch.Tensor
    x0: Optional[torch.Tensor] = None
    eps: Optional[torch.Tensor] = None
    vector_field: Optional[torch.Tensor] = None


def expand_like(src: Union[torch.Tensor, float, int], tgt: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(src):
        src = torch.tensor(src, device=tgt.device, dtype=tgt.dtype)
    else:
        src = src.to(device=tgt.device)
        if torch.is_floating_point(tgt) and torch.is_floating_point(src):
            src = src.to(dtype=tgt.dtype)
    src = src.reshape(*src.shape, *[1 for _ in tgt.shape[len(src.shape):]])
    return src.expand_as(tgt)


def maybe_expand_like(
    src: Union[torch.Tensor, float, int],
    tgt: torch.Tensor,
    batch_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not torch.is_tensor(src):
        src = torch.tensor(src, device=tgt.device, dtype=tgt.dtype)
    if batch_ids is not None and src.ndim > 0 and src.shape[0] != tgt.shape[0] and src.shape[0] != 1:
        src = src[batch_ids]
    return expand_like(src, tgt)


class ModelTarget(str, Enum):
    X0 = "x0"
    EPS = "eps"
    MINUS_EPS = "minus_eps"
    VECTOR_FIELD = "vector_field"


ModelTargets = Mapping[str, Union[ModelTarget, str]]

def convert_model_prediction_to_xtinfo(xt: torch.Tensor, model_prediction: torch.Tensor, model_target: Union[ModelTarget, str]) -> XtInfo:
    if model_target == ModelTarget.X0:
        x0 = model_prediction
        eps = None
        vector_field = None
    elif model_target == ModelTarget.EPS:
        eps = model_prediction
        x0 = None
        vector_field = None
    elif model_target == ModelTarget.MINUS_EPS:
        eps = -model_prediction
        x0 = None
        vector_field = None
    elif model_target == ModelTarget.VECTOR_FIELD:
        vector_field = model_prediction
        x0 = None
        eps = None
    else:
        raise ValueError(f"Unknown model target: {model_target}")
    
    return XtInfo(xt=xt, x0=x0, eps=eps, vector_field=vector_field)

def field_mse_loss(
    prediction: torch.Tensor,
    xt_info: XtInfo,
    model_target: Union[ModelTarget, str]
):
    if model_target == ModelTarget.X0:
        target = xt_info.x0
    elif model_target == ModelTarget.EPS:
        target = xt_info.eps
    elif model_target == ModelTarget.MINUS_EPS:
        target = -xt_info.eps
    elif model_target == ModelTarget.VECTOR_FIELD:
        target = xt_info.vector_field
    else:
        raise ValueError(f"Unknown model target: {model_target}")
    
    loss = F.mse_loss(prediction, target, reduction='none')
    return loss

def safe_eval(expr: str, context: Optional[dict] = None):
    safe_local = BUILTIN_WHITELIST.copy()
    if context:
        safe_local.update(context)
    return eval(expr, {"__builtins__": {}}, safe_local)

class LossWeight:

    def __init__(
        self,
        weight: float | str | Callable = 1.0
    ):
        self.weight = weight

    def get_weight(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.weight, float):
            return torch.full_like(t, self.weight)
        elif isinstance(self.weight, str):
            weight = safe_eval(self.weight, {'t': t})
        elif callable(self.weight):
            weight = self.weight(t)
        else:
            raise ValueError(f"Unknown weight type: {type(self.weight)}")
        
        return weight
