#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Checkpoint loading helpers.

Load a model by constructing a fresh, code-current instance from a training
config and then loading the checkpoint's ``state_dict`` into it (same pattern as
``train.py``). Relying on ``torch.load`` to unpickle a full model object can
break when the saved class definition has gained new attributes since training
(e.g. older LDM checkpoints whose ``FinalLayer`` is missing the
``use_final_linear`` attribute, which raises ``AttributeError`` on forward).
"""
import torch

import utils.register as R


def load_model(ckpt_path, model_config, map_location='cpu', strict=True):
    """Construct a model from ``model_config`` and load checkpoint weights.

    Args:
        ckpt_path: path to the ``.ckpt`` file.
        model_config: model config dict (must contain a ``class`` key), e.g.
            ``config['model']`` from a training yaml.
        map_location: torch.load map location.
        strict: forwarded to ``load_state_dict`` (defaults to ``True`` so any
            key/shape mismatch raises instead of being silently ignored).

    Returns:
        nn.Module constructed from ``model_config`` with weights loaded from
        the checkpoint.
    """
    model = R.construct(model_config)
    obj = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state_dict = obj.state_dict() if hasattr(obj, 'state_dict') else obj
    model.load_state_dict(state_dict, strict=strict)
    return model
