#!/usr/bin/python
# -*- coding:utf-8 -*-

from .MEET.meet import MEET

def create_net(
    name,
    hidden_size,
    edge_size,
    H_only=False,
    opt={}
):
    if name == 'MEET':
        required_keys = ['n_heads', 'n_layers', 'adaLN_zero', 'backend']
        missing_keys = [key for key in required_keys if key not in opt]
        assert len(missing_keys) == 0, f'keys {missing_keys} are missing, which are required for MEET'
        return MEET(
            d_model=hidden_size,
            H_only=H_only,
            **opt
        )
    else:
        raise NotImplementedError(f'{name} not implemented')
