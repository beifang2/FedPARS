#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.optim as toptim

def sgd(params, optim_args):
    return toptim.SGD(params, **optim_args)

def adam(params, optim_args):
    return toptim.Adam(params, **optim_args)
