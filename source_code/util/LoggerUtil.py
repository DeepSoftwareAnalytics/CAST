# !/usr/bin/env python
# !-*-coding:utf-8 -*- 

import logging
import sys
import torch
from torch.nn.modules.module import _addindent
import numpy as np

log_name = 'CodeSummary'


def set_logger(DEBUG, log_file, checkpoint=False):
    if DEBUG:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger = logging.getLogger(log_name)
    logger.propagate = False  # https://stackoverflow.com/a/44426266
    # ch = logging.StreamHandler(sys.stdout)
    ch = logging.StreamHandler()
    # ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.setLevel(log_level)
    if log_file:
        if checkpoint:
            logfile = logging.FileHandler(log_file, 'a')
        else:
            logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    return logger


def info_logger(msg):
    logger = logging.getLogger(log_name)
    logger.info(msg)


def debug_logger(msg):
    logger = logging.getLogger(log_name)
    logger.debug(msg)


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
