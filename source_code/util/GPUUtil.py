# !/usr/bin/env python
# !-*-coding:utf-8 -*-

import torch
import os
from util.LoggerUtil import info_logger
from util.LoggerUtil import *
from util.Config import Config as cf

# device = None
USE_GPU = True

def set_device(gpu_id):
    # global device
    global USE_GPU
    if gpu_id == -1:
        cf.device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        if torch.cuda.is_available():
            cf.device = torch.device("cuda:" + str(gpu_id))
            USE_GPU = True
        else:
            info_logger("Cannot find GPU id %s! Use CPU." % gpu_id)
            cf.device = torch.device("cpu")
            USE_GPU = False
    cf.use_gpu = USE_GPU
    info_logger("[Setting] device: %s" % cf.device)


def move_to_device(data):
    # global device
    global USE_GPU

    if USE_GPU:
        return data.cuda(cf.device, non_blocking=True)
    else:
        return data

