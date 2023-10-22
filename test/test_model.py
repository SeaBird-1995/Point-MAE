'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-16 14:51:00
Email: haimingzhang@link.cuhk.edu.cn
Description: Testing model.
'''

import sys
sys.path.append("./")
sys.path.append("../")

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import torchvision
import cv2
from PIL import Image
from omegaconf import OmegaConf
import torch

from tools import builder

from utils.config import *


def test_PointMAE():
    config = "./cfgs/pretrain.yaml"
    config = cfg_from_yaml_file(config)
    base_model = builder.model_builder(config.model)
    base_model = base_model.cuda()
    # print(base_model)

    input = torch.randn(4, 1024, 3).cuda()
    output = base_model(input)
    print(output)


def test_PointAE():
    config = "./cfgs/pretrain_AE.yaml"
    config = cfg_from_yaml_file(config)

    base_model = builder.model_builder(config.model)
    base_model = base_model.cuda()

    input = torch.randn(4, 1024, 3).cuda()
    output = base_model(input)
    print(output)


def test_PointVQAE():
    config = "./cfgs/pretrain_VQAE.yaml"
    config = cfg_from_yaml_file(config)

    base_model = builder.model_builder(config.model)
    base_model = base_model.cuda()

    input = torch.randn(4, 1024, 3).cuda()
    loss, log_dict = base_model(input)
    print(loss)
    print(log_dict['unique_idx_length'], log_dict['loss_vq'])


if __name__ == "__main__":
    test_PointVQAE()
