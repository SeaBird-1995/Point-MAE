'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-17 12:13:52
Email: haimingzhang@link.cuhk.edu.cn
Description: 
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

from utils.config import *
from tools import builder
from utils import misc
from datasets import build_dataset_from_cfg

config = "./cfgs/pretrain.yaml"
config = cfg_from_yaml_file(config)


def get_roll_pitch(taxonomy_ids):
    if taxonomy_ids[0] == "02691156":
        a, b= 90, 135
    elif taxonomy_ids[0] == "04379243":
        a, b = 30, 30
    elif taxonomy_ids[0] == "03642806":
        a, b = 30, -45
    elif taxonomy_ids[0] == "03467517":
        a, b = 0, 90
    elif taxonomy_ids[0] == "03261776":
        a, b = 0, 75
    elif taxonomy_ids[0] == "03001627":
        a, b = 30, -45
    else:
        a, b = 0, 0
    return a, b

def test_TestDataloader():
    args = EasyDict(distributed=False, num_workers=16)
    config.dataset.test.others.bs = 1

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    print(len(test_dataloader))

    test_iter = iter(test_dataloader)

    for i, entry in enumerate(test_dataloader):
        if i >= 50:
            break

    taxonomy_ids, model_ids, data = entry
    print(taxonomy_ids, model_ids, type(taxonomy_ids))
    print(data.shape)

    a, b = get_roll_pitch(taxonomy_ids)

    points = data.squeeze().detach().cpu().numpy()
    print(points.shape)
    np.savetxt("input_points.xyz", points)

    points_image = misc.get_ptcloud_img(points, a, b)
    print(points_image.shape)

    img_path = "./input_points.png"
    cv2.imwrite(img_path, points_image[150:650, 150:675, :])


def test_TrainDataloader():
    args = EasyDict(distributed=False, num_workers=16)
    config.dataset.train.others.bs = config.total_bs

    # _, train_dataloader = builder.dataset_builder(args, config.dataset.train)

    dataset_config = config.dataset.train
    dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
    print(len(dataset))

    entry = dataset[0]
    print(type(entry))


def test_PU1KDataset():
    config = "./cfgs/pretrain_VQAE_PU1K.yaml"
    config = "./cfgs/pretrain_AE_PU1K_h5.yaml"
    config = "./cfgs/pretrain_VQAE_PU1K_h5.yaml"
    
    config = cfg_from_yaml_file(config)

    config.dataset.train.others.bs = config.total_bs

    dataset_config = config.dataset.test

    dataset = build_dataset_from_cfg(dataset_config._base_, dataset_config.others)
    print(len(dataset))

    entry = dataset[0]
    _, _, data = entry
    print(data.shape)
    print(type(entry))


if __name__ == "__main__":
    # import fire
    # fire.Fire()

    test_PU1KDataset()
