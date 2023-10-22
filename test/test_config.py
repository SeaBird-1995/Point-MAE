'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-16 14:46:45
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

config = "./cfgs/pretrain.yaml"
config = cfg_from_yaml_file(config)
print(config, type(config))
