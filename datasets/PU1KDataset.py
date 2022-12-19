'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-18 17:09:45
Email: haimingzhang@link.cuhk.edu.cn
Description: PU1K dataset.
'''
import os
import numpy as np
import os.path as osp
import torch.utils.data as data

from utils.pc_utils.operations import normalize_point_cloud
import utils.pc_utils.transform_utils as utils
from .build import DATASETS


@DATASETS.register_module()
class PU1KCompleteData(data.Dataset):

    def __init__(self, config, is_train=True):

        super().__init__()

        data_dir = config.data_dir
        
        self.is_train = is_train
        self.data_dir = data_dir
        self.gt_dir = data_dir

        self.all_pc_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".xyz")])

    def __getitem__(self, index):
        pc_file = self.all_pc_files[index]
        pc_fp = osp.join(self.data_dir, pc_file)

        pc_data = np.loadtxt(pc_fp, dtype=np.float32)

        gt_data, centroid, furthest_distance = normalize_point_cloud(pc_data)

        ## Data augmentation
        if self.is_train:
            gt_data, _ = utils.rotate_point_cloud_and_gt(gt_data)

        name = pc_file[:-4]
        return name, name, gt_data

    def __len__(self):
        return len(self.all_pc_files)
