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
import h5py

from utils.pc_utils.operations import normalize_point_cloud
import utils.pc_utils.transform_utils as utils
from .build import DATASETS


@DATASETS.register_module()
class PU1KStage1Dataset(data.Dataset):

    def __init__(self,
                 config):

        super().__init__()

        h5_path = config.h5_path
        split = config.split

        self.isTrain = config.isTrain
        self.use_aug = config.use_aug

        h5_file = h5py.File(h5_path)
        poisson_1024 = h5_file['poisson_1024'][:]  # (69000, 1024, 3)

        normalized_data, centroid, furthest_distance = normalize_point_cloud(poisson_1024)

        self.gt = normalized_data
        self.radius = np.ones(shape=(len(self.gt)))

        if split is not None and osp.exists(split):
            self._load_split_file(split)

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, index):
        gt_data = self.gt[index].astype(np.float32)  # (1024, 3)
        radius_data = np.array([self.radius[index]]).astype(np.float32)

        if self.isTrain and self.use_aug:
            gt_data, _ = utils.rotate_point_cloud_and_gt(gt_data)

        return "04090263", "04090263", gt_data

    def _load_split_file(self, split):
        index = np.loadtxt(split).astype(np.int)
        self.gt = self.gt[index, :]


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

@DATASETS.register_module()
class PU1KTestData(data.Dataset):

    def __init__(self, config):
        test_dir_path = config.data_dir
        num_input_point = config.num_input_point
        up_ratio = config.up_ratio

        testing_dir = osp.join(test_dir_path, f"input_{num_input_point}")
        self.input_dir = osp.join(testing_dir, f"input_{num_input_point}")
        self.gt_dir = osp.join(testing_dir, f"gt_{num_input_point * up_ratio}")

        self.all_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith(".xyz")])

        self.length = len(self.all_files)

    def __getitem__(self, index):
        obj_file = self.all_files[index]

        input_fp = osp.join(self.input_dir, obj_file)
        gt_fp = osp.join(self.gt_dir, obj_file)

        input_points = np.loadtxt(input_fp).astype(np.float32)
        gt_points = np.loadtxt(gt_fp).astype(np.float32)

        input_pc, input_centroid, input_furthest_distance = normalize_point_cloud(input_points)
        gt_pc, gt_centroid, gt_furthest_distance = normalize_point_cloud(gt_points)

        name = obj_file[:-4]
        return name, (gt_centroid, gt_furthest_distance), gt_pc

    def __len__(self):
        return self.length