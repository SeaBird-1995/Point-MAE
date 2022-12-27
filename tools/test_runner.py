'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-21 10:21:30
Email: haimingzhang@link.cuhk.edu.cn
Description: The testing runner.
'''

import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import os
import os.path as osp
from tqdm import tqdm
from einops import rearrange

from utils.logger import *

import numpy as np
from pointnet2_ops import pointnet2_utils
from torchmetrics import MeanMetric
from .pu_loss import ChamferLoss
import utils.pc_utils.sampling_utils as operations


def test_net(args, config, device):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)

    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    base_model.to(device)
     
    test(base_model, test_dataloader, args, config, device, logger=logger)


def test(base_model, test_dataloader, args, config, device, logger=None):
    chamfer_criteria = ChamferLoss()

    base_model.eval()  # set model to eval mode

    npoints = config.npoints

    avg_loss_metric = MeanMetric().to(device)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            points = data.to(device)
            bs = points.shape[0]

            dense_points, vis_points, centers = base_model(points, vis=True)
            dense_points = rearrange(dense_points, 'x (b n) c -> (x b) n c', b=bs)

            pred = misc.fps(dense_points, npoints)
            
            ## Compute the loss
            loss_cd = chamfer_criteria(pred, points)
            avg_loss_metric.update(loss_cd)
        
        avg_loss = avg_loss_metric.compute()
        print_log('[Validation] avg cd loss = %.4f' % (avg_loss), logger=logger)


def _predict_patches(model, input_pc, num_train_points, patch_num_ratio, num_target_points, NCHW=False):
    ## 1) Split the input into multiple patches
    # divide to patches
    num_patches = int(input_pc.shape[1] / num_train_points * patch_num_ratio)

    # FPS sampling
    idx, seeds = operations.fps_subsample(input_pc, num_patches, NCHW=NCHW)

    patches, _, _ = operations.group_knn(num_train_points, seeds, input_pc, NCHW=NCHW)

    ## 2) Forward each patch
    up_point_list = []
    for k in range(num_patches):
        patch = patches[:, :, k, :] if NCHW else patches[:, k, :, :]

        patch, centroid, furthest_distance = operations.normalize_point_batch(patch, NCHW=NCHW)

        up_point, vis_points, centers= model(patch.detach(), vis=True)

        up_point = up_point * furthest_distance + centroid
        up_point_list.append(up_point)

    pred_pc = torch.cat(up_point_list, dim=1) # to (B, N*k, 3)

    _, pred_pc = operations.fps_subsample(pred_pc,
                                          num_target_points,
                                          NCHW=NCHW)

    return pred_pc


def test_metric(base_model, test_dataloader, args, config, device, logger=None):
    save_dir = osp.join(args.experiment_path, "results")
    os.makedirs(save_dir, exist_ok=True)
    
    base_model.eval()  # set model to eval mode

    with torch.no_grad():
        for idx, (name, norm_params, data) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            points = data.to(device)
            centroid, furthest_distance = norm_params

            pred_pc = _predict_patches(base_model, points, 1024, 3, 8192, NCHW=False)
            ## Convert to the original scale
            pred_pc = centroid + pred_pc * furthest_distance

            ## save the results
            save_path = osp.join(save_dir, f"{name}.xyz")
            np.savetxt(save_path, pred_pc[0].cpu().numpy(), fmt='%.6f')