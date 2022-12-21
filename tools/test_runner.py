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
import time
from tqdm import tqdm
from einops import rearrange

from utils.logger import *

import numpy as np
from pointnet2_ops import pointnet2_utils
from torchmetrics import MeanMetric
from .pu_loss import ChamferLoss


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

            dense_points, vis_points, centers= base_model(points, vis=True)
            dense_points = rearrange(dense_points, 'x (b n) c -> (x b) n c', b=bs)

            pred = misc.fps(dense_points, npoints)
            
            ## Compute the loss
            loss_cd = chamfer_criteria(pred, points)
            avg_loss_metric.update(loss_cd)
        
        avg_loss = avg_loss_metric.compute()
        print_log('[Validation] avg cd loss = %.4f' % (avg_loss), logger=logger)