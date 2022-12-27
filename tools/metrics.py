'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-02 20:48:00
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import os
from collections import defaultdict, OrderedDict
import numpy as np
from glob import glob
from .pu_loss import CD_dist
from utils.pc_utils import io, operations
import csv


def online_evaluation_old(PRED_DIR, GT_DIR, save_path):
    DEVICE = torch.device('cuda', 0)

    precentages = np.array([0.008, 0.012])

    fieldnames = ["name", "CD", "hausdorff", "p2f avg", "p2f std"]
    fieldnames += ["uniform_%d" % d for d in range(precentages.shape[0])]

    gt_paths = glob(os.path.join(GT_DIR, '*.xyz'))

    gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
    # gt = load(gt_paths[0])[:, :3]
    cd_dist_compute = CD_dist()
    avg_md_forward_value = 0
    avg_md_backward_value = 0
    avg_hd_value = 0
    avg_emd_value = 0
    counter = 0
    pred_paths = glob(os.path.join(PRED_DIR, "*.xyz"))
    gt_pred_pairs = []
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert (ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))
    torch.set_printoptions(precision=6)

    with open(os.path.join(save_path, "evaluation.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()
        for gt_path, pred_path in gt_pred_pairs:
            row = {}
            gt = io.load(gt_path)[:, :3]
            gt = gt[np.newaxis, ...]
            gt, centroid, furthest_distance = operations.normalize_point_cloud(gt)

            gt = torch.from_numpy(gt).to(device=DEVICE)

            pred = io.load(pred_path)
            pred = pred[:, :3]

            row["name"] = os.path.basename(pred_path)
            pred = pred[np.newaxis, ...]

            pred, centroid, furthest_distance = operations.normalize_point_cloud(pred)

            pred = torch.from_numpy(pred).to(device=DEVICE)

            cd_forward_value, cd_backward_value = cd_dist_compute(pred, gt)
            cd_forward_value = np.array(cd_forward_value.cpu())
            cd_backward_value = np.array(cd_backward_value.cpu())

            md_value = np.mean(cd_forward_value) + np.mean(cd_backward_value)
            hd_value = np.max(np.amax(cd_forward_value, axis=1) + np.amax(cd_backward_value, axis=1))
            cd_backward_value = np.mean(cd_backward_value)
            cd_forward_value = np.mean(cd_forward_value)

            row["CD"] = cd_forward_value + cd_backward_value
            row["hausdorff"] = hd_value
            avg_md_forward_value += cd_forward_value
            avg_md_backward_value += cd_backward_value
            avg_hd_value += hd_value

            writer.writerow(row)
            counter += 1

        row = OrderedDict()
        avg_md_forward_value /= counter
        avg_md_backward_value /= counter
        avg_hd_value /= counter
        avg_emd_value /= counter
        avg_cd_value = avg_md_forward_value + avg_md_backward_value
        row["CD"] = avg_cd_value
        row["hausdorff"] = avg_hd_value
        row["EMD"] = avg_emd_value

        writer.writerow(row)

        row = OrderedDict()
        row["CD (1e-3)"] = avg_cd_value * 1000.
        row["hausdorff (1e-3)"] = avg_hd_value * 1000.
        return avg_cd_value * 1000., avg_hd_value * 1000.


def online_evaluation(PRED_DIR, GT_DIR, save_path=None):
    DEVICE = torch.device('cuda', 0)

    gt_paths = glob(os.path.join(GT_DIR, '*.xyz'))

    gt_names = [os.path.basename(p)[:-4] for p in gt_paths]
    # gt = load(gt_paths[0])[:, :3]
    cd_dist_compute = CD_dist()

    avg_hd_value = 0
    total_cd_values = 0.0
    counter = 0
    pred_paths = glob(os.path.join(PRED_DIR, "*.xyz"))
    gt_pred_pairs = []
    for p in pred_paths:
        name, ext = os.path.splitext(os.path.basename(p))
        assert (ext in (".ply", ".xyz"))
        try:
            gt = gt_paths[gt_names.index(name)]
        except ValueError:
            pass
        else:
            gt_pred_pairs.append((gt, p))
    torch.set_printoptions(precision=6)

    evaluation_results_list = []
    for gt_path, pred_path in gt_pred_pairs:
        row = {}
        gt = io.load(gt_path)[:, :3]
        gt = gt[np.newaxis, ...]
        gt, centroid, furthest_distance = operations.normalize_point_cloud(gt)

        gt = torch.from_numpy(gt).to(device=DEVICE)

        pred = io.load(pred_path)
        pred = pred[:, :3]

        row["name"] = os.path.basename(pred_path)
        pred = pred[np.newaxis, ...]

        pred, centroid, furthest_distance = operations.normalize_point_cloud(pred)

        pred = torch.from_numpy(pred).to(device=DEVICE)

        cd_forward_value, cd_backward_value = cd_dist_compute(pred, gt)
        cd_forward_value = np.array(cd_forward_value.cpu())
        cd_backward_value = np.array(cd_backward_value.cpu())

        hd_value = np.max(np.amax(cd_forward_value, axis=1) + np.amax(cd_backward_value, axis=1))
        cd_backward_value = np.mean(cd_backward_value)
        cd_forward_value = np.mean(cd_forward_value)

        row["CD"] = 1000.0 * (cd_forward_value + cd_backward_value)
        row["hausdorff"] = 1000.0 * hd_value

        total_cd_values += row["CD"]
        avg_hd_value += row["hausdorff"]

        counter += 1
        evaluation_results_list.append(row)

    avg_cd_value = total_cd_values / counter
    avg_hd_value /= counter

    if save_path is not None:
        save_fp = os.path.join(save_path, "0-evaluation.txt")
        fp = open(save_fp, 'w')

        ## the header
        fp.write(f"{'Name' : >45} \t CD(1e-3) \t hausdorff(1e-3)\n")

        ## the avg results
        fp.write(f"{'Average' : >45} \t {avg_cd_value} {avg_hd_value}\n\n")

        ## Sort the results
        evaluation_results_list = sorted(evaluation_results_list, key=lambda x: x['CD'], reverse=True)

        for line in evaluation_results_list:
            line_str = f"{line['name'] : >45} \t {line['CD']} {line['hausdorff']}"
            fp.write(line_str + '\n')
        fp.close()

    return avg_cd_value, avg_hd_value