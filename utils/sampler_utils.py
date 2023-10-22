'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-15 22:18:13
Email: haimingzhang@link.cuhk.edu.cn
Description: Borrowed from PUCRN.
'''

import numpy as np
import torch
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


def __batch_distance_matrix_general(A, B):
    """
    :param
        A, B [B,N,C], [B,M,C]
    :return
        D [B,N,M]
    """
    r_A = torch.sum(A * A, dim=2, keepdim=True)
    r_B = torch.sum(B * B, dim=2, keepdim=True)
    m = torch.matmul(A, B.permute(0, 2, 1))
    D = r_A - 2 * m + r_B.permute(0, 2, 1)
    return D


def group_knn(k, query, points, unique=True, NCHW=True):
    """
    group batch of points to neighborhoods
    :param
        k: neighborhood size
        query: BxCxM or BxMxC
        points: BxCxN or BxNxC
        unique: neighborhood contains *unique* points
        NCHW: if true, the second dimension is the channel dimension
    :return
        neighbor_points BxCxMxk (if NCHW) or BxMxkxC (otherwise)
        index_batch     BxMxk
        distance_batch  BxMxk
    """
    if NCHW:
        batch_size, channels, num_points = points.size()
        points_trans = points.transpose(2, 1).contiguous()
        query_trans = query.transpose(2, 1).contiguous()
    else:
        points_trans = points.contiguous()
        query_trans = query.contiguous()

    batch_size, num_points, _ = points_trans.size()
    assert (num_points >= k), "points size must be greater or equal to k"

    with torch.no_grad():
        D = __batch_distance_matrix_general(query_trans, points_trans)
    if unique:
        # prepare duplicate entries
        points_np = points_trans.detach().cpu().numpy()
        indices_duplicated = np.ones((batch_size, 1, num_points), dtype=np.int32)

        for idx in range(batch_size):
            _, indices = np.unique(points_np[idx], return_index=True, axis=0)
            indices_duplicated[idx, :, indices] = 0

        indices_duplicated = torch.from_numpy(indices_duplicated).to(device=D.device, dtype=torch.float32)
        D += torch.max(D) * indices_duplicated

    # (B,M,k)
    distances, point_indices = torch.topk(-D, k, dim=-1, sorted=True)
    # (B,N,C)->(B,M,N,C), (B,M,k)->(B,M,k,C)
    knn_trans = torch.gather(
        points_trans.unsqueeze(1).expand(-1, query_trans.size(1), -1, -1), 2,
        point_indices.unsqueeze(-1).expand(-1, -1, -1, points_trans.size(-1)))

    if NCHW:
        knn_trans = knn_trans.permute(0, 3, 1, 2)

    return knn_trans, point_indices, -distances


def fps_subsample(xyz, npoint, NCHW=False):
    """
    :param
        xyz (B, 3, N) or (B, N, 3)
        npoint a constant
    :return
        torch.LongTensor
            (B, npoint) tensor containing the indices
        torch.FloatTensor
            (B, npoint, 3) or (B, 3, npoint) point sets"""
    assert (xyz.dim() == 3), "input for furthest sampling must be a 3D-tensor, but xyz.size() is {}".format(xyz.size())
    # need transpose
    if NCHW:
        xyz = xyz.transpose(2, 1).contiguous()  # to (B, N, 3)

    assert (xyz.size(2) == 3), "furthest sampling is implemented for 3D points"
    idx = furthest_point_sample(xyz, npoint)
    sampled_pc = gather_operation(xyz.permute(0, 2, 1).contiguous(), idx)

    if not NCHW:
        sampled_pc = sampled_pc.transpose(2, 1).contiguous()
    return idx, sampled_pc


def knn_sample(pc, num_group=24, group_size=1024, centralize=False):
    """KNN sampling the input point cloud

    Args:
        pc (Tensor): (B, N, 3), should be in GPU device
        num_group (int, optional): how many patches you want to obtain. Defaults to 24.
        group_size (int, optional): for each patch, how many k-nearest points you want to get. Defaults to 1024.

    Returns:
        neighborhood(Tensor): (B, G, M, 3), G==num_group, M==group_size
        center(Tensor): (B, G, 3)
    """
    # fps the centers out
    idx, center = fps_subsample(pc, num_group)  # B G 3
    neighborhood, _, _ = group_knn(group_size, center, pc, NCHW=False)  # B G M 3

    if centralize:
        neighborhood = neighborhood - center.unsqueeze(2)
    return neighborhood, center