'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-07 23:07:31
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import numpy as np


def load(filename, count=None):
    if filename.endswith(".xyz"):
        points = np.loadtxt(filename).astype(np.float32)
        if count is not None:
            if count > points.shape[0]:
                # fill the point clouds with the random point
                tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
                tmp[:points.shape[0], ...] = points
                tmp[points.shape[0]:, ...] = points[np.random.choice(points.shape[0], count - points.shape[0]), :]
                points = tmp
            elif count < points.shape[0]:
                raise NotImplementedError
    else:
        raise NotImplementedError
    return points
