# -*- coding: utf-8 -*-
"""Utils functions for nan handling"""

import numpy as np

def replace_bad_data_with_nans(x, bad_data):
    """given a data matrix and a parameter for bad data, replaces all instances of that value with nan"""

    replacer = lambda p: p if p != bad_data else np.nan
    vfunc = np.vectorize(replacer)
    return vfunc(x)

def delete_nan_points(x,y):
    """given a dataset x and related labels y, returns points (x,y) whom's features contain no nans"""

    ret_x, ret_y = zip(*[(x_point,y_point) for x_point,y_point in zip(x,y) if not np.isnan(x_point).any()])
    return np.array(ret_x), np.array(ret_y)

def store_nan_points(x,y):
    """given a dataset x and related labels y, returns points (x,y) whom's features contain 1+ nans"""
    
    ret_x, ret_y = zip(*[(x_point,y_point) for x_point,y_point in zip(x,y) if np.isnan(x_point).any()])
    return np.array(ret_x), np.array(ret_y)

def replace_nans_with_median(x, threshold):
    """given a dataset, replaces nans with the median for that column if nans/all_points <= threshold, deletes otherwise"""
    
    ret = np.empty((x.shape[0],0))
    for col in range(x.shape[1]):
        if np.isnan(x[:,col]).sum() / x.shape[0] <= threshold:
            m = np.nanmedian(x[:,col])
            nan_to_median = lambda p: p if not np.isnan(p) else m
            vfunc = np.vectorize(nan_to_median)
            ret = np.c_[ret, vfunc(x[:,col])]
    return ret