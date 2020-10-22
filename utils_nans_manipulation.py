# -*- coding: utf-8 -*-
"""Utils functions for nan handling"""

import numpy as np
import random as rd


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

def replace_nans_with_median(x, threshold, seed=1):
    """given a dataset, replaces nans with the median for that column if nans/all_points <= threshold, deletes otherwise"""
    
    np.random.seed(seed)
    
    ret = np.empty((x.shape[0],0))
    dropped = []
    for col in range(x.shape[1]):
        if np.isnan(x[:,col]).sum() / x.shape[0] <= threshold:
            m = np.nanmedian(x[:,col])
            noise = m * (np.random.ranf()*2-1)/100
            m = m + noise
            nan_to_median = lambda p: p if not np.isnan(p) else m
            vfunc = np.vectorize(nan_to_median)
            ret = np.c_[ret, vfunc(x[:,col])]
        else:
            dropped.append(col)
    return ret, dropped

def delete_nans_indexes(x, nans_indexes):
    """returns a dataset without the columns in nans_indexes"""
    return np.delete(x, nans_indexes, axis=1)

def replace_test_nans_with_median(xtest, xtrain):
    """given a cleaned train and dirty test dataset, replace test nans with median for all values in that column"""
    
    for col in range(xtest.shape[1]):
        if np.isnan(xtest[:,col]).any():
            m = np.nanmedian(np.r_[xtrain[:,col], xtest[:,col]])
            nan_to_median = lambda p: p if not np.isnan(p) else m
            vfunc = np.vectorize(nan_to_median)
            xtest[:,col] = vfunc(xtest[:,col])
            
