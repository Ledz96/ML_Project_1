# -*- coding: utf-8 -*-
"""Utils functions for standardizing data and build poly"""

import numpy as np

def standardize(x):
    """Standardize the original data set ignoring column of 1."""
    xnew=x
    subx=x[:,1:]
    mean_x = np.mean(subx, axis=0)
    subx = subx - mean_x
    substd_x = np.std(subx, axis=0)
    subx = subx / substd_x
    xnew[:,1:]=subx
    return xnew, mean_x, substd_x

def build_poly_multi(tx, degree):
    """polynomial basis functions for input data tx, for all features expanded to d degree.Extended matrix is inputed np.array"""
    # ***************************************************
    # augmented x matrix with column of 1 input
    #list of indices for features modified
    #degrees of polynomial to apply to all features irrespectively
    # ***************************************************
    xmat=np.ones(tx.shape[0]).reshape(-1,1)
    for i in range(1,tx.shape[1]):
        for d in range(1,degree+1):
            coltmp=tx[:,i]**d
            xmat = np.append(xmat, coltmp.reshape(-1,1), axis=1)

        
    return np.array(xmat)

def build_poly_index(tx, index_list, degree):
    """polynomial basis functions for input data tx, for all features expanded to d degree.Extended matrix is inputed np.array. Index list includes all variables that will be expanded, mostly useful to avoid expanding categorical variables"""
    # ***************************************************
    # augmented x matrix with column of 1 input
    #list of indices for features modified
    #degrees of polynomial to apply to all features irrespectively
    # ***************************************************
    xmat=np.ones(tx.shape[0]).reshape(-1,1)
    for i in range(1,tx.shape[1]):
        if i in index_list:
            for d in range(1,degree+1):
                coltmp=tx[:,i]**d
                xmat = np.append(xmat, coltmp.reshape(-1,1), axis=1)
        if i not in index_list:
            coltmp=tx[:,i]
            xmat = np.append(xmat, coltmp.reshape(-1,1), axis=1)
    return np.array(xmat)

def min_max_scale(x):
    """Compute min-max scaling on x and return the scaled matrix"""
    
    x_min = (x.min(axis=0))
    x_max = (x.max(axis=0))

    return (x - x_min) / (x_max - x_min)