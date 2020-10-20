# -*- coding: utf-8 -*-
"""Utils functions for standardizing data and build poly"""

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