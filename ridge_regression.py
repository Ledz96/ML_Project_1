# -*- coding: utf-8 -*-
"""ridge regression"""

import numpy as np
from costs import*

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    XX = np.dot(tx.transpose(),tx)
    I = np.identity(XX.shape[0])
    XX = XX + lambda_*I
    B = np.dot(tx.transpose(),y)
    if len(XX.shape)==0:
        w_s = B/XX
    else:
        w_s = np.linalg.solve(XX,B)
    
    loss = np.sqrt(2*compute_mse(y, tx, w_s))
    return w_s, loss