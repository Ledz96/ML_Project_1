# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    XX = np.dot(tx.transpose(),tx)
    I = np.identity(XX.shape[0])
    XX = XX + lambda_*I
    
    B = np.dot(tx.transpose(),y)
    
    w_s = np.linalg.solve(XX,B)
    return w_s