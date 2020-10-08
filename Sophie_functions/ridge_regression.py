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
    p=lambda_*(2*y.size)
    weights= np.linalg.inv(tx.T@tx+p*np.identity(tx.shape[1]))@tx.T@y 
    return weights