# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    lambda_prime = 2*lambda_*y.shape[0]
    gram = tx.transpose().dot(tx)
    w = (np.linalg.inv(gram + lambda_prime*np.eye(gram.shape[0]))).dot(tx.transpose()).dot(y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss
