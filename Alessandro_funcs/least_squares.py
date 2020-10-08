# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    
    gram = tx.transpose().dot(tx)
    w = (np.linalg.inv(gram)).dot(tx.transpose()).dot(y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss
