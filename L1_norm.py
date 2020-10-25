# -*- coding: utf-8 -*-
"""Functions for least squares estimation using L1 norm"""
import numpy as np
from costs import *

def compute_gradient_L1(y, tx, w, lambda_):
    """Compute the gradient."""
    
    e = (y - tx.dot(w))
    return -(tx.transpose().dot(e) / y.shape[0]) + lambda_*np.sign(w)

def least_squares_L1(y, tx, initial_w, max_iters, gamma, lambda_):
    """Gradient descent algorithm."""

    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        g = compute_gradient_L1(y, tx, w, lambda_)
        loss = compute_mse(y, tx, w)

        # update w by gradient
        w = w - gamma*g

        # store w and loss
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss