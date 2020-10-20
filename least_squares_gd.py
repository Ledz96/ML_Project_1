# -*- coding: utf-8 -*-
"""GD functions"""

import numpy as np
from costs import*

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    e = (y - tx.dot(w))
    return - tx.transpose().dot(e) / y.shape[0]

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        g = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        # update w by gradient
        w = w - gamma*g

        # store w and loss
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    return w, loss
