# -*- coding: utf-8 -*-
"""SGD functions"""

import numpy as np
from costs import*

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    return - tx.transpose().dot(e) / y.shape[0]

def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Stochastic gradient descent algorithm."""
    
    threshold = 1e-8

    losses = []
    w = initial_w
    n=batch_size
    for n_iter in range(max_iters):
        indexn=np.random.choice(y.size, n)
        yn=y[indexn]
        txn=tx[indexn]
        loss=compute_mse(y, tx, w)
        grad=compute_stoch_gradient(y, tx, w)

        w=w-grad*gamma

        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break


    return w, loss