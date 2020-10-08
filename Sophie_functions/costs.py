# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx.dot(w)
    lv = 0.5*np.mean(np.square(e))
    return lv