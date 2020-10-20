# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    return 0.5*np.mean(np.square(e))
    