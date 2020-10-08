# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    lv = 0.5*np.mean(np.square(e))
    
    return lv

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    
    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array([w0[i],w1[j]])
            losses[i,j] = compute_loss(y,tx,w)
        
    
    return losses

