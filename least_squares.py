# -*- coding: utf-8 -*-
"""Least squares"""

import numpy as np
from costs import*

def least_squares(y, tx):
    """calculate the least squares solution."""
    XX = np.dot(tx.transpose(),tx)
    B = np.dot(tx.transpose(),y)
    if len(XX.shape)==0:
        w_s = B/XX
    else:
        w_s = np.linalg.solve(XX,B)
    
    loss = compute_mse(y, tx, w_s)
    
    return w_s, loss 

"""
def least_squares(y, tx):
    gram = tx.transpose().dot(tx)
    w = (np.linalg.inv(gram)).dot(tx.transpose()).dot(y)
    loss = compute_mse(y, tx, w)
    return w, loss
"""