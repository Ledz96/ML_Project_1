# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
from build_polynomial import *
import matplotlib.pyplot as plt


def compute_loss_MSE(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx.dot(w)
    lv = np.mean(np.square(e))
    return lv

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    tx_t = tx.transpose()
    dl = -1/len(tx)*tx_t.dot(e)
    return dl

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute gradient and loss
        dl = compute_gradient(y,tx,w)
        loss = compute_loss_MSE(y,tx,w)
        
        # Update w by gradient
        w = w - gamma*dl
        
        # Store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws