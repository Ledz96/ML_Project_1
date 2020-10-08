# -*- coding: utf-8 -*-

import numpy as np

"""Function used to compute the loss."""


def compute_loss_mse(y, tx, w):
    return np.sum((y - tx.dot(w.transpose())) ** 2, axis=0) / (y.shape[0])


def compute_loss_mae(y, tx, w):
    return np.sum(abs(y - tx.dot(w)), axis=0) / (y.shape[0])


def compute_loss_rmse(y, tx, w):
    return np.sqrt(2 * compute_loss_mse(y, tx, w))
