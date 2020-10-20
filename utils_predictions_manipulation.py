# -*- coding: utf-8 -*-
"""Utils functions for handling predictions"""

import numpy as np

def probability_to_prediction(y):
    """given an array of probabilities y, returns a prediction (1 or -1) for each"""
    
    replacer = lambda p: 1 if p > 0.5 else -1
    vfunc = np.vectorize(replacer)
    return vfunc(y)

def get_prediction_accuracy(y, pred_y):
    """given a true y and a prediction, gets the predition accuracy"""
    
    return (np.array(y == pred_y)).sum()/len(y)
