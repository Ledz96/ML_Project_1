# -*- coding: utf-8 -*-
"""mapping function names to functions"""

import numpy as np

from least_squares_gd import*
from least_squares_sgd import*
from least_squares import*
from ridge_regression import*
from logistic_regression import*
from reg_logistic_regression import*

def get_function(argument):
    switcher = {
        'gd': least_squares_GD,
        'sgd': least_squares_SGD,
        'least_squares': least_squares,
        'ridge_regression': ridge_regression,
        'logistic_regression': logistic_regression,
        'reg_logistic_regression': reg_logistic_regression
    }
    
    f = switcher.get(argument, lambda: "Incorrect function name")
    return f