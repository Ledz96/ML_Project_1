# -*- coding: utf-8 -*-
"""functions to load and structure data"""

import csv
import numpy as np

def load_data(filepath):
    dtypes = "i8,S5,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,i8,f8,f8,f8,f8,f8,f8,f8"
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
    return np.genfromtxt('Data/train.csv', delimiter=",",names=True, dtype=dtypes), header

def structure_data(data):
    # Building y
    Y_total = np.array((data['Prediction']==b's').astype(int))

    # Filtering out useless data from x
    names = list(data.dtype.names)[2:]
    data_filtered = data[names]

    # Building x
    x = np.empty((len(data_filtered),len(data_filtered[0].item())))
    for i in range(len(data_filtered)):
        x[i] = np.array(data_filtered[i].item()).reshape(1,-1)

    # Adding 1s at the beginning
    X_total = np.c_[np.ones(len(x)),x]
    
    return X_total, Y_total