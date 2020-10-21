import numpy as np

from utils_predictions_manipulation import *
from utils_function_mapper import*

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_fold, function_name, lambda_=0, max_iters=0, gamma=0, seed=1):
    """return the loss of ridge regression."""

    #rmse_tr = []
    #rmse_te = []
    losses_tr = []
    losses_te = []
    acc_te = []
    acc_tr = []
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    initial_w = np.ones((x.shape[1]))*(-0.01)
        
    # get k'th subgroup in test, others in train
    for i in range(k_fold):
        x_tr = np.empty((0,x.shape[1]))
        y_tr = np.empty((0))
    
        # test data is taken from k'th (i) group
        x_te = x[k_indices[i]]
        y_te = y[k_indices[i]]
        
        # all the other subgroups are in train data
        for j in range(k_fold):
            if j != i:
                x_tr = np.r_[(x_tr, x[k_indices[j]])]
                y_tr = np.r_[(y_tr, y[k_indices[j]])]
                        
        # form data with polynomial degree
        #x_tr = build_poly(x_tr, degree)
        #x_te = build_poly(x_te, degree)
    
        # select function to execute
        f = get_function(function_name)
        
        if function_name == 'least_squares':
            w,loss = f(y_tr, x_tr)
        elif function_name == 'reg_logistic_regression':
            w,loss = f(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
        elif function_name == 'ridge_regression':
            w,loss = f(y_tr, x_tr, lambda_)
        else:
            w,loss = f(y_tr, x_tr, initial_w, max_iters, gamma)
    
        # calculate the error for train and test data
        #rmse_tr.append(2*compute_mse(y_tr, x_tr, w))
        #rmse_te.append(2*compute_mse(y_te, x_te, w))
        
        # calculate predictions for train and test data
        y_tr_prb = x_tr.dot(w)
        y_te_prb = x_te.dot(w)
        
        # calculate accuracy for train and test data
        y_tr_pr = probability_to_prediction(y_tr_prb)
        y_te_pr = probability_to_prediction(y_te_prb)
        y_te = probability_to_prediction(y_te)
        y_tr = probability_to_prediction(y_tr)
        
        # getting accuracy
        acc_tr.append(get_prediction_accuracy(y_tr, y_tr_pr))
        acc_te.append(get_prediction_accuracy(y_te, y_te_pr))
        
    return np.mean(acc_tr), np.mean(acc_te)