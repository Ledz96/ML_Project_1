# -*- coding: utf-8 -*-
"""reg logit functions"""
import numpy as np

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and Hessian."""

    loss = calculate_loss(y, tx, w)
    g = calculate_gradient(y, tx, w) + lambda_*w
    
    return loss, g

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
 
    # return loss, gradient
    loss, g = penalized_logistic_regression(y, tx, lambda_)

    # update w
    w = w - gamma*g
    
    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        
        # log info
        #if iter % 100 == 0:
         #   print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, losses[-1]