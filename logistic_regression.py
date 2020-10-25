# -*- coding: utf-8 -*-
"""logit functions"""
import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    
    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""

    return np.sum(-y*tx.dot(w) + np.log(1 + np.exp(tx.dot(w))))/len(y)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    
    return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y)/len(y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
            
    # compute gradient and loss    
    g = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)

    # update w by gradient
    w = w - gamma*g

    return loss, w

def logistic_regression(y, tx, initial_w, max_iters, gamma, print_ = False):
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
        # log info
        #if iter % 100 == 0:
        if print_:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w, loss