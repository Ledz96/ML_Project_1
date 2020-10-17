# -*- coding: utf-8 -*-
"""logit functions"""
import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    z = 1/(1 + np.exp(-t))
    return z

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    L=np.sum(np.log(1+np.exp(tx@w)) - y*tx@w)
    return L

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    grad=tx.T@(sigmoid(tx@w)-y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """

    loss=calculate_loss(y,tx,w)
    grad=calculate_gradient(y,tx,w)
    w=w-gamma*calculate_gradient(y,tx,w)

    return loss, w



def logistic_regression(y,tx,initials_w,max_iters,gamma):
    w=initials_w
    losses = []
    threshold = 1e-8
    
    for i in range(max_iters):
        loss, w= learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def logistic_regression_Sgd(y,tx,initials_w,max_iters,gamma,batchsize=1):
    w=initials_w
    losses = []
    threshold = 1e-8
    
    for i in range(max_iters):
        loss, w= stochastic_gradient_descent(
        y, tx, initials_w, max_iters, gamma, batchsize)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

def logistic_accuracy(y, tx, w):
    t=tx@w
    n=y.shape[0]
    proba=sigmoid(t)
    ypred=(proba>0.5).astype(int)
    acc=(y==ypred).sum()/n
    
    return acc
        


def stochastic_gradient_descent(
        y, tx, initials_w, max_iters, gamma,batch_size=1):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    ws = [initials_w]
    losses = []
    w = initials_w
    n=batch_size
    for i in range(max_iters):
        indexn=np.random.choice(y.size, n)
        yn=y[indexn]
        txn=tx[indexn]
        loss=calculate_loss(y, tx, w)
        grad=calculate_gradient(y, tx, w)

        w=w-grad*gamma

     

    return loss, w