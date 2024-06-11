from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range (X.shape[0]):
        scores = X[i].dot(W)
        scores = np.exp(scores)
        sum = 0
        for j in range(W.shape[1]):
            sum += scores[j]
        loss += -np.log(scores[y[i]]) + np.log(sum)

        dW[:, y[i]] -= X[i]
        for j in range(W.shape[1]):
            dW[:, j] += ((1/sum) * (scores[j])) * X[i]


    
    loss /= X.shape[0]
    dW /= X.shape[0]

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X @ W
    scoresEXP = np.exp(scores)
    sums = np.sum(scoresEXP, axis=1)
    correct = scoresEXP[np.arange(len(y)), y]
    correct_log = -np.log(correct)
    sums_log = np.log(sums)
    loss += np.sum(correct_log + sums_log)
    loss /= X.shape[0]
    loss += reg * np.sum(W * W)

    dloss = 1
    dloss /= X.shape[0]
    dcorrect_log = dloss * np.ones(X.shape[0])
    dsums_log = dloss * np.ones(X.shape[0])

    dcorrect = (-1/correct)  * dcorrect_log
    dsums = (1/sums) * dsums_log

    dscoresEXP = np.zeros((X.shape[0], W.shape[1]))
    dscoresEXP[np.arange(len(y)), y] += dcorrect
    dscoresEXP = (np.ones_like(dscoresEXP) * dsums[:, np.newaxis]) + dscoresEXP


    dscores = scoresEXP * dscoresEXP
    dW = X.T @ dscores
    dW += 2 * reg * W


    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
