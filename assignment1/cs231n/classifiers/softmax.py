import numpy as np
from random import shuffle

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
    
  num_train = X.shape[0]
  dp = np.zeros((num_train, W.shape[1]))
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  for i in range(num_train):
    scores = X[i].dot(W)
    # numeric stability
    scores -= np.max(scores)
    p = np.exp(scores) / np.sum(np.exp(scores))
    loss += -np.log(p[y[i]])
      
    # Mathematical expression : https://math.stackexchange.com/a/945918/359714    
    p[y[i]] -= 1
    dp[i] = p
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  
  dp /= num_train
  dW = np.dot(X.T, dp)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  dp = np.zeros((num_train, W.shape[1]))
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass    
  scores = X.dot(W)
  # numeric stability
  scores -= np.max(scores)
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
  # extract classes scores
  correct_p = -np.log(p[range(num_train), y])
  loss = np.sum(correct_p) / num_train
    
  # Mathematical expression : https://math.stackexchange.com/a/945918/359714    
  dp = p
  dp[range(num_train), y] -= 1
        

  loss += 0.5 * reg * np.sum(W * W)
  dp /= num_train
  dW = np.dot(X.T, dp)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

