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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  # For each training example in X
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    denominator = np.sum(np.exp(scores))
    loss += -1 * np.log(np.exp(correct_class_score) / denominator)

    for j in range(num_classes):
      s_j = np.exp(scores[j]) / denominator
      if j == y[i]:
        dW[:, y[i]] += X[i]*(s_j - 1) + np.dot(W[j],W[j]) * 2 * reg
      else:
        dW[:, j] += s_j * X[i] + np.dot(W[j],W[j]) * 2 * reg

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train

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
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(scores.shape[0]), y] 
  denominators = np.sum(np.exp(scores), axis = 1)
  loss_vec = -1 * np.log(np.exp(correct_class_scores) / denominators)
  loss = np.sum(loss_vec)
  loss /= X.shape[0]

  s_js = np.exp(scores) / denominators.reshape(-1, 1)

  # Noticing that in the gradient the only difference between the updates is that when 
  # j == y[i] we simply subtract 1 from s_j
  s_js[np.arange(s_js.shape[0]), y] -= 1
  dW = X.T @ s_js
  
  dW /= X.shape[0]
  dW += 2 * W * reg
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

