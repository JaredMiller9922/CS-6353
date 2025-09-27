import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on mini batches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a mini batch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # Note: Both partial derivatives only have an effect when the margin is greater than 0
      if margin > 0:
        loss += margin
        dW[:, j] += X[i] + np.dot(W[j],W[j]) * 2 * reg
        dW[:, y[i]] -= X[i] + np.dot(W[i], W[i]) * 2 * reg

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # We must also average the dW because if the loss is averaged that means we also have a 1/N in the gradient as well

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # SEE ABOVE CODE
  #####################################################################
  #                       END OF YOUR CODE                            #
  #####################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  loss = 0.0
  scores = X @ W 

  # Find a vectorized way to get the correct class for each row (example)
  correct_scores = scores[np.arange(scores.shape[0]), y].reshape(-1, 1)
  margins = np.maximum(0, scores - correct_scores + 1)

  # We don't want the correct classes to contribute to the margins
  margins[np.arange(margins.shape[0]), y] = 0

  # Update true classes gradients
  violation_count = np.count_nonzero(margins, axis = 1)
  dW[:, y] -= (violation_count.reshape(-1,1) * X).T

  # Update non true classes gradients
  incorrect_classes = margins > 0
  print("shape of incorect classes")
  print(incorrect_classes.shape)
  print("shape of incorect classes")
  print(dW.shape)
  dW[incorrect_classes] += X 
 

  loss = np.sum(margins)
  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  
  dW /= X.shape[0]
  dW += reg * 2 * np.dot(W,W)
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
