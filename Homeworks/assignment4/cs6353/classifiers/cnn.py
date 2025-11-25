import numpy as np

from cs6353.layers import *
from cs6353.fast_layers import *
from cs6353.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # H' = 1 + (H + 2pad - HH) / stride = N, num_filters, 32,32
    # Pooling layer divides by 2 
    # N, num_filters, 16, 16
    # Flatten to get w2
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(scale = weight_scale, size = (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(scale = weight_scale, size = ((num_filters * H//2 * W//2, hidden_dim)))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': int((filter_size - 1) / 2)}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    l1_out, l1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    l2_out, l2_cache = affine_relu_forward(l1_out, W2, b2)
    l3_out, l3_cache = affine_forward(l2_out, W3, b3)
    scores = l3_out


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    s_l, s_dx = softmax_loss(l3_out, y)
    r_l = self.reg * 0.5 * (np.sum(self.params['W1']**2) + 
                            np.sum(self.params['W2']**2) + 
                            np.sum(self.params['W3']**2))

    r_dx_w1 = self.reg * self.params['W1']
    r_dx_w2 = self.reg * self.params['W2']
    r_dx_w3 = self.reg * self.params['W3']

    loss = s_l + r_l
    
    l3_dx, l3_dw, l3_db = affine_backward(s_dx, l3_cache)
    l2_dx, l2_dw, l2_db = affine_relu_backward(l3_dx, l2_cache)
    l1_dx, l1_dw, l1_db = conv_relu_pool_backward(l2_dx, l1_cache)

    grads['W1'] = r_dx_w1 + l1_dw
    grads['b1'] = l1_db

    grads['W2'] = r_dx_w2 + l2_dw
    grads['b2'] = l2_db

    grads['W3'] = r_dx_w3 + l3_dw
    grads['b3'] = l3_db


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads