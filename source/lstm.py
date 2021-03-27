import numpy as np
from numpy.random import randn
from maths import sigmoid, softmax

class LSTM:
  def __init__(self, input_size, output_size, hidden_size=18):
    '''
    :param input_size: x
    :param output_size: y
    :param hidden_size: h
    Wf -- Weight matrix of the forget gate, numpy array of shape (h, h + x)
    bf -- Bias of the forget gate, numpy array of shape (h, 1)
    Wi -- Weight matrix of the update gate, numpy array of shape (h, h + x)
    bi -- Bias of the update gate, numpy array of shape (h, 1)
    Wc -- Weight matrix of the first "tanh", numpy array of shape (h, h + x)
    bc --  Bias of the first "tanh", numpy array of shape (h, 1)
    Wo -- Weight matrix of the output gate, numpy array of shape (h, h + x)
    bo --  Bias of the output gate, numpy array of shape (h, 1)
    Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (y, h)
    by -- Bias relating the hidden-state to the output, numpy array of shape (y, 1)
    '''
    self.Wf = randn(hidden_size, hidden_size + input_size) / 1000
    self.bf = randn(hidden_size, 1) / 1000
    self.Wi = randn(hidden_size, hidden_size + input_size) / 1000
    self.bi = randn(hidden_size, 1) / 1000
    self.Wc = randn(hidden_size, hidden_size + input_size) / 1000
    self.bc = randn(hidden_size, 1) / 1000
    self.Wo = randn(hidden_size, hidden_size + input_size) / 1000
    self.bo = randn(hidden_size, 1) / 1000
    self.Wy = randn(output_size, hidden_size) / 1000
    self.by = randn(output_size, 1) / 1000

  def forward(self, inputs):
    '''
    Perform a forward pass of LSTM using the given inputs.
    :param inputs: is an array of one hot vectors with shape (input_size, 1).
    :return: final output and hidden state.
    '''
    n_y, n_h = self.Wy.shape

    h = np.zeros((n_h, 1))
    c = np.zeros((n_h, 1))
    y = np.zeros((n_y, 1))

    self.last_inputs = inputs
    self.last_hs = {0:h} # save hidden state
    self.last_cs = {0:c} # save cell state

    # Perform each step of the LSTM
    for i, x in enumerate(inputs):
      n_x, m = x.shape

      # Concatenate hidden state and input
      concat = np.zeros((n_x + n_h, 1))
      concat[: n_h, :] = h
      concat[n_h :, :] = x

      ft = sigmoid(np.dot(self.Wf, concat) + self.bf)
      it = sigmoid(np.dot(self.Wi, concat) + self.bi)
      cct = np.tanh(np.dot(self.Wc, concat) + self.bc)
      c = ft * c + it * cct
      self.last_cs[i+1] = c
      ot = sigmoid(np.dot(self.Wo, concat) + self.bo)
      h = ot*np.tanh(c)
      self.last_hs[i+1] = h

    y = np.dot(self.Wy, h) + self.by
    return y

  def backprop(self, d_y, learn_rate=2e-2):
    '''
    Perform a backward pass of the LSTM.
    :param d_y: (dL/dy) has shape (output_size, 1).
    :param learn_rate: is a float.
    :return:
    '''
    n = len(self.last_inputs)
    d_Wy = d_y @ self.last_hs[n].T
    d_y_d_h = self.Wy
