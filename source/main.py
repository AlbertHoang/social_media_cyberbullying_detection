import numpy as np
import random

from rnn import RNN
from data import train_data, test_data
from maths import softmax
from lstm import LSTM


# Create the vocabulary.
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)
print(vocab)

# Assign indices to each word.
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}


# print(word_to_idx['good'])
# print(idx_to_word[0])

def createInputs(text):
  '''
  Returns list of 2d arrays of one-hot vectors representing the words in the input text string.
  - text is a string
  - Each one-hot vector has shape (vocab_size, 1)
  '''
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs


# Initialize our RNN!

def initiateRNN():
  rnn = RNN(vocab_size, 2)
  rnnTrain(rnn, 1000)


def rnnProcessData(rnn, data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)


def rnnTrain(rnn, numLoop):
  '''
  Train data and save model after training
  :param numLoop: number of training loop
  :return:
  '''
  # Training loop
  for epoch in range(numLoop):
    train_loss, train_acc = rnnProcessData(rnn, train_data)

    if epoch % 100 == 99:
      print('--- Epoch %d' % (epoch + 1))
      print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

      test_loss, test_acc = rnnProcessData(rnn, test_data, backprop=False)
      print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

  # Save model
  rnn.saveModel()


def initiateLSTM():
  lstm = LSTM(vocab_size, 2)
  lstmTrain(lstm,400)

def lstmProcessData(lstm, data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    # Forward
    out = lstm.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      lstm.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)

def lstmTrain(lstm, numLoop):
  '''
  Train data and save model after training
  :param numLoop: number of training loop
  :return:
  '''
  # Training loop
  for epoch in range(numLoop):
    train_loss, train_acc = lstmProcessData(lstm, train_data)

    if epoch % 100 == 99:
      print('--- Epoch %d' % (epoch + 1))
      print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

      test_loss, test_acc = lstmProcessData(lstm, test_data, backprop=False)
      print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

  # Save model

initiateLSTM()