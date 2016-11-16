__author__ = 'Zhiwei Jia'

import numpy as np
import numpy.random as rand
import random
from numpy import *
import theano
from theano import *
import theano.tensor as T

class NN:

    def __init__(self, shape):
        self.max_accuracy = 0.0
        self.shape = shape
        self.num_layers = len(shape)
        self.biases = [shared(np.array([0.0 for i in range(b_len)]).astype(np.float32)) for b_len in shape[1:]]
        self.weights = [shared((np.sqrt(2.0 / (pre+nex)) * rand.randn(pre, nex)).astype(np.float32))
                        for pre, nex in zip(shape[:-1], shape[1:])]

        self.mom_biases = [shared(np.array([0.0 for i in range(b_len)]).astype(np.float32)) for b_len in shape[1:]]
        self.mom_weights = [shared(np.zeros((pre, nex), np.float32)) for pre, nex in zip(shape[:-1], shape[1:])]

    def forward(self, data, stable_version=False):
        """input has each row as data vector; output also does so"""
        count = 1
        for bias, weight in zip(self.biases, self.weights):
            if count < self.num_layers - 1:
                data = T.nnet.relu(T.dot(data, weight) + bias)
            elif not stable_version:
                data = T.nnet.softmax(T.dot(data, weight) + bias)
            else:
                data = log_softmax(T.dot(data, weight) + bias)
            count += 1
        return data

    def predict(self, data):
        """input has each row as data vector; output is a single array indicating prediction"""
        return T.argmax(self.forward(data), axis=1)

    def train(self, train_x, train_y, test_x, test_y, max_epoch=500, mini_batch_size=100, learning_rate=0.1,
              momentum=0.9, reg=0.001):
        """train_x is a num_sample by num_feature matrix; train_y is a num_sample by num_class matrix"""

        # first create update functions for each mini-batch-learning
        X, Y = T.fmatrix('X'), T.fmatrix('Y')
        D_weights, D_biases = self.get_derivative(X, Y)
        prev_mom_biases = [shared(np.array([0.0 for i in range(b_len)]).astype(np.float32)) for b_len in self.shape[1:]]
        prev_mom_weights = [shared(np.zeros((pre, nex), np.float32))
                            for pre, nex in zip(self.shape[:-1], self.shape[1:])]
        updatePrevMomList = [(w0, w1) for w0, w1 in zip(prev_mom_weights, self.mom_weights)]
        updatePrevMomList += [(b0, b1) for b0, b1 in zip(prev_mom_biases, self.mom_biases)]
        updateCurrMomList = [(w0, momentum * w0 - learning_rate * (w1 + reg * w2))
                             for w0, w1, w2 in zip(self.mom_weights, D_weights, self.weights)]
        updateCurrMomList += [(b0, momentum * b0 - learning_rate * b1) for b0, b1 in zip(self.mom_biases, D_biases)]
        updateModelParamsList = [(w0, w0 - momentum * w1 + (1 + momentum) * w2) for w0, w1, w2 in
                                 zip(self.weights, prev_mom_weights, self.mom_weights)]
        updateModelParamsList += [(b0, b0 - momentum * b1 + (1 + momentum) * b2) for b0, b1, b2 in
                                  zip(self.biases, prev_mom_biases, self.mom_biases)]
        updatePrevMom = function([], updates=updatePrevMomList)
        updateCurrMom = function([X, Y], updates=updateCurrMomList)
        updateModelParams = function([], updates=updateModelParamsList)

        # and then the function for evaluate the training result
        correctness = function([X, Y], self.evaluate(X, Y))

        # start training
        for i in range(max_epoch):
            train_x, train_y = shuffle_union(train_x, train_y)
            for j in range(0, train_x.shape[0], mini_batch_size):
                updatePrevMom()
                updateCurrMom(train_x[j: j + mini_batch_size, :], train_y[j: j + mini_batch_size, :])
                updateModelParams()

            # show correctness info.
            train_correctedness = 100.0 * correctness(train_x, train_y) / train_x.shape[0]
            test_correctedness = 100.0 * correctness(test_x, test_y) / test_x.shape[0]
            print "Epoch {0}: on train_data, {1} % are correct".format(i + 1, train_correctedness)
            print "    on test_data: {0} % are correct".format(test_correctedness)
            if self.max_accuracy < test_correctedness:
                self.max_accuracy = test_correctedness
            print "Current best test accuracy", self.max_accuracy

    def get_derivative(self, x, y):
        predicted_y_lsm = self.forward(x, stable_version=True)
        E_xen = T.mean(log_softmax_crossentropy(predicted_y_lsm, y))
        D_weights = [T.grad(E_xen, weight) for weight in self.weights]
        D_biases = [T.grad(E_xen, bias) for bias in self.biases]
        return D_weights, D_biases

    def evaluate(self, x, y):
        """return the number of correct prediction givin input x and true labels y"""
        return T.sum(T.eq(self.predict(x), T.argmax(y, axis=1)))

def shuffle_union(x, y):
    arr = range(x.shape[0])
    random.shuffle(arr)
    return x[arr, :], y[arr, :]

def log_softmax(x):
    # numerically stable log-softmax
    xdev = x - x.max(1, keepdims=True)
    lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
    return lsm

def log_softmax_crossentropy(predicted_y_lsm, y):
    # numerically stable crossentropy for log-softmax
    return -T.sum(y * predicted_y_lsm, axis=1)
