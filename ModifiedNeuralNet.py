__author__ = 'Zhiwei Jia'

import numpy as np
import numpy.random as rand
import random
from numpy import *
import theano
from theano import *
import theano.tensor as T
from theano.ifelse import ifelse

class NN:
    def __init__(self, shape, num_modifier):
        self.shape = shape
        self.num_layers = len(shape)
        self.num_modifier = num_modifier
        self.biases = [shared(np.array([0.0 for i in range(b_len)]).astype(np.float32)) for b_len in shape[1:]]
        self.weights = [shared((np.sqrt(2.0 / (pre+nex)) * rand.randn(pre, nex)).astype(np.float32))
                        for pre, nex in zip(shape[:-1], shape[1:])]
        self.pre_w = [[shared((0.1 * rand.randn(int(np.sqrt(s)), int(np.sqrt(s))) +
                               np.identity(int(np.sqrt(s)))).astype(np.float32))
                       for i in range(num_modifier)] for s in shape[:-1]]
        self.post_w = [[shared((0.1 * rand.randn(int(np.sqrt(s)), int(np.sqrt(s))) +
                                np.identity(int(np.sqrt(s)))).astype(np.float32))
                        for i in range(num_modifier)] for s in shape[:-1]]
        self.mom_weights = [shared(np.zeros((pre, nex), np.float32)) for pre, nex in zip(shape[:-1], shape[1:])]
        self.mom_biases = [shared(np.array([0.0 for i in range(b_len)]).astype(np.float32)) for b_len in shape[1:]]
        self.mom_pre_w = [[shared(np.zeros(((int(np.sqrt(s))), int(np.sqrt(s)))).astype(np.float32))
                           for i in range(num_modifier)] for s in shape[:-1]]
        self.mom_post_w = [[shared(np.zeros(((int(np.sqrt(s))), int(np.sqrt(s)))).astype(np.float32))
                            for i in range(num_modifier)] for s in shape[:-1]]

    def forward(self, data, stable_version=False):
        """input has each row as data vector; output also does so"""
        count = 1
        for bias, weight, pre_w, post_w in zip(self.biases, self.weights, self.pre_w, self.post_w):
            size = pre_w[0].shape[0]
            zeros_pre_w = T.zeros((size + 4, size + 4))
            zeros_post_w = T.zeros((size + 4, size + 4))
            pre_w_padding = T.set_subtensor(zeros_pre_w[2: size + 2, 2: size + 2], pre_w[0])
            post_w_padding_T = T.set_subtensor(zeros_post_w[2: size + 2, 2: size + 2], post_w[0])
            pre, updt = scan(process_pre_post_w, sequences=[pre_w_padding, zeros_pre_w])
            post_T, updt = scan(process_pre_post_w, sequences=[post_w_padding_T, zeros_post_w])
            pre, post_T = pre[2:size + 2, :], post_T[2:size + 2, :]
            ori_shape = data.shape
            data = T.reshape(data, (ori_shape[0], pre_w[0].shape[0], pre_w[0].shape[0]))
            product, updt = scan(lambda x, A, B: T.dot(T.dot(A, x), B), sequences=data, non_sequences=[pre, post_T.T])
            data = T.reshape(product, ori_shape)
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
              momentum=0.9, reg=0.001, extra_learning_rate=0.0001):
        """train_x is a num_sample by num_feature matrix; train_y is a num_sample by num_class matrix"""

        # first create update functions for each mini-batch-learning
        X, Y = T.fmatrix('X'), T.matrix('Y')
        D_weights, D_biases, D_pre_w, D_post_w = self.get_derivative(X, Y)

        # initialize previous momentum
        prev_mom_biases = [shared(np.array([0.0 for i in range(b_len)]).astype(np.float32)) for b_len in self.shape[1:]]
        prev_mom_weights = [shared(np.zeros((pre, nex), np.float32))
                            for pre, nex in zip(self.shape[:-1], self.shape[1:])]
        prev_mom_pre_w = [[shared(np.zeros(((int(np.sqrt(s))), int(np.sqrt(s)))).astype(np.float32))
                           for i in range(self.num_modifier)] for s in self.shape[:-1]]
        prev_mom_post_w = [[shared(np.zeros(((int(np.sqrt(s))), int(np.sqrt(s)))).astype(np.float32))
                            for i in range(self.num_modifier)] for s in self.shape[:-1]]

        # update list for previous momentum
        update_prev_mom_list = [(w0, w1) for w0, w1 in zip(prev_mom_weights, self.mom_weights)]
        update_prev_mom_list += [(b0, b1) for b0, b1 in zip(prev_mom_biases, self.mom_biases)]
        for i in range(self.num_layers - 1):
            update_prev_mom_list += [(w0, w1) for w0, w1 in zip(prev_mom_pre_w[i], self.mom_pre_w[i])]
            update_prev_mom_list += [(w0, w1) for w0, w1 in zip(prev_mom_post_w[i], self.mom_post_w[i])]

        # update list for current momentum
        update_curr_mom_list = [(w0, momentum * w0 - learning_rate * (w1 + reg * w2))
                                for w0, w1, w2 in zip(self.mom_weights, D_weights, self.weights)]
        update_curr_mom_list += [(b0, momentum * b0 - learning_rate * b1) for b0, b1 in zip(self.mom_biases, D_biases)]
        for i in range(self.num_layers - 1):
            update_curr_mom_list += [(w0, momentum * w0 - extra_learning_rate * w1)
                                     for w0, w1 in zip(self.mom_pre_w[i], D_pre_w[i])]
            update_curr_mom_list += [(w0, momentum * w0 - extra_learning_rate * w1)
                                     for w0, w1 in zip(self.mom_post_w[i], D_post_w[i])]

        # update list for actually changing model parameters
        update_model_params_list = [(w0, w0 - momentum * w1 + (1 + momentum) * w2) for w0, w1, w2 in
                                    zip(self.weights, prev_mom_weights, self.mom_weights)]
        update_model_params_list += [(b0, b0 - momentum * b1 + (1 + momentum) * b2) for b0, b1, b2 in
                                     zip(self.biases, prev_mom_biases, self.mom_biases)]
        for i in range(self.num_layers - 1):
            update_model_params_list += [(w0, w0 - momentum * w1 + (1 + momentum) * w2) for w0, w1, w2 in
                                         zip(self.pre_w[i], prev_mom_pre_w[i], self.mom_pre_w[i])]
            update_model_params_list += [(w0, w0 - momentum * w1 + (1 + momentum) * w2) for w0, w1, w2 in
                                         zip(self.post_w[i], prev_mom_post_w[i], self.mom_post_w[i])]

        # Theano functions
        update_prev_mom = function([], updates=update_prev_mom_list)
        update_curr_mom = function([X, Y], updates=update_curr_mom_list)
        update_model_params = function([], updates=update_model_params_list)

        # and then Theano function for evaluate the training result
        correctness = function([X, Y], self.evaluate(X, Y))

        # start training
        for i in range(max_epoch):
            train_x, train_y = shuffle_union(train_x, train_y)
            for j in range(0, train_x.shape[0], mini_batch_size):
                update_prev_mom()
                update_curr_mom(train_x[j: j + mini_batch_size, :], train_y[j: j + mini_batch_size, :])
                update_model_params()

            # show correctness info.
            train_correctedness = 100.0 * correctness(train_x, train_y) / train_x.shape[0]
            test_correctedness = 100.0 * correctness(test_x, test_y) / test_x.shape[0]
            print "Epoch {0}: on train_data, {1} % are correct".format(i + 1, train_correctedness)
            print "    on test_data: {0} % are correct".format(test_correctedness)

    def get_derivative(self, x, y):
        predicted_y_lsm = self.forward(x, stable_version=True)
        E_xen = T.mean(log_softmax_crossentropy(predicted_y_lsm, y))
        D_weights = [T.grad(E_xen, weight) for weight in self.weights]
        D_biases = [T.grad(E_xen, bias) for bias in self.biases]
        D_pre_w = [[T.grad(E_xen, pre_w[i]) for i in range(self.num_modifier)] for pre_w in self.pre_w]
        D_post_w = [[T.grad(E_xen, post_w[i]) for i in range(self.num_modifier)] for post_w in self.post_w]
        return D_weights, D_biases, D_pre_w, D_post_w

    def evaluate(self, x, y):
        """return the number of correct prediction givin input x and true labels y"""
        return T.sum(T.eq(self.predict(x), T.argmax(y, axis=1)))

def shuffle_union(x, y):
    arr = range(x.shape[0])
    random.shuffle(arr)
    return x[arr, :], y[arr, :]

def process_pre_post_w(padding_arr, zeros_arr):
    argmax = T.argmax(padding_arr)
    zeros_arr = ifelse(T.eq(padding_arr[argmax], 0), zeros_arr,
                       T.set_subtensor(zeros_arr[argmax-2:argmax+3], 1.5 / (T.sum(padding_arr[argmax-2:argmax+3]))))
    return_arr = (zeros_arr * padding_arr)[2: -2]
    return return_arr

def log_softmax(x):
    # numerically stable log-softmax
    xdev = x - x.max(1, keepdims=True)
    lsm = xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))
    return lsm

def log_softmax_crossentropy(predicted_y_lsm, y):
    # numerically stable crossentropy for log-softmax
    return -T.sum(y * predicted_y_lsm, axis=1)
