__author__ = 'Zhiwei Jia'

import numpy as np
from numpy import *
import numpy.matlib as mat
try :
    import pycuda.autoinit
    import pycuda.gpuarray as gpu
    import pycuda.cumath as gpum
    from pycuda import driver, compiler, tools
    print 'GPU mode ready!\n'
except ImportError:
    print 'PyCUDA not found! GPU mode not ready.\n'

class NeuralNet:

    def __init__(self, sizes=0, act=0, gpu_mod=False, num_thread_per_block=256):
        """the initialization function to create a new neural network with specified size"""
        self.gpu = gpu_mod
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.sqrt(2.0/(pre+nex)) * np.random.randn(nex, pre) for pre, nex in zip(sizes[:-1], sizes[1:])]
        self.act = act

        # these two are for momentum
        self.mom_b = [np.zeros((y, 1)) for y in sizes[1:]]
        self.mom_w = [np.zeros((nex, pre)) for pre, nex in zip(sizes[:-1], sizes[1:])]

        # these are for gpu mode
        self.kernel1 = None
        self.kernel2 = None
        self.kernel3 = None
        self.kernel4 = None
        self.kernel5 = None
        self.num_block1 = None
        self.num_block2 = None
        self.num_block3 = None
        self.num_block4 = None
        self.num_block5 = None
        self.num_thread_per_block = num_thread_per_block

    def load_from_file(self, load_file):
        """load learned parameter from local file"""
        mat = np.load(load_file + ".npy")
        self.sizes = mat[0]
        self.biases = mat[1]
        self.weights = mat[2]

    def save_to_file(self, save_file=None):
        """save learned parameters to local file"""
        mat = np.array([self.sizes, self.biases, self.weights])
        np.save(save_file, mat)

    def forward(self, a):
        """the activation function"""
        count = 0
        for biases, weights in zip(self.biases, self.weights):
            count += 1
            if count < self.num_layers - 1:
                if self.act == 0:
                    a = Sigmoid(np.dot(weights, a) + biases)
                elif self.act == 1:
                    a = Tanh(np.dot(weights, a) + biases)
                else:
                    a = ReLU(np.dot(weights, a) + biases)
        a = Softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a

    def training(self, train_data, test_data, max_epoch=100, mini_batch_size=30, learning_rate=0.1,
                 momentum=0, reg=0, num_comp_per_thread=None, num_comp_in_batch=None, store_file=None):
        """the training function which use mini-batch stochastic gradient descent to minimize the cost"""
        num_samples = len(train_data)
        num_tests = len(test_data)
        for i in range(max_epoch):
            random.shuffle(train_data)

            # prepare mini-batch data (for gpu mode)
            X = None
            Y = None
            if self.gpu:
                X, Y = separation(train_data)
                mini_batches = None
            else:
                mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, num_samples, mini_batch_size)]

            # gpu mode configurations
            if self.gpu:

                # check whether num_comp_per_thread is valid
                if num_comp_per_thread is None or len(num_comp_per_thread) != self.num_layers:
                    raise Exception(
                        "Error! You should specify appropriate array for the number of computations per thread.")
                for layer in range(self.num_layers):
                    if self.sizes[layer] % num_comp_per_thread[layer] != 0 \
                            or self.sizes[layer] < num_comp_per_thread[layer]:
                        raise Exception("Error! Each layer should has number of params as a multiple of the "
                                        "corresponding number of computation per thread.")
                if num_comp_in_batch is None or mini_batch_size % num_comp_in_batch != 0:
                    raise Exception("Error! The num_comp_in_batch should be an appropriate number for the number "
                                    "of computations per thread in mini-batch learning, so that mini_batch_size is "
                                    "a multiple of it.")

                # gpu kernels
                self.kernel1 = [createMatProductKernel(pre, nex, mini_batch_size, n, self.num_thread_per_block,
                                                  pre * nex * mini_batch_size / n) for pre, nex, n in
                                zip(self.sizes[:-1], self.sizes[1:], num_comp_per_thread)]
                self.num_block1 = [pre * nex * mini_batch_size / (self.num_thread_per_block * n) + 1
                                   for pre, nex, n in zip(self.sizes[:-1], self.sizes[1:], num_comp_per_thread)]
                self.kernel2 = [createMatProductKernel(self.sizes[-j], self.sizes[-j-1], mini_batch_size,
                                    num_comp_per_thread[-j], self.num_thread_per_block, self.sizes[-j] *
                                    self.sizes[-j-1] * mini_batch_size / num_comp_per_thread[-j])
                                for j in range(1, self.num_layers-1, 1)]
                self.num_block2 = [self.sizes[-j] * self.sizes[-j-1] * mini_batch_size / (self.num_thread_per_block *
                                    num_comp_per_thread[-j]) + 1 for j in range(1, self.num_layers-1, 1)]
                self.kernel3 = [createReluKernel(self.num_thread_per_block, row * mini_batch_size)
                                for row in self.sizes[1:]]
                self.num_block3 = [row * mini_batch_size / self.num_thread_per_block + 1
                                   for row in self.sizes[1:]]
                self.kernel4 = [createReluGradKernel(self.num_thread_per_block, self.sizes[-j-1] * mini_batch_size)
                                for j in range(1, self.num_layers - 1, 1)]
                self.num_block4 = [self.sizes[-j-1] * mini_batch_size / self.num_thread_per_block + 1
                                   for j in range(1, self.num_layers - 1, 1)]
                self.kernel5 = [createMatProductKernel(mini_batch_size, self.sizes[j+1], self.sizes[j],
                                    num_comp_in_batch, self.num_thread_per_block, mini_batch_size *
                                    self.sizes[j+1] * self.sizes[j] / num_comp_in_batch)
                                for j in range(self.num_layers - 1)]
                self.num_block5 = [mini_batch_size * self.sizes[j+1] * self.sizes[j] / (num_comp_in_batch *
                                     self.num_thread_per_block) + 1 for j in range(self.num_layers - 1)]

            # perform mini-batch learning
            if self.gpu:
                for k in range(0, num_samples, mini_batch_size):
                    self.mini_batch((X[:, k:k+mini_batch_size], Y[:, k:k+mini_batch_size]),
                                    learning_rate, momentum, reg, mini_batch_size)
            else:
                for samples in mini_batches:
                    self.mini_batch(samples, learning_rate, momentum, reg, mini_batch_size)

            # show the testing statistics after each epoch of all mini_batches
            print "Epoch {0}: on train_data, {1} % are correct".format(
                i+1, self.evaluate(train_data)/(num_samples+0.0)*100)
            print "    on test_data: {0} % are correct".format(
                self.evaluate(test_data)/(num_tests+0.0)*100)

            # storing learned parameters
            if store_file is not None:
                self.save_to_file(store_file)
                print "Learned parameters stored."
            print

    def mini_batch(self, mini_batch, learning_rate, m, r, mini_batch_size):
        """the mini_batch technique such that we update the weights by computing the gradient just using
           a portion of the samples for better global minimum"""

        # zero initialize current gradient (actually ahead due to momentum)
        D_biases = [np.zeros(b.shape) for b in self.biases]
        D_weights = [np.zeros(w.shape) for w in self.weights]

        # store previous momentum
        prev_m_b = [m_b + 0.0 for m_b in self.mom_b]
        prev_m_w = [m_w + 0.0 for m_w in self.mom_w]

        # compute gradient for each sample
        if self.gpu:
            D_biases, D_weights = self.batch_backward_gpu(mini_batch[0], mini_batch[1], mini_batch_size)
        else:
            for (x, y) in mini_batch:
                curr_biases, curr_weights = self.backward(x, y)
                D_biases = [a + b for a, b in zip(D_biases, curr_biases)]
                D_weights = [a + b for a, b in zip(D_weights, curr_weights)]

        # update current change with regularization (actually gradient ahead)
        change_w = [learning_rate/mini_batch_size * d + learning_rate * r * w
                       for w, d in zip(self.weights, D_weights)]
        change_b = [learning_rate/mini_batch_size * d for d in D_biases]

        # update current momentum
        self.mom_w = [m * m_w - change for m_w, change in zip(self.mom_w, change_w)]
        self.mom_b = [m * m_b - change for m_b, change in zip(self.mom_b, change_b)]

        # update weights and biases
        self.weights = [w - m*p_m_w + (1.0+m)*m_w
                        for w, p_m_w, m_w in zip(self.weights, prev_m_w, self.mom_w)]
        self.biases = [w - m*p_m_b + (1.0+m)*m_b
                       for w, p_m_b, m_b in zip(self.biases, prev_m_b, self.mom_b)]

    def backward(self, x, y):
        """the backward propagation algorithm to compute the gradient of the cost function"""
        D_biases = [np.zeros(b.shape) for b in self.biases]
        D_weights = [np.zeros(w.shape) for w in self.weights]

        # feed-forward
        activation = x
        activations = [x]
        zs = []
        count = 0
        for b, w in zip(self.biases, self.weights):
            count += 1
            z = np.dot(w, activation)+b
            zs.append(z)
            if count < self.num_layers - 1:
                if self.act == 0:
                    activation = Sigmoid(z)
                elif self.act == 1:
                    activation = Tanh(z)
                else:
                    activation = ReLU(z)
                activations.append(activation)
            else:
                activation = Softmax(z)
                activations.append(activation)

        # backward
        # first the special case for output layer (due to softmax layer)
        delta = activations[-1] - y
        D_biases[-1] = delta
        D_weights[-1] = np.dot(delta, activations[-2].T)

        # then compute the derivative of other hidden layers, from large to small
        for l in range(2, self.num_layers):
            z = zs[-l]
            if self.act == 0:
                grad = sigmoid_grad(z)
            elif self.act == 1:
                grad = tanh_grad(z)
            else:
                grad = relu_grad(z)
            delta = np.dot(self.weights[-l+1].T, delta) * grad
            D_biases[-l] = delta
            D_weights[-l] = np.dot(delta, activations[-l-1].T)

        return D_biases, D_weights

    def batch_backward_gpu(self, X, Y, mini_batch_size):
        """the backward propagation algorithm to compute the gradient of the cost function, with a batched version
           for using CUDA-based gpu"""
        D_biases = [np.zeros(b.shape) for b in self.biases]
        D_weights = [np.zeros(w.shape) for w in self.weights]
        weights_gpu = [gpu.to_gpu(self.weights[i].astype(np.float32)) for i in range(self.num_layers - 1)]

        # feed-forward
        activation = gpu.to_gpu(X.astype(np.float32))
        activations = [activation]
        zs = []
        count = 0
        for b, w in zip(self.biases, self.weights):

            # gpu mode variables
            Z = gpu.zeros((self.sizes[count + 1], mini_batch_size), np.float32)
            kernel = self.kernel1[count]
            kernel(weights_gpu[count], activation, Z, grid=(self.num_block1[count], 1, 1),
                   block=(self.num_thread_per_block, 1, 1))
            Z += gpu.to_gpu(mat.repmat(b, 1, mini_batch_size).astype(np.float32))
            zs.append(Z)

            if count < self.num_layers - 2:
                if self.act == 0:
                    activation = 1.0 / (1.0 + gpum.exp(-1 * Z))
                    activations.append(activation)
                elif self.act == 1:
                    activation = 1.7159 * gpum.tanh(2.0 / 3.0 * Z)
                    activations.append(activation)
                else:
                    activation = gpu.zeros(Z.shape, np.float32)
                    kernel = self.kernel3[count]
                    kernel(Z, activation, grid=(self.num_block3[count], 1, 1), block=(self.num_thread_per_block, 1, 1))
                    activations.append(activation)
            else:
                activation = Softmax(Z.get())
                activations.append(activation)
            count += 1

        # backward
        # first the special case for output layer (due to softmax layer)
        delta = activations[-1] - Y
        D_biases[-1] = np.array([np.sum(delta, axis=1)]).T
        delta = gpu.to_gpu(delta.astype(np.float32))
        D_weights_gpu = gpu.zeros(D_weights[-1].shape, np.float32)
        kernel = self.kernel5[-1]

        # these are for handling the bug in PyCuda library regarding matrix transpose
        a_t = gpu.to_gpu(np.zeros((activations[-2].shape[1], activations[-2].shape[0]), np.float32)
                         + gpu.transpose(activations[-2]).get())

        # execute the kernel to update D_weights[-1]
        kernel(delta, a_t, D_weights_gpu, grid=(self.num_block5[-1], 1, 1),
               block=(self.num_thread_per_block, 1, 1))
        D_weights[-1] = D_weights_gpu.get()

        # then compute the derivative of other hidden layers, from large to small
        count = 0
        for l in range(2, self.num_layers):
            Z = zs[-l]
            if self.act == 0:
                grad = (1.0 / (1.0 + gpum.exp(-1 * Z))) * (1 - (1.0 / (1.0 + gpum.exp(-1 * Z))))
            elif self.act == 1:
                grad = 1.7159 * 2 / 3.0 * (1 - (gpum.tanh(2.0/3.0 * Z)) ** 2)
            else:
                grad = gpu.zeros(Z.shape, np.float32)
                kernel = self.kernel4[count]
                kernel(Z, grad, grid=(self.num_block4[count], 1, 1), block=(self.num_thread_per_block, 1, 1))

            product = gpu.zeros((self.weights[-l+1].shape[1], mini_batch_size), np.float32)
            kernel = self.kernel2[count]
            weights_t = gpu.to_gpu((np.zeros((weights_gpu[-l + 1].shape[1], weights_gpu[-l + 1].shape[0]))
                                    + self.weights[-l+1].T).astype(np.float32))
            kernel(weights_t, delta, product, grid=(self.num_block2[count], 1, 1),
                        block=(self.num_thread_per_block, 1, 1))
            delta = product * grad

            # for each weights and biases
            D_biases[-l] = np.array([np.sum(delta.get(), axis=1)]).T
            kernel = self.kernel5[-l]
            a_t = gpu.to_gpu(np.zeros((activations[-l-1].shape[1], activations[-l-1].shape[0]), np.float32)
                                    + gpu.transpose(activations[-l-1]).get())
            D_weights_gpu = gpu.zeros(D_weights[-l].shape, np.float32)
            kernel(delta, a_t, D_weights_gpu, grid=(self.num_block5[-l], 1, 1),
                   block=(self.num_thread_per_block, 1, 1))
            D_weights[-l] = D_weights_gpu.get()
            count += 1

        return D_biases, D_weights

    def evaluate(self, test_data):
        """test the accuracy of the training outcome"""
        test_results = []
        for (x, y) in test_data:
            a = np.argmax(self.forward(x))
            b = np.argmax([y[i][0] for i in range(10)])
            test_results.append((a, b))
        sum_correct = 0
        for (x, y) in test_results:
            if x == y:
                sum_correct += 1
        return sum_correct


def Sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_grad(z):
    """gradient of the sigmoid function."""
    return Sigmoid(z) * (1-Sigmoid(z))


def Softmax(z):
    """the softmax activation function for the output layer, best suitable for
       disjoint classes"""
    out = exp(z)
    sum_exp = np.sum(out, axis=0)
    res = out/sum_exp
    return res


def Tanh(z):
    """the funny Tanh activation function"""
    return 1.7159 * np.tanh(2 / 3.0 * z)


def tanh_grad(z):
    """the gradient of funny Tanh(z)"""
    return 1.7159 * 2 / 3.0 * (1 - (np.tanh(2/3.0 * z)) ** 2)


def ReLU(z):
    """the ReLU activation function"""
    return np.max([z, np.zeros(z.shape)], axis=0)


def relu_grad(z):
    """the gradient of ReLU(z)"""
    index = z >= 0
    result = np.zeros(z.shape)
    result[index] = 1.0
    return result


def classification_vector(this_class, num_classes):
    """from a target value, return a vector to represent its value"""
    result = [0 for i in range(num_classes)]
    result[this_class] = 1
    return np.array([result]).T


def createMatProductKernel(pre, nex, mini_batch_size, num_comp_per_thread, num_thread_per_block, max_size):
    """create kernel for matrix-matrix product using GPU"""

    kernel = """
    __global__ void MatMatProductKernel(float * A, float * B, float * C) {

        const uint pre = """ + str(pre) + """;
        const uint nex = """ + str(nex) + """;
        const uint miniBatchSize = """ + str(mini_batch_size) + """;
        const uint numComp = """ + str(num_comp_per_thread) + """;
        const uint blockSize = """ + str(num_thread_per_block) + """;
        const uint maxSize = """ + str(max_size) + """;
        const uint divide = pre / numComp;
        const uint x = blockIdx.x * blockSize + threadIdx.x;

        if (x < maxSize) {
            uint sampleIdx = x / (nex * divide);
            uint resultIdx = (x % (nex * divide)) / divide;
            uint compIdx = (x % (nex * divide)) % divide;

            float sum = 0;
            uint start = resultIdx * pre;
            for (int i = compIdx * numComp; i < (compIdx + 1) * numComp; i++) {
                sum += A[start + i] * B[i * miniBatchSize + sampleIdx];
            }

            atomicAdd(&(C[resultIdx * miniBatchSize + sampleIdx]), sum);
         }
    } """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("MatMatProductKernel")


def createReluKernel(num_thread_per_block, max_size):
    """create kernel for the rectified linear unit using GPU"""
    kernel = """
    __global__ void ReluKernel(float * A, float * B) {

        const uint blockSize = """ + str(num_thread_per_block) + """;
        const uint maxSize = """ + str(max_size) + """;
        const uint x = blockIdx.x * blockSize + threadIdx.x;

        if (x < maxSize) {
            float curr = A[x];
            if (curr > 0) {
                B[x] = curr;
            } else {
                B[x] = 0;
            }
        }
    } """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("ReluKernel")


def createReluGradKernel(num_thread_per_block, max_size):
    """create kernel for the gradient of rectified linear unit using GPU"""
    kernel = """
    __global__ void ReluGradKernel(float * A, float * B) {

        const uint blockSize = """ + str(num_thread_per_block) + """;
        const uint maxSize = """ + str(max_size) + """;
        const uint x = blockIdx.x * blockSize + threadIdx.x;

        if (x < maxSize) {
            float curr = A[x];
            if (curr > 0) {
                B[x] = 1;
            } else {
                B[x] = 0;
            }
        }
    } """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("ReluGradKernel")


def separation(samples):
    """a function for separation of data and its labels."""
    X = []
    Y = []
    for x, y in samples:
        X.append(x)
        Y.append(y)
    X = np.concatenate(tuple(X), axis=1)
    Y = np.concatenate(tuple(Y), axis=1)
    return X, Y

