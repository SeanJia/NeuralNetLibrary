__author__ = 'Zhiwei Jia'

import numpy as np
import numpy.matlib
try :
    import pycuda.autoinit
    import pycuda.gpuarray as gpu
    import pycuda.cumath as gpum
    from pycuda import driver, compiler, tools
    print 'GPU mode ready!\n'
except ImportError:
    print 'PyCUDA not found! GPU mode not ready.\n'

class LSTM:

    def __init__(self, size=0, hidden_s=0, gpu_mode=False, num_thread_per_block=256):
        """initialization of a lstm node"""

        # configurations
        if gpu_mode and size % 2 != 0:
            raise Exception('ERROR! when using gpu mode, the input size should be an even number.')
        self.input_s = size
        self.hidden_s = hidden_s
        self.output_s = size
        self.h = np.zeros((hidden_s, 1))
        self.c = np.zeros((hidden_s, 1))
        self.gpu = gpu_mode
        self.num_block = 0
        self.num_thread_per_block = 0
        self.kernel = None

        # weights and bias parameters
        variance = np.sqrt(2.0 / (self.input_s + self.output_s + self.hidden_s))
        self.forget_w = np.random.randn(self.hidden_s, self.input_s + self.hidden_s) * variance
        self.forget_b = np.zeros((self.hidden_s, 1))
        self.sel_w = np.random.randn(self.hidden_s, self.input_s + self.hidden_s) * variance
        self.sel_b = np.zeros((self.hidden_s, 1))
        self.add_w = np.random.randn(self.hidden_s, self.input_s + self.hidden_s) * variance
        self.add_b = np.zeros((self.hidden_s, 1))
        self.write_w = np.random.randn(self.hidden_s, self.input_s + self.hidden_s) * variance
        self.write_b = np.zeros((self.hidden_s, 1))
        self.biases = np.zeros((self.output_s, 1))
        self.weights = np.random.randn(self.output_s, self.hidden_s) * variance

        # memory variables for AdaGrad
        self._forget_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        self._forget_b = np.zeros((self.hidden_s, 1))
        self._sel_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        self._sel_b = np.zeros((self.hidden_s, 1))
        self._add_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        self._add_b = np.zeros((self.hidden_s, 1))
        self._write_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        self._write_b = np.zeros((self.hidden_s, 1))
        self._biases = np.zeros((self.output_s, 1))
        self._weights = np.zeros((self.output_s, self.hidden_s))

        if self.gpu:

            # for gpu configuration
            self.num_thread_per_block = num_thread_per_block
            self.num_block = self.hidden_s * 4 * 2 / num_thread_per_block + 1

            # a matrix vector multiplication kernel
            self.kernel = createMultiplyKernel(self.num_thread_per_block,
                                               self.hidden_s + self.input_s, self.hidden_s * 4)

    def load_from_file(self, load_file, gpu_mode=False, num_thread_per_block=256):
        """load learned parameter from local file"""
        mat1 = np.load(load_file + "1.npy")
        mat2 = np.load(load_file + "2.npy")
        self.input_s = mat1[0]
        self.hidden_s = mat1[1]
        self.output_s = mat1[2]
        self.gpu = mat1[3]

        if gpu_mode:
            if self.gpu:
                self.num_block = mat1[4]
                self.num_thread_per_block = mat1[5]
            else:
                self.gpu = True
                self.num_thread_per_block = num_thread_per_block
                self.num_block = self.hidden_s * 4 * 2 / num_thread_per_block + 1
            self.kernel = createMultiplyKernel(self.num_thread_per_block,
                                               self.hidden_s + self.input_s, self.hidden_s * 4)
        else:
            self.num_block = 0
            self.num_thread_per_block = 0
            self.kernel = None
            self.gpu = False

        # load parameters
        self.h = mat2[0]
        self.c = mat2[1]
        self.forget_w = mat2[2]
        self.forget_b = mat2[3]
        self.sel_w = mat2[4]
        self.sel_b = mat2[5]
        self.add_w = mat2[6]
        self.add_b = mat2[7]
        self.write_w = mat2[8]
        self.write_b = mat2[9]
        self.weights = mat2[10]
        self.biases = mat2[11]
        self._forget_w = mat2[12]
        self._forget_b = mat2[13]
        self._sel_w = mat2[14]
        self._sel_b = mat2[15]
        self._add_w = mat2[16]
        self._add_b = mat2[17]
        self._write_w = mat2[18]
        self._write_b = mat2[19]
        self._weights = mat2[20]
        self._biases = mat2[21]

    def save_to_file(self, save_file=None):
        """save learned parameters to local file"""
        if self.gpu:
            mat1 = [self.input_s, self.hidden_s, self.output_s, self.gpu, self.num_block, self.num_thread_per_block]
        else:
            mat1 = [self.input_s, self.hidden_s, self.output_s, self.gpu]
        mat2 = [self.h, self.c, self.forget_w,
                self.forget_b, self.sel_w, self.sel_b, self.add_w, self.add_b, self.write_w, self.write_b,
                self.weights, self.biases, self._forget_w, self._forget_b, self._sel_w, self._sel_b,
                self._add_w, self._add_b, self._write_w, self._write_b, self._weights, self._biases]
        np.save(save_file + "1", np.array(mat1))
        np.save(save_file + "2", np.array(mat2))

    def forget(self, x):
        """the forget gate"""
        info = np.concatenate((self.h, x), axis=0)
        forget = np.dot(self.forget_w, info)
        forget += self.forget_b
        return Sigmoid(forget)

    def read(self, x):
        """the input gate"""
        info = np.concatenate((self.h, x), axis=0)
        select = np.dot(self.sel_w, info)
        select += self.sel_b
        add = np.dot(self.add_w, info)
        add += self.add_b
        return Sigmoid(select), Tanh(add)

    def write(self, x, forget, select, add):
        """the output gate"""
        info = np.concatenate((self.h, x), axis=0)
        res1 = np.dot(self.write_w, info) + self.write_b
        res1 = Sigmoid(res1)
        new_c = self.c * forget + select * add
        res2 = Tanh(new_c)
        new_h = res1 * res2
        return new_h, new_c

    def output(self, h, temperature):
        """generate result"""
        res = np.dot(self.weights, h) + self.biases
        return Softmax(res, temperature)

    def forward(self, x, temperature):
        """forward propagation"""
        forget = self.forget(x)
        select, add = self.read(x)
        new_h, new_c = self.write(x, forget, select, add)
        self.h = new_h
        self.c = new_c
        res = self.output(self.h, temperature)
        return res

    def forward_gpu(self, x, temperature):
        """forward propagation in gpu mode"""

        # obtain z
        hx = np.concatenate((self.h, x))
        hx_gpu = gpu.to_gpu(hx.astype(np.float32))
        all_weights = np.concatenate((self.forget_w, self.sel_w, self.write_w, self.add_w))
        all_biases = np.concatenate((self.forget_b, self.sel_b, self.write_b, self.add_b))
        all_weights_gpu = gpu.to_gpu(all_weights.astype(np.float32))
        all_biases_gpu = gpu.to_gpu(all_biases.astype(np.float32))
        z = gpu.zeros((self.hidden_s * 4, 1), np.float32)
        self.kernel(all_weights_gpu, hx_gpu, z, grid=(self.num_block, 1, 1), block=(self.num_thread_per_block, 1, 1))
        z += all_biases_gpu

        # non-linearity
        z[:self.hidden_s * 3, :1] = 1.0 / (gpum.exp(-1 * z[:self.hidden_s * 3, :1]) + 1.0)
        z[self.hidden_s * 3:, :1] = 1.7159 * gpum.tanh(2.0 / 3.0 * z[self.hidden_s * 3:, :1])
        z_cpu = z.get()

        # update cell and hidden
        self.c = z_cpu[:self.hidden_s, :1] * self.c + \
                 z_cpu[self.hidden_s:self.hidden_s*2, :1] * z_cpu[self.hidden_s*3:, :1]
        self.h = z_cpu[self.hidden_s * 2: self.hidden_s * 3, :1] * Tanh(self.c)

        # output
        res = np.dot(self.weights, self.h) + self.biases
        return Softmax(res, temperature)

    def train(self, train_data, num_epoch=100, mini_batch_size=4, learning_rate=0.1,
              temperature=1, length=50, show_res_every=100, num_shown_res=400, store_every=100, store_file=None):
        """training process"""
        num_samples = len(train_data)
        count = 0
        smooth_loss = -np.log(1.0 / self.output_s)               # loss at iteration 0
        curr_loss = 0

        # epoches
        for i in xrange(num_epoch):

            # at the beginning of each epoch, reset hidden and cell
            print '\nEPOCH', i, '----------------------------\n'
            self.h = np.zeros((self.hidden_s, 1))
            self.c = np.zeros((self.hidden_s, 1))

            # mini-batch
            mini_batches = [train_data[k:k+mini_batch_size*length+1]
                            for k in range(0, num_samples, mini_batch_size*length)]

            # iterations within each mini-batches
            for k in xrange(len(mini_batches)):
                samples = mini_batches[k]
                if not len(samples) == mini_batch_size * length + 1:
                    break

                # show sampling result from the neural net
                if k % show_res_every == 0:
                    string = "    After {0} updates: smooth loss is {1}, current loss is {2}\n".format(
                        count * show_res_every, smooth_loss, curr_loss/(show_res_every+0.0))
                    curr_loss = 0
                    print string
                    count += 1
                    temp_h = self.h + 0
                    temp_c = self.c + 0
                    test = samples[0]
                    string = self.evaluate(test, temperature, num_shown_res)
                    self.h = temp_h
                    self.c = temp_c
                    print "Sampling: ", string

                # save the relevant data to local files
                if k % store_every == 0:
                    print "----- store weights at the {0}th updates in the {1}th epoch -----".format(k, i + 1)
                    if store_file is not None:
                        self.save_to_file(store_file)

                # learning via mini-batch
                smooth_loss, loss = self.mini_batch(samples, learning_rate, temperature, length, smooth_loss)
                curr_loss += loss

    def mini_batch(self, samples, learning_rate, temperature, length, smooth_loss):
        """mini-batch learning"""

        # zero initialize current gradient
        D_forget_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_forget_b = np.zeros((self.hidden_s, 1))
        D_sel_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_sel_b = np.zeros((self.hidden_s, 1))
        D_add_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_add_b = np.zeros((self.hidden_s, 1))
        D_write_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_write_b = np.zeros((self.hidden_s, 1))
        D_biases = np.zeros((self.output_s, 1))
        D_weights = np.zeros((self.output_s, self.hidden_s))

        # back propagation through time to get gradient
        i = 0
        curr_loss = 0
        while i + length + 1 <= len(samples):
            curr = samples[i:i+length+1]
            curr_forget_w, curr_forget_b, curr_sel_w, curr_sel_b, curr_add_w, curr_add_b, curr_write_w, \
                    curr_write_b, curr_weights, curr_biases, loss = self.bptt(curr, temperature, length)
            i += length
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            curr_loss += loss

            D_forget_w += curr_forget_w
            D_forget_b += curr_forget_b
            D_sel_w += curr_sel_w
            D_sel_b += curr_sel_b
            D_add_w += curr_add_w
            D_add_b += curr_add_b
            D_write_w += curr_write_w
            D_write_b += curr_write_b
            D_weights += curr_weights
            D_biases += curr_biases

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([self.forget_w, self.forget_b, self.sel_w, self.sel_b, self.add_w,
                                          self.add_b, self.write_w, self.write_b, self.weights, self.biases],
                                [D_forget_w, D_forget_b, D_sel_w, D_sel_b, D_add_w,
                                          D_add_b, D_write_w, D_write_b, D_weights, D_biases],
                                [self._forget_w, self._forget_b, self._sel_w, self._sel_b, self._add_w,
                                          self._add_b, self._write_w, self._write_b, self._weights, self._biases]):
            mem += dparam * dparam

            # this lin updates the parameters
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

        return smooth_loss, curr_loss/length

    def evaluate(self, test_data, temperature, length):
        """the sampling process to generate samples from current learning state"""

        # insert the first test element
        string = []
        idx = np.argmax(test_data)
        string.append(chr(idx))

        # convert back to the all-zero-except-one form
        test_data = np.zeros((self.input_s, 1))
        test_data[idx, [0]] += 1

        # loop through for generating samples
        for t in range(length - 1):
            if self.gpu:
                result = self.forward_gpu(test_data, temperature)
                test_data[idx, 0] -= 1
                idx = np.random.choice(range(self.output_s), p=result.ravel())
                test_data[idx, 0] += 1
            else:
                result = self.forward(test_data, temperature)
                test_data[idx, [0]] -= 1
                idx = np.random.choice(range(self.output_s), p=result.ravel())
                test_data[idx, [0]] += 1
            string.append(chr(idx))

        return ''.join(string)

    def bptt(self, data, temperature, length):
        """full back propagation through time"""
        loss = 0
        D_forget_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_forget_b = np.zeros((self.hidden_s, 1))
        D_sel_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_sel_b = np.zeros((self.hidden_s, 1))
        D_add_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_add_b = np.zeros((self.hidden_s, 1))
        D_write_w = np.zeros((self.hidden_s, self.input_s + self.hidden_s))
        D_write_b = np.zeros((self.hidden_s, 1))
        D_biases = np.zeros((self.output_s, 1))
        D_weights = np.zeros((self.output_s, self.hidden_s))
        hANDx = []
        forget_in = []
        sel_in = []
        add_in = []
        write_in = []
        c_hist = []
        h_hist = []
        forget_ = []
        sel_ = []
        add_ = []
        write_ = []
        prediction = []
        c_init = np.copy(self.c)

        E_over_write_next = np.zeros((1, self.hidden_s))
        E_over_c_next = np.zeros((1, self.hidden_s))

        # first forward propagation
        if self.gpu:  # in gpu mode

            all_weights = np.concatenate((self.forget_w, self.sel_w, self.write_w, self.add_w))
            all_biases = np.concatenate((self.forget_b, self.sel_b, self.write_b, self.add_b))
            all_weights_gpu = gpu.to_gpu(all_weights.astype(np.float32))
            all_biases_gpu = gpu.to_gpu(all_biases.astype(np.float32))
            z = gpu.zeros((self.hidden_s * 4, 1), np.float32)

            for i in range(length):
                x = data[i]

                # obtain z
                hx = np.concatenate((self.h, x))
                hANDx.append(hx)
                hx_gpu = gpu.to_gpu(hx.astype(np.float32))
                self.kernel(all_weights_gpu, hx_gpu, z, grid=(self.num_block, 1, 1),
                            block=(self.num_thread_per_block, 1, 1))
                z += all_biases_gpu
                z_cpu = z.get()
                forget_in.append(z_cpu[:self.hidden_s, :1])
                sel_in.append(z_cpu[self.hidden_s:self.hidden_s * 2, :1])
                write_in.append(z_cpu[self.hidden_s * 2:self.hidden_s * 3, :1])
                add_in.append(z_cpu[self.hidden_s * 3:, :1])

                # non-linearity
                z[:self.hidden_s * 3, :1] = 1.0 / (gpum.exp(-1 * z[:self.hidden_s * 3, :1]) + 1.0)
                z[self.hidden_s * 3:, :1] = 1.7159 * gpum.tanh(2 / 3.0 * z[self.hidden_s * 3:, :1])
                z_cpu = z.get()
                forget_.append(z_cpu[:self.hidden_s, :1])
                sel_.append(z_cpu[self.hidden_s:self.hidden_s * 2, :1])
                write_.append(z_cpu[self.hidden_s * 2:self.hidden_s * 3, :1])
                add_.append(z_cpu[self.hidden_s * 3:, :1])

                # update cell and hidden
                self.c = z_cpu[:self.hidden_s, :1] * self.c + z_cpu[self.hidden_s:self.hidden_s * 2, :1] \
                                * z_cpu[self.hidden_s * 3:, :1]
                self.h = z_cpu[self.hidden_s * 2: self.hidden_s * 3, :1] * Tanh(self.c)
                c_hist.append(self.c + 0)
                h_hist.append(self.h + 0)

                # output
                res = Softmax(np.dot(self.weights, self.h) + self.biases, temperature)
                prediction.append(res)
                loss += -np.log(res[np.argmax(data[i + 1]), 0])

        else:
            for i in range(length):
                x = data[i]
                info = np.concatenate((self.h, x), axis=0)
                hANDx.append(info)

                a = np.dot(self.forget_w, info) + self.forget_b
                forget_in.append(a)
                forget = Sigmoid(a)
                forget_.append(forget)

                a = np.dot(self.sel_w, info) + self.sel_b
                sel_in.append(a)
                select = Sigmoid(a)
                sel_.append(select)

                a = np.dot(self.add_w, info) + self.add_b
                add_in.append(a)
                add = Tanh(a)
                add_.append(add)

                self.c = self.c * forget + select * add
                a = np.dot(self.write_w, info) + self.write_b
                write_in.append(a)
                write = Sigmoid(a)
                write_.append(write)
                c_hist.append(np.copy(self.c))
                self.h = write * Tanh(self.c)
                h_hist.append(np.copy(self.h))

                a = np.dot(self.weights, self.h) + self.biases
                res = Softmax(a, temperature)
                prediction.append(res)
                loss += -np.log(res[np.argmax(data[i+1]), 0])

        # back propagation through time
        for i in range(length-1, -1, -1):

            # some variable
            hx_t = np.transpose(hANDx[i])

            # obtain current layer delta
            delta = prediction[i] - data[i+1]
            D_biases += delta
            D_weights += np.dot(delta, h_hist[i].T)

            # obtain E_over_h w.r.t. current layer delta
            delta_h = np.dot(delta.T, self.weights)

            # obtain E_over_h w.r.t. write gate
            if i == length-1:
                write_h = np.zeros((1, self.hidden_s))
            else:
                diag_sigmoid_grad = numpy.matlib.repmat(sigmoid_grad(write_in[i+1]), 1, self.hidden_s)
                write_w_part = self.write_w[:, :self.hidden_s]
                write_over_h = diag_sigmoid_grad * write_w_part
                write_h = np.dot(E_over_write_next, write_over_h)

            # obtain E_over_h w.r.t. memory cell
            if i == length-1:
                c_h = np.zeros((1, self.hidden_s))
            else:

                # part A: forget_over_h
                diag_sigmoid_grad = numpy.matlib.repmat(sigmoid_grad(forget_in[i+1]), 1, self.hidden_s)
                forget_w_part = self.forget_w[:, :self.hidden_s]
                forget_over_h = diag_sigmoid_grad * forget_w_part
                forget_over_h *= numpy.matlib.repmat(c_hist[i], 1, self.hidden_s)

                # part B: sel_over_h
                diag_sigmoid_grad = numpy.matlib.repmat(sigmoid_grad(sel_in[i+1]), 1, self.hidden_s)
                sel_w_part = self.sel_w[:, :self.hidden_s]
                sel_over_h = diag_sigmoid_grad * sel_w_part
                sel_over_h *= numpy.matlib.repmat(add_[i+1], 1, self.hidden_s)

                # part C: add_over_h
                diag_sigmoid_grad = numpy.matlib.repmat(sigmoid_grad(add_in[i+1]), 1, self.hidden_s)
                add_w_part = self.add_w[:, :self.hidden_s]
                add_over_h = diag_sigmoid_grad * add_w_part
                add_over_h *= numpy.matlib.repmat(sel_[i+1], 1, self.hidden_s)

                # finally c_h
                c_over_h = forget_over_h + sel_over_h + add_over_h
                c_h = np.dot(E_over_c_next, c_over_h)

            # obtain E_over_h and relevant gradients
            E_over_h = delta_h + write_h + c_h

            # write gate update
            update_write = E_over_h * np.transpose(Tanh(c_hist[i]))
            update_write *= np.transpose(sigmoid_grad(write_in[i]))
            D_write_b += update_write.T
            D_write_w += np.dot(update_write.T, hx_t)

            # memory cell update, with E_over_c recursively, and update E_over_c_next as well
            E_over_c = E_over_h * np.transpose(write_[i]) * np.transpose(tanh_grad(c_hist[i]))
            if i == length-1:
                E_over_c_next = E_over_c
            else:
                E_over_c += E_over_c_next * np.transpose(forget_[i+1])
                E_over_c_next = E_over_c

            # forget gate update
            if i == 0:
                c_last = c_init
            else:
                c_last = c_hist[i-1]
            update_forget = E_over_c * np.transpose(c_last) * np.transpose(sigmoid_grad(forget_in[i]))
            D_forget_b += update_forget.T
            D_forget_w += np.dot(update_forget.T, hx_t)

            # sel update
            update_sel = E_over_c * np.transpose(add_[i])
            update_sel *= np.transpose(sigmoid_grad(sel_in[i]))
            D_sel_b += update_sel.T
            D_sel_w += np.dot(update_sel.T, hx_t)

            # add update
            update_add = E_over_c * np.transpose(sel_[i])
            update_add *= np.transpose(tanh_grad(add_in[i]))
            D_add_b += update_add.T
            D_add_w += np.dot(update_add.T, hx_t)

            # update E_over_write
            E_over_write_next = E_over_h * np.transpose(Tanh(c_hist[i]))

        for each in [D_forget_w, D_forget_b, D_sel_w, D_sel_b, D_add_w, D_add_b,
                D_write_w, D_write_b, D_weights, D_biases]:
            np.clip(each, -30, 30, out=each)

        return D_forget_w, D_forget_b, D_sel_w, D_sel_b, D_add_w, D_add_b, \
                D_write_w, D_write_b, D_weights, D_biases, loss/(length+0.0)


def Sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_grad(z):
    """gradient of the sigmoid function."""
    return Sigmoid(z) * (1-Sigmoid(z))


def Tanh(z):
    """the funny Tanh activation function"""
    return 1.7159 * np.tanh(2 / 3.0 * z)


def tanh_grad(z):
    """the gradient of funny Tanh(z)"""
    return 1.7159 * 2 / 3.0 * (1 - (np.tanh(2/3.0 * z)) ** 2)


def Softmax(z, t):
    """the softmax activation function for the output layer, with t as temperature"""
    out = np.exp(z / t)
    sum_e = np.sum(out)
    return out/sum_e


def gpu_concatenate_two(a_gpu, b_gpu, size1, size2):
    """concatenate two gpu vectors vertically"""
    array = gpu.zeros((size1 + size2, 1), np.float32)
    array[:size1, :1] += a_gpu
    array[size1:, :1] += b_gpu
    return array


def gpu_concatenate_four(a_gpu, b_gpu, c_gpu, d_gpu, height, width):
    """concatenate four gpu matrices vertically"""
    array = gpu.zeros((height * 4, width), np.float32)
    array[:height, :width] += a_gpu
    array[height:height*2, :width] += b_gpu
    array[height*2:height*3, :width] += c_gpu
    array[height*3:, :width] += d_gpu
    return array


def createMultiplyKernel(num_thread_per_block, width, height):
    """a function to return a cuda kernel for multiplication between a matrix and a vector"""
    kernel = """
    __global__ void MatVecProductKernel(float * A, float * B, float * C) {

        const uint width = """ + str(width) + """;
        const uint height = """ + str(height) + """;
        const uint blockSize = """ + str(num_thread_per_block) + """;
        const uint x = blockIdx.x * blockSize + threadIdx.x;
        uint x_ = x / 2 * width;

        float sum = 0;
        if (x < height * 2) {
            if (x % 2 == 0) {
                for (int i = 0; i < width / 2; i++) sum += A[x_ + i] * B[i];
            } else {
                for (int i = width / 2; i < width; i++) sum += A[x_ + i] * B[i];
            }
        }

        if (x < height * 2) {
            if (x % 2 == 1) {
                __syncthreads();
                C[x / 2] += sum;
            } else {
                C[x / 2] = sum;
                __syncthreads();
            }
        } else {
            __syncthreads();
        }
    } """

    mod = compiler.SourceModule(kernel)
    return mod.get_function("MatVecProductKernel")

