# Deep Learning Library

## Overview
This deep learning library currently contains two kinds of tools that are usually used in the field of machine learning; specifically, they are fully-connected feedforward neural network and long short-term memory. The object-oriented implementation is designed in a way that it makes use of the parallel computing power of GPUs. Special configurations are needed, though.

## GPU mode prerequisite
This library utilizes codes for CUDA-supported Nvidia cards. CUDA Toolkits 7.x is required. For instance, follow the steps [here](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu) if your have Ubuntu 14.04. Furthermore, PyCUDA should be installed as is used in this library for the python wrapper for cuda codes. Similarly for Ubuntu 14.04, click [here](https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu) for installation process. 

## Feedforward Neural Net
![what a feedforward neural net looks like](https://github.com/SeanJia/DeepLearningLibrary/blob/master/readme-images/1.png)
#### Basic features
* The feedforward neural network in this library is designed for classification problems, and is fully connected, supportive of mini-batch stochastic gradient descent, L2-norm regularization, and three different kinds of activation functions. The output layer use Softmax as the activation function for multi-class classification.
* To create a neural net, use `nn = NerualNet(sizes=layers, act=activation, gpu_mod=False)`. 
* The layers here is a list of integers for the topology of the network. For instance, the network in the image above would have `layers = [3, 10, 10, 3]`. 
* The activation is for the nonlinearity in the network. With value of 0, 1 and 2 representing Sigmoid, funny Tanh, and recified linear (ReLU), where funny tanh is a modified version of tanh function described in Yann LeCun's [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). By default it uses Sigmoid for activation function.
* Notice that for ReLU to be used in fully-connected neural nets, heuristicaly large mini-batch size is recommended to avoid extremely large terms in forward propagation, which might cause overflow for the exponential calculation in the output layer.

Other features are listed below.

#### Training and testing data
* Use `nn.training(train_data, test_data, max_epoch=200, mini_batch_size=100, learning_rate=0.01, momentum=0.9)` for training, where `train_data` and `test_data` are considered as lists of two tuples. Each component in the list is of the form `(x, y)` where x is the input data as column vector (a 2d numpy array), and y is the label, a binary column vector (also 2d numpy array) to indicate which class the data x belongs to. 
* Notice that in reality the algorithms require you to wrap the lists (train or test data) as numpy arrays. This can be easily acheived by something similar to `train_data = numpy.array(list_for_train_data)`.
* Normally training data has some preprocessing, such as make the mean to be zero and normalize the variance of each dimension.
* This neural net provides with a built-in function for creating a binary label vector representing class information. You can call by `classification_vector(num_represent_class_for_this_data, num_of_total_classes)`.

#### Nesterov's momentum
* This neural net supports Nesterov's momentum, which is a technique for boosting up the convergence of stochastic gradient descent. Detail can be found [here](https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/).

#### Xavier initialization
* For layers with large number of weights, we want to maintain the activated output to be of similar scale. Xavier initialization of weights is a good approach, and is supported in this neural net by default. More mathematical background of this technique can be fount [here](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization).

#### Storage of learned parameters
* Learned parameters will be stored into local file specified by the last parameter in calling `training(..., store_file)`. E.g., `store_file = "learnedStuff"` will create and store paramters into a local file called `learnedStuff.npy`.
* It provides load function to load learned parameters into the neural net model. E.g. `nn.load_from_file("learnedStuff")` will read from `learnedStuff.npy` to extract your learned weights and biases, as well as previous topology configurations.

#### GPU mode
* For feedforward neural net, mini-batch gradient descent could enable a drastically faster learning process if using GPU for parallel computation. Note that if PyCUDA is loaded and imported correctly, `GPU mode ready!` will show up upon importing the neural net module.
* When creating the neural net in gpu mode, specify `gpu_mod=True` and `num_thread_per_block=256`. The latter is used to suggest how many threads will be in one CUDA block (in CUDA, threads within a block are execuated in a parallel way, yet different blocks are not necessarily so). The exact number influences the performance, and depends on your card; however usually it's 128, 256 or 512. 
* Upon training process, you also need to specify `num_comp_per_thread=list_num` and `num_comp_in_batch=num` correctly, where `list_num` is a list of number whose length is the same as `sizes` when creating the neural net. Furthermore, `list_num` should be set up so that each component in `sizes` is a multiple of the corresponding component in the it, and should be larger or equal to it as well. The `num_comp_in_batch` is used in the batched learning settings, and similarly, `mini_batch_sizes` should be a multiple of `num_comp_in_batch` and no less than it.
* If these two parameters are not set up correctly, it will raise exceptions.

#### Sample training performance
When training on [MNIST](http://yann.lecun.com/exdb/mnist/), with zero mean and variance normalization as preprocessing of the data, this algorithm achieved a best testing accuracy as 98.6%, which is around the state-of-the-art level for fully connected neural nets.

## Long Short-term Memory (LSTM) 
LSTM is a special structure of the recurrent neural network. The implementation here has the version with this topology. 
![structure of this LSTM](https://github.com/SeanJia/DeepLearningLibrary/blob/master/readme-images/2.png)

#### Basic features
* This LSTM is designed primarily for sequence to sequence learning, with same input and output size, e.g., language modeling. It can be used for other sorts of sequence involved learning after some minor modifications, though.
* 

