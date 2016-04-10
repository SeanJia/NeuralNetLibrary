# Deep Learning Library

## Overview
This deep learning library currently contains two kinds of tools that are usually used in the field of machine learning; specifically, they are fully-connected feedforward neural network and long short-term memory. The object-oriented implementation is designed in a way that it makes use of the parallel computing power of GPUs. Special configurations are needed, though.

## GPU mode prerequisite
This library utilizes codes for CUDA-supported Nvidia cards. CUDA Toolkits 7.x is required. For instance, follow the steps [here](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu) if your have Ubuntu 14.04. Furthermore, PyCUDA should be installed as is used in this library for the python wrapper for cuda codes. Similarly for Ubuntu 14.04, click [here](https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu) for installation process. 

## Feedforward Neural Net
![what a feedforward neural net looks like](https://github.com/SeanJia/DeepLearningLibrary/blob/master/readme-images/1.png)
#### Basic features
* The feedforward neural network in this library is fully connected, supportive of mini-batch gradient-based learning, L2-norm regularization, and three different kinds of activation functions. 

* To create a neural net, use `nn = NerualNet(sizes=layers, act=activation, gpu_mod=False)`. 

* The layers here is a list of integers for the topology of the network. For instance, the network in the image above would have `layers = [3, 10, 10, 3]`. 

* The activation is for the nonlinearity in the network. With value of 0, 1 and 2 representing Sigmoid, funny Tanh, and recified linear (ReLU), where funny tanh is a modified version of tanh function described in Yann LeCun's [paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). By default it uses Sigmoid for activation function.

* Notice that for ReLU to be used in fully-connected neural nets, heuristicaly large mini-batch size is recommended to avoid extremely large terms in forward propagation, which might cause overflow for the exponential calculation in the output layer.

Other features are listed below.


