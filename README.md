# Deep Learning Library

## Overview
This deep learning library currently contains two kinds of tools that are usually used in the field of machine learning; specifically, they are fully-connected feedforward neural network and long short-term memory. The object-oriented implementation is designed in a way that it makes use of the parallel computing power of GPUs. Special configurations are needed, though.

## GPU mode prerequisite
This library utilizes codes for CUDA-supported Nvidia cards. CUDA Toolkits 7.x is required. For instance, follow the steps [here](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.5-ubuntu) if your have Ubuntu 14.04. Furthermore, PyCUDA should be installed as is used in this library for the python wrapper for cuda codes. Similarly for Ubuntu 14.04, click [here](https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu) for installation process. 

## Feedforward Neural Net
![what a feedforward neural net looks like](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
#### Basic features
The feedforward neural network in this library is fully connected, supportive of mini-batch gradient-based learning, L2-norm regularization, and three different kinds of activation functions. 
