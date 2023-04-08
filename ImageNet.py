# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:01:58 2023

@author: Christopher Hansen
         Kelvin lu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import FeedForwardNeuralNet as ffnet

"""
    This class defines an artificial neural network based on a 2D CNN and 
    a series of fully connected layers. The CNN layers will perform the feature
    extraction of the input (images), and the feed forward neural networks will
    perform the classification of the images. 
"""
class ImageNet(torch.nn.Module):
    def __init__(self, input_dims): # config for wandb tuning dropout
        super(ImageNet, self).__init__()
        self.dropout = 0.1
        self.kernel_size = 4 # We are only using square kernels
        self.kernel = [self.kernel_size, self.kernel_size]
        self.stride = 1
        self.n_channels = 1 # 1 for grayscale or 3 for rgb
        self.height = input_dims[0] # input image height
        self.width = input_dims[1] # input image width
        self.height_top_pad=0
        self.height_bottom_pad=0
        self.width_top_pad = 0
        self.width_bottom_pad = 0
        conv_channels = 1
        
        # input convolutional layer
        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=conv_channels, 
                               kernel_size=self.kernel, stride=1, bias=True)

        # output vector size of previous layer -> input vector to next layer
        self.height = (self.height + self.height_top_pad + self.height_bottom_pad - self.kernel_size) /self.stride + 1
        self.width = (self.width + self.width_top_pad + self.width_bottom_pad - self.kernel_size) /self.stride + 1 
        self.output_shape = self.width*self.height
        
        # pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride)
        self.height = (self.height + self.height_top_pad + self.height_bottom_pad - self.kernel_size) /self.stride + 1
        self.width = (self.width + self.width_top_pad + self.width_bottom_pad - self.kernel_size) /self.stride + 1 
        self.output_shape = self.width*self.height
        
        self.ffnet = ffnet.NeuralNetwork(n_feature=int(self.output_shape), n_hidden_nodes=40, 
                                         n_hidden_layers=5, n_output=1, 
                                         dropout_rate=self.dropout)
               
    def forward(self, x):
        # apply convolutional layer and ReLU
        x = F.relu(self.conv1(x))
        # apply max pooling layer
        x = self.pool1(x)

        # reshape output for a fully connected layer
        x=x.view(x.size(0),-1)
        x = self.ffnet(x)
        
        return x