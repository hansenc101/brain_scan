# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:01:58 2023

@author: Christopher Hansen
         Kelvin lu
"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import FeedForwardNeuralNet as ffnet

"""
    This class defines an artificial neural network based on a 2D CNN and 
    a series of fully connected layers. The CNN layers will perform the feature
    extraction of the input (images), and the feed forward neural networks will
    perform the classification of the images. 
"""
class ImageNet(torch.nn.Module):
    def __init__(self, input_dims, n_channels=1): # config for wandb tuning dropout
        super(ImageNet, self).__init__()
        self.dropout = 0.1
        self.kernel_size = 10 # We are only using square kernels
        self.kernel = [self.kernel_size, self.kernel_size] # nxn kernel
        self.stride = 1
        self.n_channels = n_channels # 1 for grayscale or 3 for rgb
        self.height = input_dims[0] # input image height
        self.width = input_dims[1] # input image width
        self.pad = [0,0,0,0] # top, bottom, left, right padding
        conv_channels = 12
        
        # input convolutional layer
        self.conv1 = nn.Conv2d(in_channels=self.n_channels, out_channels=conv_channels, 
                               kernel_size=self.kernel, stride=1, bias=True)

        # output vector size of previous layer -> input vector to next layer
        self.height = (self.height + self.pad[0] + self.pad[1] - self.kernel_size) /self.stride + 1
        self.width = (self.width + self.pad[2] + self.pad[3] - self.kernel_size) /self.stride + 1 
        self.output_shape = self.width*self.height
        
        # 1st pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride)
        self.height = (self.height + self.pad[0] + self.pad[1] - self.kernel_size) /self.stride + 1
        self.width = (self.width + self.pad[2] + self.pad[3] - self.kernel_size) /self.stride + 1 
        self.output_shape = self.width*self.height
        
        # 2nd Convolutional layer
        self.kernel_size=6
        self.kernel = [self.kernel_size, self.kernel_size]
        self.conv2 = nn.Conv2d(in_channels=conv_channels, out_channels=1, kernel_size=self.kernel,
                               stride=1, bias=True)
        self.height = (self.height + self.pad[0] + self.pad[1] - self.kernel_size) /self.stride + 1
        self.width = (self.width + self.pad[2] + self.pad[3] - self.kernel_size) /self.stride + 1 
        self.output_shape = self.width*self.height
        
        # 2nd pooling layer
        self.stride = 3
        self.pool2 = nn.MaxPool2d(kernel_size=self.kernel, stride=self.stride)
        self.height = (self.height + self.pad[0] + self.pad[1] - self.kernel_size) /self.stride + 1
        self.width = (self.width + self.pad[2] + self.pad[3] - self.kernel_size) /self.stride + 1 
        self.output_shape = self.width*self.height
        
        print('Final height: ', self.height)
        print('Final width: ', self.width)
        print('=====\n Linear Layer Input Layer size: ', int(self.output_shape), '\n======\n')
        
        self.linear_layer = nn.Linear(int(self.output_shape), 1)
        #self.sigmoid = nn.Sigmoid()
               
    def forward(self, x):
        # apply convolutional layer and ReLU
        x = self.conv1(x)
        # apply max pooling layer
        x = self.pool1(x)
        # apply 2nd convolutional layer
        x = self.conv2(x)
        # apply 2nd max pooling layer
        x = self.pool2(x)
        # reshape output for a fully connected layer
        x=x.view(x.size(0),-1)
        x = self.linear_layer(x)
        #x = self.sigmoid(x)
        #print('Output: ', x)
        return x