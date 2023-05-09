# -*- coding: utf-8 -*-
"""
Christopher Hansen
Artificial Neural Networks
"""

import ImageNet as IN
from PIL import Image
#import LightningLearning as LL
import TorchUtils as tu
import numpy as np
import pytorch_lightning as pl
#import wandb
#from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import torch
#%% Load Data and Prepare for Training
#no_tumor_train_path = '/media/chris/New Volume/brain_scan/brain_data/train/notumor'
#tumor_train_path = '/media/chris/New Volume/brain_scan/brain_data/train/tumor'
#no_tumor_test_path = '/media/chris/New Volume/brain_scan/brain_data/test/notumor'
#tumor_test_path = '/media/chris/New Volume/brain_scan/brain_data/test/tumor'

no_tumor_path = '/media/chris/New Volume/brain_scan/data/no_tumor'
tumor_path = '/media/chris/New Volume/brain_scan/data/tumor'
#%% Load images into an np array and create the target value arrays (0=no tumor or 1=tumor)
#train_tumors_x = tu.load_images(image_directory=tumor_train_path)
#train_notumors_x = tu.load_images(image_directory=no_tumor_train_path)
#test_tumors_x = tu.load_images(image_directory=tumor_test_path)
#test_notumors_x = tu.load_images(image_directory=no_tumor_test_path)
#train_tumors_y = np.ones(len(train_tumors_x))
#train_notumors_y = np.zeros(len(train_notumors_x))
#test_tumors_y = np.ones(len(test_tumors_x))
#test_notumors_y = np.zeros(len(test_notumors_x))

tumors = tu.load_images(tumor_path)
no_tumors = tu.load_images(no_tumor_path)
y_tumors = np.ones(len(tumors))
y_no_tumors = np.zeros(len(no_tumors))

#%% Combine into two lists: train and test
tumors.extend(no_tumors) # All x inputs
y = np.concatenate((y_tumors, y_no_tumors)) # All train y targets

#%% Create the dataset and dataloaders

#%% Dummy Test -> Load the image
dims = [512, 512]
n_channels = 1 # 1 for greyscale, 3 for rgb; default is 1 in ImageNet
image = tumors[0]
image = torch.tensor(image).float().unsqueeze(0)
#%% Create the neural network model 
model = IN.ImageNet(dims,n_channels=n_channels)

# Make a prediction
output = model(image)

# Print the prediction
print(output)