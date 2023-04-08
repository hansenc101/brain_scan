# -*- coding: utf-8 -*-
"""
Christopher Hansen
Artificial Neural Networks
"""

import ImageNet as IN
#import LightningLearning as LL
import TorchUtils as tu
import numpy as np
import pytorch_lightning as pl
#import wandb
#from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import torch
#%% Load Data and Prepare for Training
# Load the image
dims = [28, 28]
image = torch.randn(1, dims[0], dims[1]) # A 28x28 random greyscale image

#%% Create the neural network model 
model = IN.ImageNet(dims)

# Make a prediction
output = model(image)

# Print the prediction
print(output)