# -*- coding: utf-8 -*-
"""
Christopher Hansen
Artificial Neural Networks
"""

import ImageNet as IN
from PIL import Image
import LightningLearning as LL
import TorchUtils as tu
import numpy as np
import pytorch_lightning as pl
#import wandb
#from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.callbacks import TQDMProgressBar as bar
#%% Load Data and Prepare for Training
no_tumor_path = '/media/chris/New Volume/brain_scan/data/no_tumor'
tumor_path = '/media/chris/New Volume/brain_scan/data/tumor'

#%% Load images into an np array and create the target value arrays (0=no tumor or 1=tumor)
tumors = tu.load_images(tumor_path)
no_tumors = tu.load_images(no_tumor_path)
y_tumors = np.ones(len(tumors))
y_no_tumors = np.zeros(len(no_tumors))

#%% Combine into two lists: train and test
tumors.extend(no_tumors) # All x inputs
y = np.concatenate((y_tumors, y_no_tumors)) # All train y targets

#%% Create the dataset and dataloaders
scan_dataset = tu.ImageDataset(tumors,y)

# set batch_size to 0 to use entire dataset as batch
batch_size = 24
brainScanModule = tu.ScanDataModule(scan_dataset, batch_size=batch_size) 
train_dataloader = brainScanModule.train_dataloader()
val_dataloader = brainScanModule.val_dataloader()
test_dataloader = brainScanModule.test_dataloader()

#%% Create the model
dims = [512, 512] # pixel dimension of the brain scans
model = LL.StockLightningModule(input_dims=dims, n_channels=1, learning_rate=1e-2)

#%% Prepare system - determine if using gpu or cpu
used_gpu = False
num_epochs = 3
if torch.cuda.is_available():
  used_gpu = True
  print('\n--Using GPU for training--\n')
  torch.set_default_tensor_type(torch.FloatTensor)
  torch.backends.cudnn.benchmark = True
  torch.set_float32_matmul_precision('high')
  trainer = pl.Trainer(accelerator='gpu', max_epochs=num_epochs, 
                       callbacks=[bar(refresh_rate=10)], log_every_n_steps=20)
else:
  print('\n--Using CPU for training--\n')
  trainer = pl.Trainer(max_epochs=num_epochs, 
                       callbacks=[bar(refresh_rate=10)], log_every_n_steps=20)

#%% Train the model
trainer.fit(model, train_dataloader, val_dataloader)

#%% Gather loss data
train_data = model.get_train_loss_data()
val_data = model.get_val_loss_data()
#val_pred = model.get_val_predictions()
#val_target = model.get_val_targets()

if used_gpu:
    # get loss data off of gpu to cpu and convert to np.arrays
    train_data = np.array([tensor.cpu().detach().numpy() for tensor in train_data])
    val_data = np.array([tensor.cpu().detach().numpy() for tensor in val_data])
    min_train = np.min(train_data) 
    min_val = np.min(val_data) 
    final_train = train_data[-1]
    final_val = val_data[-1]
    
    # get prediction and target data off of gpu to cpu and convert to np.arrays
    #val_pred = np.array([tensor.cpu().detach().numpy() for tensor in val_pred])
    #val_target = np.array([tensor.cpu().detach().numpy() for tensor in val_target])
else:
    # Take loss data and convert to np.arrays
    train_data = np.array([tensor.detach().numpy() for tensor in train_data])
    val_data = np.array([tensor.detach().numpy() for tensor in val_data])
    min_train = np.min(train_data) 
    min_val = np.min(val_data) 
    final_train = train_data[-1]
    final_val = val_data[-1]
    
    # get prediction and target data off of gpu to cpu and convert to np.arrays
    #val_pred = np.array([tensor.detach().numpy() for tensor in val_pred])
    #val_target = np.array([tensor.detach().numpy() for tensor in val_target])


#% Plot the training and validation loss curves
train_steps = np.linspace(0, len(train_data), len(train_data))
val_steps = np.linspace(0, len(train_data), len(val_data))
plt.plot(train_steps, train_data, label='Training Loss')
plt.plot(val_steps, val_data, label='Validation Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Testing and Validation Loss using Lightning')
plt.legend()
plt.show()
