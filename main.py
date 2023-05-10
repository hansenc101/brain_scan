# -*- coding: utf-8 -*-
"""
Christopher Hansen
Artificial Neural Networks
"""

import LightningLearning as LL
import TorchUtils as tu
import numpy as np
import pytorch_lightning as pl
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
model = LL.StockLightningModule(input_dims=dims, n_channels=1, learning_rate=3e-4)

#%% Prepare system - determine if using gpu or cpu
#used_gpu = False
num_epochs = 2
if torch.cuda.is_available():
  #used_gpu = True
  print('\n--Using GPU for training--\n')
  torch.set_default_tensor_type(torch.FloatTensor)
  torch.backends.cudnn.benchmark = True
  torch.set_float32_matmul_precision('high')
  trainer = pl.Trainer(accelerator='gpu', max_epochs=num_epochs, 
                       callbacks=[bar(refresh_rate=1)], log_every_n_steps=20)
else:
  print('\n--Using CPU for training--\n')
  trainer = pl.Trainer(max_epochs=num_epochs, 
                       callbacks=[bar(refresh_rate=1)], log_every_n_steps=20)

#%% Train the model
print('Training Model...')
trainer.fit(model, train_dataloader, val_dataloader)

#%% Gather loss data
train_data = model.get_train_loss_data()
val_data = model.get_val_loss_data()

# get loss data off of gpu to cpu and convert to np.arrays
train_data = np.array([tensor.cpu().detach().numpy() for tensor in train_data])
val_data = np.array([tensor.cpu().detach().numpy() for tensor in val_data])
min_train = np.min(train_data) 
min_val = np.min(val_data) 
final_train = train_data[-1]
final_val = val_data[-1]

print('Final Training Loss Error: ', final_train)
print('Final Validation Loss Error: ', final_val)

#%% Evaluate the performance of the model
accuracy_ls = [] # list to hold number of predictions we got correct
trained_model = model.get_model()
print('Running test dataset...')
length = len(test_dataloader)
#length = 100
j = 0
for x, target in test_dataloader:
    j = j+1
    x=x.float().unsqueeze(1)
    pred = model.forward(x.float())
    pred = pred.squeeze().detach().numpy()
    target = target.detach().numpy()
    
    for i in range(len(pred)):
        if  pred[i] > 0.5:
            pred[i] = 1
        else:
            pred[i] = 0

        ## check prediction accuracy
        if pred[i] == target[i]:
          accuracy_ls.append(1)
        else:
          accuracy_ls.append(0)
    
    percentage = 100*((j+1)/length)
    print(f"{percentage:.0f}% done...", end="\r")

## check testing accuracy
print(f"Testing dataset accuracy: {100*(sum(accuracy_ls)/len(accuracy_ls)):.2f}%")
i#nput('Press [Enter] to exit...')

#%%

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