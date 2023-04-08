# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:11:48 2023

@author: hanse
"""

"""
1a. Create a subclass of pytorch_lightning.LightningModule. It should include 
__init__, training_step, validation_step, configure_optimizers in the class. 
(6 points)

"""

from torch import optim, nn, stack
import torch
import pytorch_lightning as pl
import ConvNet1D as CN1D

# Computational Code goes into LightningModule
class StockLightningModule(pl.LightningModule):
    def __init__(self, config, days): # model architecture into __init()__
        super().__init__()
        # our untrained neural network
        self.net = CN1D.ConvNet1D(config, days)
        self.lr = config.lr # wandb tuning for learning rate
        self.val_loss_data = []
        self.train_loss_data = []
        self.test_loss_data = []
        self.test_y_data = []
        self.test_output_data = []
    
    # Set forward hook; in lightning, forward() defines the prediction
    # and interference actions
    def forward(self, x):
        embedding = self.net(x)
        return embedding
    
    # Training logic goes into training_step LightningModule hook
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y.view(y.size(0),-1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        #print("===\ntrain loss : ", loss,"\n===")
        self.train_loss_data.append(loss)
        return loss
    
    # Validation logic goes into validation_step LightningModule hook
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y.view(y.size(0),-1))
        # calling this from validation_step will automatically accumulate
        # and log at the end of the epoch
        self.log('val_loss', loss)
        #print("===\nval loss : ", loss,"\n===")
        self.val_loss_data.append(loss)
    
    # Optimizers go into configure_optimizers LightningModule hook
    # self.parameters will contain parameters from neural net
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        # This is the test loop
        x, y = batch
        #x = x.view(x.size(0),-1)
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y.view(y.size(0),-1))
        self.log("test_loss", loss)
        self.test_loss_data.append(loss)
        self.test_y_data.append(y)
        self.test_output_data.append(output)
    
    def get_train_data(self):
        return stack(self.train_loss_data)
    
    def get_val_data(self):
        return stack(self.val_loss_data)
    
    def get_test_data(self):
        return stack(self.test_loss_data)

    def get_test_y_data(self):
        return self.test_y_data

    def get_test_output_data(self):
        return self.test_output_data
    
    def save_model(self, name):
        # Open a file called "n_samps" in write mode
        with open("n_inputs", "w") as file:
            # Write the integer to the file as a string
            file.write(str(self.days)) #save number of inputs

        path = name + '.pth'
        torch.save(self.net.state_dict(), path)
        
""" I need to determine how to save and load the data
    def load_model(self, name):
        path = name + '.pth'
        model = 
        model.load_state_dict(torch.load('model.pth'))
        model.eval()  # Set the model to evaluation mode if needed
"""