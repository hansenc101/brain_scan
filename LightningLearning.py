# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:11:48 2023

@author: hanse
"""

from torch import optim, nn, stack
import torch
import pytorch_lightning as pl
import ImageNet as IN

# Computational Code goes into LightningModule
class StockLightningModule(pl.LightningModule):
    def __init__(self, input_dims, n_channels=1, learning_rate=1e-4): # model architecture into __init()__
        super().__init__()
        # our untrained neural network
        self.net = IN.ImageNet(input_dims=input_dims, n_channels=n_channels)
        self.lr = learning_rate
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
        x=x.float().unsqueeze(1)
        y=y.float()
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y.unsqueeze(1))
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_loss_data.append(loss)
        return loss
    
    # Validation logic goes into validation_step LightningModule hook
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x=x.float().unsqueeze(1)
        y=y.float()
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y.unsqueeze(1))
        self.log('val_loss', loss)
        self.val_loss_data.append(loss)
    
    # Optimizers go into configure_optimizers LightningModule hook
    # self.parameters will contain parameters from neural net
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        # This is the test loop
        x, y = batch
        x=x.float()
        y=y.float()
        output = self.net(x)
        loss = nn.functional.mse_loss(output,y)
        self.log("test_loss", loss)
        self.test_loss_data.append(loss)
        self.test_y_data.append(y)
        self.test_output_data.append(output)
    
    def get_train_loss_data(self):
        return stack(self.train_loss_data)
    
    def get_val_loss_data(self):
        return stack(self.val_loss_data)
    
    def get_test_loss_data(self):
        return stack(self.test_loss_data)

    def get_test_y_data(self):
        return self.test_y_data

    def get_test_output_data(self):
        return self.test_output_data
    
    def save_model(self, file_name='Model.pt'):
        torch.save(self.net, file_name)
