# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:23:13 2023

@author: hanse
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader,Dataset
import torch.utils.data as data
import numpy as np
import os
from PIL import Image


def load_images(image_directory, image_height=512, image_width=512):
    # Define the desired width and height of the downscaled image
    new_width = image_width
    new_height = image_height
    
    image_arrays = []

    # Iterate over the files in the folder
    for filename in os.listdir(image_directory):
        # Construct the full file path
        file_path = os.path.join(image_directory, filename)
        
        # Load the image using PIL
        image = Image.open(file_path)
        
        # Convert the image to grayscale
        image = image.convert("L")
        
        # Downscale the image (if larger than requested dimensions)
        image = image.resize((new_width, new_height)) 

        # Convert the image to a NumPy array and append it to the list
        image_array = np.array(image)
        image_arrays.append(image_array)
    
    return image_arrays


class ImageDataset(Dataset): # I think this should be good now =========
    def __init__(self,X,Y,days): # X and Y must be np.array() types
        self.X = X # array of [height x width] greyscale images    
        self.Y = Y # 1 or 0
        
    def __len__(self):
        return (len(self.Y))
        
    def __getitem__(self,index):
        x=self.X[index]
        y=self.Y[index]
        return x,y

class ScanDataModule(pl.LightningDataModule): # NEEDS FIXING
    def __init__(self, ImageData, batch_size = 0):
        train_set_size = int(len(ImageData)*0.7)
        valid_set_size = int(len(ImageData)*0.15)
        test_set_size = len(ImageData)-train_set_size-valid_set_size

        self.train_set, self.valid_set, self.test_set = data.random_split(ImageData,[train_set_size,valid_set_size,test_set_size],\
                                                      generator=torch.Generator().manual_seed(42))  
        if batch_size==0:
            self.batch_size = test_set_size # use entire dataset as batch
        else:
            self.batch_size = batch_size
    
    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_set,batch_size=self.batch_size,shuffle=True)  # input:(512,512), label:1
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(self.valid_set,batch_size=self.batch_size,shuffle=False)  # input:(512,512), label:1
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_set,batch_size=self.batch_size,shuffle=False)  # input:(512,512), label:1
        return test_dataloader