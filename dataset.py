#Developer: Dillon Pulliam
#Date: 8/25/2020
#Purpose: The purpose of this file is to give the PyTorch dataset class definition


#Libraries needed
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


#PyTorch dataset class definition
class CarDataset(Dataset):
    #Name:          __init__
    #Purpose:       initialize the class variables for the dataset
    #Inputs:        root_dir -> directory containing the dataset files
    #               resizeShape -> resize value of an image
    #Output:       none
    def __init__(self, root_dir, resizeShape):
        self.root_dir = root_dir
        self.resizeShape = resizeShape

    #Name:          __len__
    #Purpose:       get the number of images in the dataset
    #Inputs:        none
    #Output:       counter -> number of images in the dataset
    def __len__(self):
        return len(os.listdir(self.root_dir))

    #Name:          __getitem__
    #Purpose:       load a random image from the dataset
    #Inputs:        idx -> random index value of the image to load
    #Outputs:       image -> image
    #               imageName -> image name
    def __getitem__(self, idx):
        #Read the image and resize it properly
        imageName = os.listdir(self.root_dir)
        image = cv2.imread(self.root_dir + "/" + imageName[idx])[:,:,::-1]
        image = cv2.resize(image,(self.resizeShape,self.resizeShape),interpolation=cv2.INTER_LINEAR)
        #Convert the image to float and move the RGB channel to index 0
        image = image.astype(float)
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 0, 1)
        #Normalize the image
        image /= 255
        return torch.from_numpy(image).float(), imageName[idx]
