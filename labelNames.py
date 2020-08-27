#Developer: Dillon Pulliam
#Date: 8/14/2020
#Purpose: The purpose of this file is to print all car labels in the dataset


#Libraries used
import numpy as np
import scipy.io as sio


#Name:          getLabelNames
#Purpose:       load all labels from the dataset
#Inputs:        none
#Output:        labelNames -> all label names from the dataset
def getLabelNames():
    #Load the label names, get the number of labels, and initialize a numpy array to store all labels
    annos = sio.loadmat('data/car_devkit/devkit/cars_meta.mat')
    _, total_size = annos["class_names"].shape
    labelNames = np.ndarray(shape=(total_size, 2), dtype=object)

    #Loop through all labels adding them to the numpy array
    for i in range(total_size):
        labelName = annos["class_names"][0][i][0]
        labelNames[i,0] = i+1
        labelNames[i,1] = labelName
    return labelNames


#Main function
if __name__ == '__main__':
    #Load the label names and print the results
    labelNames = getLabelNames()
    labelNames[:,0] = labelNames[:,0] - 1
    print(labelNames)
