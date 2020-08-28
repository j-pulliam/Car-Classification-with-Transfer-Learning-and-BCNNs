#Developer: Dillon Pulliam
#Date: 8/25/2020
#Purpose: The purpose of this file is to view an image from the training dataset and its corresponding label


#Libraries used
import random
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#Local imports needed
from utils import *


#Name:          viewImage
#Purpose:       resize and view the image based on the random file index specified
#Inputs:        path -> path to the actual folder containing all the training images
#               idx -> random index of the file to visualize
#               lables -> numpy array of all file names and corresponding labels
#               resizeShape -> shape to resize the image to for visualization
#Output:        none -> just visualizes the image
def viewImage(path, idx, labels, resizeShape):
    #Read in the image
    image_names = os.listdir(path)
    im = cv2.imread(path + "/" + image_names[idx])[:,:,::-1]
    print("Image: ", image_names[idx])

    #Reshape the image
    w, h, ch = im.shape
    print("Orignal shape:" , w, h)
    im = cv2.resize(im,(resizeShape[1],resizeShape[0]),interpolation=cv2.INTER_LINEAR)
    w, h, ch = im.shape
    print("Resized shape:" , w, h)

    #Print the image and label
    index = np.where(labels == image_names[idx])
    index = index[0]
    print("Label: ", labels[index][0][1])
    plt.imshow(im)
    plt.show()
    return


#Main function
if __name__ == '__main__':
    #Get all car image file names from the training dataset and their corresponding labels
    trainLabels = getLabels("data/car_devkit/devkit/cars_train_annos.mat")

    #Resze shape for the image and the random file to visualize
    resizeShape = [224, 224]
    imageNumber = random.randint(0, len(trainLabels))

    #View the image
    viewImage("data/cars_train", imageNumber, trainLabels, resizeShape)
