#Developer: Dillon Pulliam
#Date: 8/28/2020
#Purpose: The purpose of this file is to print all car labels in the dataset


#Libraries used
import numpy as np
import scipy.io as sio

#Local imports needed
from utils import *


#Main function
if __name__ == '__main__':
    #Load the label names and print the results
    labelNames = getLabelNames()
    print(labelNames)
