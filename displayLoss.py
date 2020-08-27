#Developer: Dillon Pulliam
#Date: 8/27/2020
#Purpose: The purpose of this file is to display the training / testing results in a plot


#Libraries used
import numpy as np
import matplotlib.pyplot as plt


#Main function
if __name__ == '__main__':
    #Loading the training set results per epoch
    trainLoss = np.load('trainLoss.npy')
    trainAccuracy = np.load('trainAccuracy.npy')
    trainAccuracy5 = np.load('trainTop5Accuracy.npy')

    #Load the test set results per epoch
    testLoss = np.load('testLoss.npy')
    testAccuracy = np.load('testAccuracy.npy')
    testAccuracy5 = np.load('testTop5Accuracy.npy')

    #Type of plot to use
    plt.style.use('seaborn-whitegrid')

    #Plot the train / test loss per epoch
    plt.plot(trainLoss[:,0], trainLoss[:,1], color='black', label='Training')
    plt.plot(testLoss[:,0], testLoss[:,1], color='blue', label='Test')
    plt.legend()
    plt.ylim(0, 6)
    plt.ylabel("Average Cross Entropy Loss")
    plt.xlabel("Epoch Number")
    plt.title('Cross Entropy Loss vs Epoch')
    plt.show()

    #Plot the train / test accuracy per epoch
    plt.plot(trainAccuracy[:,0], trainAccuracy[:,1]*100, color='black', label='Training')
    plt.plot(testAccuracy[:,0], testAccuracy[:,1]*100, color='blue', label='Test')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch Number")
    plt.title('Top Prediction Accuracy vs Epoch')
    plt.show()

    #Plot the train / test top 5 accuracy per epoch
    plt.plot(trainAccuracy5[:,0], trainAccuracy5[:,1]*100, color='black', label='Training')
    plt.plot(testAccuracy5[:,0], testAccuracy5[:,1]*100, color='blue', label='Test')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch Number")
    plt.title('Top 5 Prediction Accuracy vs Epoch')
    plt.show()
