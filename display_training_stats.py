#Developer: Dillon Pulliam
#Date: 8/27/2020
#Purpose: The purpose of this file is to display the training / testing results in a plot


#Libraries used
import argparse
import torch
import matplotlib.pyplot as plt


#Main function
if __name__ == '__main__':
    #Get the stats filename to load
    parser = argparse.ArgumentParser(description="process the command line arguments")
    parser.add_argument("--file", type=str, required=True, help='stats filename to load')
    args = parser.parse_args()

    #Load all stats
    stats = torch.load(args.file)

    #Get the training set results per epoch
    trainLoss = stats['average_loss_train']
    trainAccuracy = stats['accuracy_train']
    trainAccuracy5 = stats['accuracy_train_5']

    #Get the test set results per epoch
    testLoss = stats['average_loss_test']
    testAccuracy = stats['accuracy_test']
    testAccuracy5 = stats['accuracy_test_5']

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
