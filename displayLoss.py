#Libraries used
import numpy as np
import matplotlib.pyplot as plt

#Main function
if __name__ == '__main__':
    trainLoss = np.load('trainLoss.npy')
    trainAccuracy = np.load('trainAccuracy.npy')
    #newTrainAccuracy = 1 - trainAccuracy[:,1]
    #trainAccuracy[:,1] = newTrainAccuracy
    trainAccuracy5 = np.load('trainAccuracyTop5.npy')
    #newTrainAccuracy5 = 1 - trainAccuracy5[:,1]
    #trainAccuracy5[:,1] = newTrainAccuracy5
    valLoss = np.load('testLoss.npy')
    valAccuracy = np.load('testAccuracy.npy')
    #newValAccuracy = 1 - valAccuracy[:,1]
    #valAccuracy[:,1] = newValAccuracy
    valAccuracy5 = np.load('testAccuracyTop5.npy')
    #newValAccuracy5 = 1 - valAccuracy5[:,1]
    #valAccuracy5[:,1] = newValAccuracy5

    plt.style.use('seaborn-whitegrid')


    plt.plot(trainLoss[:,0], trainLoss[:,1], color='black', label='Training')
    plt.plot(valLoss[:,0], valLoss[:,1], color='blue', label='Test')
    plt.legend()
    plt.ylim(0, 6)
    plt.ylabel("Average Cross Entropy Loss")
    plt.xlabel("Epoch Number")
    plt.title('Cross Entropy Loss vs Epoch')
    plt.show()

    plt.plot(trainAccuracy[:,0], trainAccuracy[:,1]*100, color='black', label='Training')
    plt.plot(valAccuracy[:,0], valAccuracy[:,1]*100, color='blue', label='Test')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch Number")
    plt.title('Top Prediction Accuracy vs Epoch')
    plt.show()

    plt.plot(trainAccuracy5[:,0], trainAccuracy5[:,1]*100, color='black', label='Training')
    plt.plot(valAccuracy5[:,0], valAccuracy5[:,1]*100, color='blue', label='Test')
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch Number")
    plt.title('Top 5 Prediction Accuracy vs Epoch')
    plt.show()
