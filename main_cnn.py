#Developer: Dillon Pulliam
#Date: 8/27/2020
#Purpose: The purpose of this file is to train a CNN over the training dataset and
#         perform evaluation over the test set


#Libraries used
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

#Local imports needed
from dataset import *
from utils import *


#Main function
if __name__ == '__main__':
    #Get the model type to use, image resize shape, whether to perform fine-tuning or feature extraction, batch size, and max epochs to train for
    parser = argparse.ArgumentParser(description="process the command line arguments")
    parser.add_argument("--model", type=int, default=1, help='Numerical value corresponding to the type of CNN model to train')
    parser.add_argument("--resize_shape", type=int, default=224, help='image resize shape')
    parser.add_argument("--feature_extract", action='store_true', help='whether or not to perform feature extraction (false = finetune whole model; true = update reshaped layer parameters only)')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size to use for training and evaluation')
    parser.add_argument("--epochs", type=int, default=5, help='max epochs to train for')
    args = parser.parse_args()

    #Create label dictionaries for both the training and test datasets
    trainLabels = getLabels("data/car_devkit/devkit/cars_train_annos.mat")
    trainLabels = buildLabelDictionary(trainLabels)
    testLabels = getLabels("data/car_devkit/devkit/cars_test_annos_withlabels.mat")
    testLabels = buildLabelDictionary(testLabels)

    #Create the dataset / dataloader for both the training and test data
    trainDataset = CarDataset("data/cars_train", args.resize_shape)
    trainDataloader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataset = CarDataset("data/cars_test", args.resize_shape)
    testDataloader = DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #Build the PyTorch NN and criterion
    NN = createModel(args.model, args.feature_extract)
    criterion = nn.CrossEntropyLoss()

    #Send the network and criterion to the GPU if applicable and create the optimizer
    GPU = getGPU()
    NN = transferDevice(GPU, NN)
    criterion = transferDevice(GPU, criterion)
    optimizer = torch.optim.SGD(NN.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-03,weight_decay=0, amsgrad=False)

    #Store the average training / testing loss per epoch and accuracy values
    averageLossTrain = np.zeros(shape=(args.epochs, 2), dtype=float)
    accuracyTrain = np.zeros(shape=(args.epochs, 2), dtype=float)
    accuracyTrain5 = np.zeros(shape=(args.epochs, 2), dtype=float)
    averageLossTest = np.zeros(shape=(args.epochs, 2), dtype=float)
    accuracyTest = np.zeros(shape=(args.epochs, 2), dtype=float)
    accuracyTest5 = np.zeros(shape=(args.epochs, 2), dtype=float)
    testAccuracyBest = 0

    #Loop through epochs training and evaluating the network
    for epochIndex in range(args.epochs):
        #Train the network over the training set and test over the test set
        NN, trainLoss, trainAccuracy, trainTop5Accuracy = train(GPU, NN, trainDataloader, trainLabels, criterion, optimizer, args.batch_size)
        testLoss, testAccuracy, testTop5Accuracy = test(GPU, NN, testDataloader, testLabels, criterion, args.batch_size)

        #Print the training / testing stats from the epoch
        print("                                                                                             ", end="\r")
        print("Epoch: "+str(epochIndex))
        print("Training loss: "+str(trainLoss)+" --- Training accuracy: "+str(trainAccuracy)+" --- Training top 5 accuracy"+str(trainTop5Accuracy))
        print("Testing loss: "+str(testLoss)+" --- Testing accuracy: "+str(testAccuracy)+" --- Testing top 5 accuracy"+str(testTop5Accuracy))
        print()

        #Save the training / testing results from the epoch
        saveResults(averageLossTrain, trainLoss, epochIndex, "trainLoss")
        saveResults(accuracyTrain, trainAccuracy, epochIndex, "trainAccuracy")
        saveResults(accuracyTrain5, trainTop5Accuracy, epochIndex, "trainTop5Accuracy")
        saveResults(averageLossTest, testLoss, epochIndex, "testLoss")
        saveResults(accuracyTest, testAccuracy, epochIndex, "testAccuracy")
        saveResults(accuracyTest5, testTop5Accuracy, epochIndex, "testTop5Accuracy")

        #Save the model only if test set accuracy has improved
        if(testAccuracy > testAccuracyBest):
            torch.save(NN.state_dict(), "NN.pt")
            testAccuracyBest = testAccuracy
