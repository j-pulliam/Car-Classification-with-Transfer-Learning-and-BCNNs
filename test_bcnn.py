#Developer: Dillon Pulliam
#Date: 9/6/2020
#Purpose: The purpose of this file is to evaluate the performance of a trained BCNN over both the training and test datasets


#Libraries used
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#Local imports needed
from dataset import *
from utils import *
from model_bcnn import *


#Main function
if __name__ == '__main__':
    #Get the model filename, batch size to use for evaluation, and print count for class stats
    parser = argparse.ArgumentParser(description="process the command line arguments")
    parser.add_argument("--model", type=str, required=True, help='filename the model is saved to')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size to use for evaluation')
    parser.add_argument("--print_count", type=int, default=10, help='number of top / least class stats to print')
    args = parser.parse_args()

    #Load the model, build the PyTorch NN, load in its parameters, and create the criterion
    model = torch.load(args.model)
    NN = BCNN(model['model1'], model['model2'], model['resize_shape'], False)
    NN.load_state_dict(model['model_state_dict'])
    criterion = nn.CrossEntropyLoss()

    #Send the network and criterion to the GPU if applicable
    GPU = getGPU()
    NN = transferDevice(GPU, NN)
    criterion = transferDevice(GPU, criterion)

    #Create label dictionaries for both the training and test datasets and get all label names
    trainLabels = getLabels("data/car_devkit/devkit/cars_train_annos.mat")
    trainLabels = buildLabelDictionary(trainLabels)
    testLabels = getLabels("data/car_devkit/devkit/cars_test_annos_withlabels.mat")
    testLabels = buildLabelDictionary(testLabels)
    labelNames = getLabelNames()

    #Create the dataset / dataloader for both the training and test data
    trainDataset = CarDataset("data/cars_train", model['resize_shape'])
    trainDataloader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testDataset = CarDataset("data/cars_test", model['resize_shape'])
    testDataloader = DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #Evaluate the performance of the network over both the training and test sets
    trainLoss, trainAccuracy, trainTop5Accuracy, trainCarStats = test(GPU, NN, trainDataloader, trainLabels, labelNames, criterion, args.batch_size)
    testLoss, testAccuracy, testTop5Accuracy, testCarStats = test(GPU, NN, testDataloader, testLabels, labelNames, criterion, args.batch_size)

    #Print the training set results
    print("Training loss: "+str(trainLoss)+" --- Training accuracy: "+str(trainAccuracy)+" --- Training top 5 accuracy: "+str(trainTop5Accuracy))
    printClassStats(trainCarStats, args.print_count)

    #Print the test set results
    print("Testing loss: "+str(testLoss)+" --- Testing accuracy: "+str(testAccuracy)+" --- Testing top 5 accuracy: "+str(testTop5Accuracy))
    printClassStats(testCarStats, args.print_count)
