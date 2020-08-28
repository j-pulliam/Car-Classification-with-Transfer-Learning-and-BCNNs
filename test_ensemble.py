#Developer: Dillon Pulliam
#Date: 8/27/2020
#Purpose: The purpose of this file is to evaluate the performance of an ensemble of trained CNNs over both the training and test datasets


#Libraries used
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#Local imports needed
from dataset import *
from utils import *


#Name:          testEnsemble
#Purpose:       test the ensemble of networks
#Inputs:        GPU -> GPU device if applicable, none if no
#               NN1 -> PyTorch neural network class
#               NN2 -> PyTorch neural network class
#               NN3 -> PyTorch neural network class
#               dataset -> PyTorch dataloader class
#               labels -> dictionary of file names and their corresponding labels in a dataset
#               batchSize -> batch size for testing
#Output:        loss -> average loss across the test set
#               accuracy -> accuracy across the test set
#               top5Accuracy -> top 5 accuracy across the test set
def testEnsemble(GPU, NN1, NN2, NN3, dataset, labels, batchSize):
    #Place the networks in evaluation mode
    NN1.eval()
    NN2.eval()
    NN3.eval()

    #Create variables to store the correct predictions and top 5 correct predictions
    correctPredictions = 0
    top5CorrectPredictions = 0

    #Create variable to store the stats per label type [label, labelName, correctTop1, correctTop5, total]
    carStats = np.zeros(shape=(len(labelNames), 5), dtype=object)
    carStats[:,0:2] = labelNames

    #Iterate through the dataset testing our ensemble
    for batchNumber, (batchData, batchImageNames) in enumerate(dataset):
        #Transfer the batch data to the GPU and get the batch labels while also transferring it
        batchData = transferDevice(GPU, batchData)
        batchLabels = transferDevice(GPU, getImageLabels(batchImageNames, labels))
        #Pass the images through the networks, make predictions, and determine the number of
        #correctly predicted labels and correctly predicted top 5 labels
        output1 = NN1(batchData)
        output2 = NN2(batchData)
        output3 = NN3(batchData)
        output = output1 + output2 + output3
        batchCorrectPredictions, batchCorrectIndices = calcCorrectPredictions(output, batchLabels)
        batchTop5CorrectPredictions, batchTop5CorrectIndices = calcCorrectPredictions5(output, batchLabels)
        correctPredictions += batchCorrectPredictions
        top5CorrectPredictions += batchTop5CorrectPredictions
        #Increment the total label counts, correct counts, and top 5 correct counts corresponding to each label in car stats
        carStats = updateCarStats(carStats, batchLabels, batchCorrectIndices, batchTop5CorrectIndices)
        print("                                                                                             ", end="\r")
        print("Testing batch index: "+str(batchNumber)+"/"+str(len(dataset))+ " ( "+str(batchNumber/len(dataset)*100)+"% )", end="\r")

    #Convert count stats to accuracy values
    carStats[:,2] /= carStats[:,4]
    carStats[:,3] /= carStats[:,4]

    #Return the top prediction accuracy, top 5 prediction accuracy, and car stats
    return correctPredictions/len(labels), top5CorrectPredictions/len(labels), carStats


#Main function
if __name__ == '__main__':
    #Get the model #1, #2, and #3 filenames, get the batch size to use for evaluation, get the print count for class stats,
    #and determine if a GPU should be utilized
    parser = argparse.ArgumentParser(description="process the command line arguments")
    parser.add_argument("--model1", type=str, required=True, help='filename model1 is saved to')
    parser.add_argument("--model2", type=str, required=True, help='filename model2 is saved to')
    parser.add_argument("--model3", type=str, required=True, help='filename model3 is saved to')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size to use for evaluation')
    parser.add_argument("--print_count", type=int, default=10, help='number of top / least class stats to print')
    parser.add_argument("--GPU", action='store_true', help='whether or not to use the GPU for processing as memory could be a constraint')
    args = parser.parse_args()

    #Get the GPU if specified by the user and avaliable
    GPU = None
    if(args.GPU):
        GPU = getGPU()

    #Load model1, build the PyTorch NN, load in its parameters, and send the network to the GPU if applicable
    model1 = torch.load(args.model1)
    NN1 = createModel(model1['model'], False)
    NN1.load_state_dict(model1['model_state_dict'])
    NN1 = transferDevice(GPU, NN1)

    #Load model2, build the PyTorch NN, load in its parameters, and send the network to the GPU if applicable
    model2 = torch.load(args.model2)
    NN2 = createModel(model2['model'], False)
    NN2.load_state_dict(model2['model_state_dict'])
    NN2 = transferDevice(GPU, NN2)

    #Load model3, build the PyTorch NN, load in its parameters, and send the network to the GPU if applicable
    model3 = torch.load(args.model3)
    NN3 = createModel(model3['model'], False)
    NN3.load_state_dict(model3['model_state_dict'])
    NN3 = transferDevice(GPU, NN3)

    #Check that the resize shape is equivalent across all models
    if not(model1['resize_shape'] == model2['resize_shape'] == model3['resize_shape']):
        print("Error! Image resize shape not equivalent across all models")
        exit()

    #Create label dictionaries for both the training and test datasets and get all label names
    trainLabels = getLabels("data/car_devkit/devkit/cars_train_annos.mat")
    trainLabels = buildLabelDictionary(trainLabels)
    testLabels = getLabels("data/car_devkit/devkit/cars_test_annos_withlabels.mat")
    testLabels = buildLabelDictionary(testLabels)
    labelNames = getLabelNames()

    #Create the dataset / dataloader for both the training and test data
    trainDataset = CarDataset("data/cars_train", model1['resize_shape'])
    trainDataloader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testDataset = CarDataset("data/cars_test", model1['resize_shape'])
    testDataloader = DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    #Evaluate the performance of the ensemble of networks over both the training and test sets
    trainAccuracy, trainTop5Accuracy, trainCarStats = testEnsemble(GPU, NN1, NN2, NN3, trainDataloader, trainLabels, args.batch_size)
    testAccuracy, testTop5Accuracy, testCarStats = testEnsemble(GPU, NN1, NN2, NN3, testDataloader, testLabels, args.batch_size)

    #Print the training set results
    print("Training accuracy: "+str(trainAccuracy)+" --- Training top 5 accuracy: "+str(trainTop5Accuracy))
    printClassStats(trainCarStats, args.print_count)

    #Print the test set results
    print("Testing accuracy: "+str(testAccuracy)+" --- Testing top 5 accuracy: "+str(testTop5Accuracy))
    printClassStats(testCarStats, args.print_count)
