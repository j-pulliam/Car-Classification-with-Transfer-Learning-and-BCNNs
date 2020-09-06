#Developer: Dillon Pulliam
#Date: 8/27/2020
#Purpose: The purpose of this file is define utility functions used throughout the code base


#Libraries needed
import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn
from torchvision import models


#Name:          train
#Purpose:       train the network over a single epoch
#Inputs:        GPU -> GPU device if applicable, none if no
#               NN -> PyTorch neural network class
#               dataset -> PyTorch dataloader class
#               labels -> dictionary of file names and their corresponding labels in a dataset
#               criterion -> PyTorch criterion (such as cross entropy loss)
#               optimizer -> PyTorch optimizer (such as SGD / Adam)
#               batchSize -> batch size for training
#Output:        NN -> PyTorch neural network post training
#               loss -> average loss across the training run
#               accuracy -> accuracy across the training run
#               top5Accuracy -> top 5 accuracy across the training run
def train(GPU, NN, dataset, labels, criterion, optimizer, batchSize):
    #Place the network in training mode
    NN.train()

    #Create variables to store the total loss, correct predictions, and top 5 correct predictions
    totalLoss = 0
    correctPredictions = 0
    top5CorrectPredictions = 0

    #Iterate through the dataset training our model
    for batchNumber, (batchData, batchImageNames) in enumerate(dataset):
        #Transfer the batch data to the GPU and get the batch labels while also transferring it
        batchData = transferDevice(GPU, batchData)
        batchLabels = transferDevice(GPU, getImageLabels(batchImageNames, labels))
        #Pass the images through the network, make predictions, compute the loss, add this to the total loss, and
        #determine the number of correctly predicted labels and correctly predicted top 5 labels
        output = NN(batchData)
        loss = criterion(output, batchLabels)
        totalLoss += loss.item()
        batchCorrectPredictions, _ = calcCorrectPredictions(output, batchLabels)
        batchTop5CorrectPredictions, _ = calcCorrectPredictions5(output, batchLabels)
        correctPredictions += batchCorrectPredictions
        top5CorrectPredictions += batchTop5CorrectPredictions
        #Clear gradients from the previous training pass, compute the gradients of this pass, and update the
        #network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("                                                                                             ", end="\r")
        print("Training batch index: "+str(batchNumber)+"/"+str(len(dataset))+ " ( "+str(batchNumber/len(dataset)*100)+"% )", end="\r")

    #Return the updated network, average loss, top prediction accuracy, and top 5 prediction accuracy
    return NN, totalLoss/len(dataset), correctPredictions/len(labels), top5CorrectPredictions/len(labels)


#Name:          test
#Purpose:       test the network
#Inputs:        GPU -> GPU device if applicable, none if no
#               NN -> PyTorch neural network class
#               dataset -> PyTorch dataloader class
#               labels -> dictionary of file names and their corresponding labels in a dataset
#               labelNames -> all label names from the dataset
#               criterion -> PyTorch criterion (such as cross entropy loss)
#               batchSize -> batch size for testing
#Output:        loss -> average loss across the test set
#               accuracy -> accuracy across the test set
#               top5Accuracy -> top 5 accuracy across the test set
#               carStats -> numpy array of stats per label type [label, labelName, accuracyTop1, accuracyTop5, total]
def test(GPU, NN, dataset, labels, labelNames, criterion, batchSize):
    #Place the network in evaluation mode
    NN.eval()

    #Create variables to store the total loss, correct predictions, and top 5 correct predictions
    totalLoss = 0
    correctPredictions = 0
    top5CorrectPredictions = 0

    #Create variable to store the stats per label type [label, labelName, correctTop1, correctTop5, total]
    carStats = np.zeros(shape=(len(labelNames), 5), dtype=object)
    carStats[:,0:2] = labelNames

    #Iterate through the dataset testing our model
    for batchNumber, (batchData, batchImageNames) in enumerate(dataset):
        #Transfer the batch data to the GPU and get the batch labels while also transferring it
        batchData = transferDevice(GPU, batchData)
        batchLabels = transferDevice(GPU, getImageLabels(batchImageNames, labels))
        #Pass the images through the network, make predictions, compute the loss, add this to the total loss, and
        #determine the number of correctly predicted labels and correctly predicted top 5 labels along with their batch indices
        output = NN(batchData)
        loss = criterion(output, batchLabels)
        totalLoss += loss.item()
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

    #Return the average loss, top prediction accuracy, top 5 prediction accuracy, and car stats
    return totalLoss/len(dataset), correctPredictions/len(labels), top5CorrectPredictions/len(labels), carStats



#Name:          getLabels
#Purpose:       load all image files from the dataset and their corresponding labels
#Inputs:        path -> path to the metadata file
#Output:        labels -> all image names and their corresponding labels from the dataset
def getLabels(path):
    #Load the training files and labels, get the number of files, and initialize a numpy array to
    #store all names and labels
    annos = sio.loadmat(path)
    _, total_size = annos["annotations"].shape
    labels = np.ndarray(shape=(total_size, 2), dtype=object)

    #Loop through all files and labels adding them to the numpy array
    for i in range(total_size):
        fname = annos["annotations"][0][i][5][0]
        classLabel = annos["annotations"][0][i][4][0][0]
        labels[i,0] = fname
        labels[i,1] = classLabel

    #Ensure the index begins at 0
    labels[:,1] -= 1
    return labels


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

    #Set the start index to 0
    labelNames[:,0] = labelNames[:,0] - 1
    return labelNames


#Name:          buildLabelDictionary
#Purpose:       transform the file name with label array to a dictionary
#Inputs:        trainLabels -> numpy array with file names and their corresponding labels from the dataset
#Output:        dictionary -> dictionary of file names and their corresponding labels in a dataset
def buildLabelDictionary(trainLabels):
    dictionary = {}
    index = 0
    while(index < len(trainLabels)):
        dictionary[trainLabels[index][0]] = trainLabels[index][1]
        index += 1
    return dictionary


#Name:          createModel
#Purpose:       initialize the network, reshape the final layer, and determine if the entire network should be finetuned or just perform feature extraction
#Inputs:        model -> integer corresponding to the type of NN to build
#               feature_extract -> boolean specifying whether to finetune the entire network or only perform feature extraction
#Output:        NN -> PyTorch neural network class
def createModel(model, feature_extract):
    #Create a variable to store the NN
    NN = None

    #Initialize the network
    #Load in VGG11
    if(model == 1):
        NN = models.vgg11(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in VGG11_BN
    elif(model == 2):
        NN = models.vgg11_bn(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in VGG16
    elif(model == 3):
        NN = models.vgg16(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in VGG16_BN
    elif(model == 4):
        NN = models.vgg16_bn(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in Resnet18
    elif(model == 5):
        NN = models.resnet18(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.fc = nn.Linear(512, 196)
    #Load in Resnet50
    elif(model == 6):
        NN = models.resnet50(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.fc = nn.Linear(2048, 196)
    #Load in Resnet152
    elif(model == 8):
        NN = models.resnet152(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.fc = nn.Linear(2048, 196)
    #Load in DenseNet161
    elif(model == 8):
        NN = models.densenet161(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.fc = nn.Linear(1024, 196)
    #Load in ALexNet
    elif(model == 9):
        NN = models.alexnet(pretrained=True)
        set_parameter_requires_grad(NN, feature_extract)
        NN.classifier[6] = nn.Linear(4096, 196)
    else:
        print("Error! Model options are as follows:")
        print("1 -> VGG11")
        print("2 -> VGG11_BN")
        print("3 -> VGG16")
        print("4 -> VGG16_BN")
        print("5 -> Resnet18")
        print("6 -> Resnet50")
        print("7 -> Resnet152")
        print("8 -> Densenet161")
        print("9 -> Alexnet")
        exit()
    return NN


#Name:          set_parameter_requires_grad
#Purpose:       specifies whether or not the parameters of a NN should require a gradient
#Inputs:        model -> PyTorch NN class
#               feature_extracting -> boolean for whether or not a gradient is required
#Output:        none -> just modifies if a gradient is required for model parameters
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    return


#Name:          getGPU
#Purpose:       checks if a GPU device is avaliable
#Input:         none
#Output:        GPU -> GPU device if applicable, none if not
def getGPU():
    #Check if a GPU is avaliable and if so return it
    GPU = None
    if torch.cuda.is_available():
        print("Using GPU")
        GPU = torch.device("cuda")
    return GPU


#Name:          transferDevice
#Purpose:       transfers data to the GPU devie if present
#Inputs:        GPU -> GPU device if applicable, none if not
#               data -> data to transfer
#Output:        data -> data that has been transferred if applicable
def transferDevice(GPU, data):
    if(GPU != None):
        data = data.to(GPU)
    return data


#Name:          getImageLabels
#Purpose:       get the true class labels for a batch of data
#Inputs:        names -> names of the files in the batch
#               labels -> numpy array with file names and their corresponding labels from the dataset
#Output:        batchLabels -> labels for a batch of data
def getImageLabels(names, labels):
    #Create a variable to store the labels and get the corresponding label for each image in a batch
    batchLabels = np.zeros(len(names), dtype=int)
    idx = 0
    while(idx < len(names)):
        batchLabels[idx] = labels.get(names[idx])
        idx += 1
    return torch.from_numpy(batchLabels)


#Name:          calcCorrectPredictions
#Purpose:       calculate the number of correct predictions from the model
#Inputs:        outputs -> output predictions from the model
#               labels -> true labels
#Outputs:       correct -> number of correct predictions
#               correctIndices -> label indices the model predicted correctly
def calcCorrectPredictions(outputs, labels):
    #Variable to store the number of correct predictions, variable to store the correct prediction indices, and get the model class predictions
    correct = 0
    correctIndices = []
    predictedLabels = torch.argmax(outputs, dim=1)
    #Loop through all ouputs calculating the number correct and storing correct indices
    index = 0
    while(index < len(labels)):
        if(predictedLabels[index] == labels[index]):
            correct += 1
            correctIndices.append(index)
        index += 1
    return correct, correctIndices


#Name:          calcCorrectPredictions5
#Purpose:       calculate the number of correct predictions from the model where the correct prediction occurs within the top 5 classes
#Inputs:        outputs -> output predictions from the model
#               labels -> true labels
#Outputs:       correct -> number of correct predictions within the top 5
#               correctIndices -> label indices the model predicted correctly
def calcCorrectPredictions5(outputs, labels):
    #Variable to store the number of correct predictions, variable to store the correct prediction indices, and get the model top 5 class predictions
    correct = 0
    correctIndices = []
    predictedLabels = torch.topk(outputs, k=5, dim=1)
    #Loop through all ouputs calculating the number correct top 5 predictions and storing correct indices
    index = 0
    while(index < len(labels)):
        index2 = 0
        while(index2 < len(predictedLabels[1][0])):
            if(predictedLabels[1][index][index2] == labels[index]):
                correct += 1
                correctIndices.append(index)
            index2 += 1
        index += 1
    return correct, correctIndices


#Name:          saveResults
#Purpose:       save the training / testing results in the numpy array
#Inputs:        array -> numpy array to save the epoch index and data to
#               data -> data to save to the numpy array (loss, accuracy, or top 5 accuracy)
#               epochIndex -> epoch index for saving
#Output:        array -> numpy array with updated saved data
def saveResults(array, data, epochIndex):
    array[epochIndex][0] = epochIndex
    array[epochIndex][1] = data
    return array


#Name:          updateCarStats
#Purpose:       increment the total label counts, correct counts, and top 5 correct counts corresponding to each label in car stats
#Inputs:        carStats -> numpy array of stats per label type [label, labelName, correctTop1, correctTop5, total]
#               batchLabels -> labels for batch of data
#               correctIndices -> batch label indices the model predicted correctly
#               top5CorrectIndices -> batch label indices the model predicted correctly in the top 5
#Output:        carStats -> numpy array of stats per label type [label, labelName, correctTop1, correctTop5, total] updated
def updateCarStats(carStats, batchLabels, correctIndices, top5CorrectIndices):
    #Loop through all labels in the batch incrementing the count by 1 in car stats for each seen
    for label in batchLabels:
        carStats[label, 4] += 1

    #Loop through all correct predictions incrementing the label correct count by 1 in car stats
    for index in correctIndices:
        carStats[batchLabels[index], 2] += 1

    #Loop through all top 5 correct predictions incrementing the label top 5 correct count by 1 in car stats
    for index in top5CorrectIndices:
        carStats[batchLabels[index], 3] += 1
    return carStats


#Name:          printClassStats
#Purpose:       print the most / least accurate classes as well as the most / least accurate top 5 classes
#Inputs:        carStats -> numpy array of stats per label type [label, labelName, accuracyTop1, accuracyTop5, total]
#               printCount -> number of top / least classes to print
#Output:        none -> just prints results
def printClassStats(carStats, printCount):
    #Sort by ascending accuracy in terms of order
    topPredictedAccuracy = carStats[carStats[:,2].argsort()][:,1:3]
    top5PredictedAccuracy = carStats[carStats[:,3].argsort()][:,[1,3]]

    print("Network least accurate classes are: ")
    print(topPredictedAccuracy[0:printCount])
    print()

    print("Network most accurate classes are: ")
    print(topPredictedAccuracy[len(topPredictedAccuracy)-printCount:])
    print()

    print("Network least accurate top 5 prediction classes are: ")
    print(top5PredictedAccuracy[0:printCount])
    print()

    print("Network most accurate top 5 prediction classes are: ")
    print(top5PredictedAccuracy[len(topPredictedAccuracy)-printCount:])
    print()
    print()
    print()
    return


#Name:          checkResizeShape
#Purpose:       checks that the image resize shape is either [224x224] or [448x448] as these are the only shapes applicable to the BCNN
#Input:         resizeShape -> image resize shape
#Output:        none -> just checks image resize shape
def checkResizeShape(resizeShape):
    if((resizeShape != 224) and (resizeShape != 448)):
        print("Error! Resize shape must be either 224 or 448 corresponding to [224x224] or [448x448]")
        exit()
    return
