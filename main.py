#Libraries used
import numpy as np
import scipy.io as sio
import os
import cv2
import matplotlib.pyplot as plt
import math
import torch
import datetime
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn


class CarDataset(Dataset):
    """Stanford Cars Dataset."""
    def __init__(self, root_dir, resizeShape):
        self.root_dir = root_dir
        self.resizeShape = resizeShape

    def __len__(self):
        counter = 0
        for file in os.listdir(self.root_dir):
            counter += 1
        return counter

    def __getitem__(self, idx):
        imageName = os.listdir(self.root_dir)
        image = cv2.imread(self.root_dir + "/" + imageName[idx])[:,:,::-1]
        image = cv2.resize(image,(self.resizeShape[1],self.resizeShape[0]),interpolation=cv2.INTER_LINEAR)
        return image, imageName[idx]


def getLabels(path):
    annos = sio.loadmat(path)
    _, total_size = annos["annotations"].shape
    labels = np.ndarray(shape=(total_size, 2), dtype=object)
    for i in range(total_size):
        fname = annos["annotations"][0][i][5][0]
        classLabel = annos["annotations"][0][i][4][0][0]
        labels[i,0] = fname
        labels[i,1] = classLabel
    return labels


def trainModel(trainDataset, testDataset, maxEpochs, batchSize1, batchSize2, resizeShape, trainLabels, testLabels, NN):
    trainDatasetLength = len(trainDataset)
    testDatasetLength = len(testDataset)
    batchesPerEpoch1 = math.floor(trainDatasetLength / batchSize1)
    batchesPerEpoch2 = math.floor(trainDatasetLength / batchSize2)

    #NN Parameters
    criterion = nn.CrossEntropyLoss()

    GPU = None
    #Send our network and criterion to GPU
    if torch.cuda.is_available():
        print("Using GPU")
        GPU = torch.device("cuda")
        NNgpu = NN.to(GPU)
        criterion = criterion.to(GPU)
        optimizer = torch.optim.SGD(NNgpu.parameters(), lr=0.001, momentum=0.9)
        #optimizer = torch.optim.Adam(NNgpu.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-03,weight_decay=0, amsgrad=False)
    else:
        print("Using CPU")
        optimizer = torch.optim.SGD(NN.parameters(), lr=0.001, momentum=0.9)
        #optimizer = torch.optim.Adam(NN.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-03,weight_decay=0, amsgrad=False)

    #Store the average training loss per epoch and accuracy values
    epochAverageLossTrain = np.zeros(shape=(maxEpochs, 2), dtype=float)
    epochAccuracyTrain = np.zeros(shape=(maxEpochs, 2), dtype=float)
    epochAccuracyTrain5 = np.zeros(shape=(maxEpochs, 2), dtype=float)
    epochAverageLossTest = np.zeros(shape=(maxEpochs, 2), dtype=float)
    epochAccuracyTest = np.zeros(shape=(maxEpochs, 2), dtype=float)
    epochAccuracyTest5 = np.zeros(shape=(maxEpochs, 2), dtype=float)
    valAcc = 0

    epochIndex = 0
    while(epochIndex < maxEpochs):
        epochTotalLoss = 0
        trainCorrectPredictions = 0
        trainTop5CorrectPredictions = 0
        for batchNumber, (batchDataValues, batchImageNames) in enumerate(dataloader1):
            print("Batch Number: ", batchNumber, end='\r')
            batchLabels = getImageLabels(batchImageNames, trainLabels)
            batchDataValues = batchDataValues.float()
            batchDataValues = np.swapaxes(batchDataValues, 2, 3)
            batchDataValues = np.swapaxes(batchDataValues, 1, 2)

            #Do heavy computations on the GPU if avaliable
            if torch.cuda.is_available():
                NNgpu.train()
                batchDataValues = batchDataValues.to(GPU)
                batchLabels = batchLabels.to(GPU)
                output = NNgpu(batchDataValues/255)
                loss = criterion(output, batchLabels)
                epochTotalLoss = epochTotalLoss + loss.item()
                trainCorrectPredictions = trainCorrectPredictions + calcCorrectPredictions(output, batchLabels)
                trainTop5CorrectPredictions = trainTop5CorrectPredictions + calcCorrectPredictions5(output, batchLabels)
                # clear gradients for next train
                optimizer.zero_grad()
                #Compute gradients and update weights
                loss.backward()
                optimizer.step()

            #No GPU avaliable
            else:
                NN.train()
                output = NN(batchDataValues/255)
                loss = criterion(output, batchLabels)
                epochTotalLoss = epochTotalLoss + loss.item()
                trainCorrectPredictions = trainCorrectPredictions + calcCorrectPredictions(output, batchLabels)
                trainTop5CorrectPredictions = trainTop5CorrectPredictions + calcCorrectPredictions5(output, batchLabels)
                # clear gradients for next train
                optimizer.zero_grad()
                #Compute gradients and update weights
                loss.backward()
                optimizer.step()

        print("Epoch: ", epochIndex, " - Training Average Loss: ", epochTotalLoss/batchesPerEpoch1)
        epochAverageLossTrain[epochIndex][0] = epochIndex
        epochAverageLossTrain[epochIndex][1] = epochTotalLoss/batchesPerEpoch1
        np.save('trainLoss', epochAverageLossTrain)

        print("Epoch: ", epochIndex, " - Training Accuracy: ", trainCorrectPredictions/trainDatasetLength)
        epochAccuracyTrain[epochIndex][0] = epochIndex
        epochAccuracyTrain[epochIndex][1] = trainCorrectPredictions/trainDatasetLength
        np.save('trainAccuracy', epochAccuracyTrain)

        print("Epoch: ", epochIndex, " - Training Accuracy Top 5: ", trainTop5CorrectPredictions/trainDatasetLength)
        epochAccuracyTrain5[epochIndex][0] = epochIndex
        epochAccuracyTrain5[epochIndex][1] = trainTop5CorrectPredictions/trainDatasetLength
        np.save('trainAccuracyTop5', epochAccuracyTrain5)

        #Test model on validation set
        epochTotalLoss = 0
        valCorrectPredictions = 0
        valTop5CorrectPredictions = 0
        for batchNumber, (batchDataValues, batchImageNames) in enumerate(dataloader2):
            print("Batch Number: ", batchNumber, end='\r')
            batchLabels = getImageLabels(batchImageNames, testLabels)
            batchDataValues = batchDataValues.float()
            batchDataValues = np.swapaxes(batchDataValues, 2, 3)
            batchDataValues = np.swapaxes(batchDataValues, 1, 2)

            #Do heavy computations on the GPU if avaliable
            if torch.cuda.is_available():
                NNgpu.eval()
                batchDataValues = batchDataValues.to(GPU)
                batchLabels = batchLabels.to(GPU)
                output = NNgpu(batchDataValues/255)
                loss = criterion(output, batchLabels)
                epochTotalLoss = epochTotalLoss + loss.item()
                valCorrectPredictions = valCorrectPredictions + calcCorrectPredictions(output, batchLabels)
                valTop5CorrectPredictions = valTop5CorrectPredictions + calcCorrectPredictions5(output, batchLabels)

            #No GPU avaliable
            else:
                NN.eval()
                output = NN(batchDataValues/255)
                loss = criterion(output, batchLabels)
                epochTotalLoss = epochTotalLoss + loss.item()
                valCorrectPredictions = valCorrectPredictions + calcCorrectPredictions(output, batchLabels)
                valTop5CorrectPredictions = valTop5CorrectPredictions + calcCorrectPredictions5(output, batchLabels)

        print("Epoch: ", epochIndex, " - Validation Average Loss: ", epochTotalLoss/batchesPerEpoch2)
        epochAverageLossTest[epochIndex][0] = epochIndex
        epochAverageLossTest[epochIndex][1] = epochTotalLoss/batchesPerEpoch2
        np.save('testLoss', epochAverageLossTest)

        print("Epoch: ", epochIndex, " - Validation Accuracy: ", valCorrectPredictions/testDatasetLength)
        epochAccuracyTest[epochIndex][0] = epochIndex
        epochAccuracyTest[epochIndex][1] = valCorrectPredictions/testDatasetLength
        np.save('testAccuracy', epochAccuracyTest)

        print("Epoch: ", epochIndex, " - Validation Accuracy Top 5: ", valTop5CorrectPredictions/testDatasetLength)
        epochAccuracyTest5[epochIndex][0] = epochIndex
        epochAccuracyTest5[epochIndex][1] = valTop5CorrectPredictions/testDatasetLength
        np.save('testAccuracyTop5', epochAccuracyTest5)
        print()

        #Save the model only if validation set accuracy has improved
        if(valCorrectPredictions/testDatasetLength > valAcc):
            torch.save(NN.state_dict(), "NN")
            valAcc = valCorrectPredictions/testDatasetLength
        epochIndex += 1
    return


def calcCorrectPredictions(outputs, labels):
    correct = 0
    predictedLabels = torch.argmax(outputs, dim=1)
    index = 0
    while(index < len(labels)):
        if(predictedLabels[index] == labels[index]):
            correct += 1
        index += 1
    return correct


def calcCorrectPredictions5(outputs, labels):
    correct = 0
    predictedLabels = torch.topk(outputs, k=5, dim=1)
    index = 0
    while(index < len(labels)):
        index2 = 0
        while(index2 < len(predictedLabels[1][0])):
            if(predictedLabels[1][index][index2] == labels[index]):
                correct += 1
            index2 += 1
        index += 1
    return correct


def getImageLabels(names, trainLabels):
    batchLabelsNumpy = np.zeros(len(names), dtype=int)
    idx = 0
    while(idx < len(names)):
        batchLabelsNumpy[idx] = trainLabels.get(names[idx])
        idx += 1
    return torch.from_numpy(batchLabelsNumpy)


def buildLabelDictionary(trainLabels):
    dictionary = {}
    index = 0
    while(index < len(trainLabels)):
        dictionary[trainLabels[index][0]] = trainLabels[index][1]
        index += 1
    return dictionary


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



#Main function
if __name__ == '__main__':
    trainLabels = getLabels("carDevkit/devkit/cars_train_annos.mat")
    trainLabels[:,1] = trainLabels[:,1] - 1
    trainLabels = buildLabelDictionary(trainLabels)
    testLabels = getLabels("carDevkit/devkit/cars_test_annos_withlabels.mat")
    testLabels[:,1] = testLabels[:,1] - 1
    testLabels = buildLabelDictionary(testLabels)

    maxEpochs = 5
    batchSize1 = 32
    batchSize2 = 8
    resizeShape = [224, 224]
    # Flag for feature extracting. False -> finetune whole model, True -> update reshaped layer params
    feature_extract = False

    dataTransforms = {transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

    trainCars = CarDataset("carsTrain", resizeShape)
    dataloader1 = DataLoader(dataset=trainCars, batch_size=batchSize1, shuffle=True, num_workers=4)

    testCars = CarDataset("carsTest", resizeShape)
    dataloader2 = DataLoader(dataset=testCars, batch_size=batchSize2, shuffle=True, num_workers=4)

    """#Load in VGG11
    NN = models.vgg11(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.classifier[6] = nn.Linear(4096, 196)"""

    """#Load in VGG11_BN
    NN = models.vgg11_bn(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.classifier[6] = nn.Linear(4096, 196)"""

    """#Load in VGG16
    NN = models.vgg16(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.classifier[6] = nn.Linear(4096, 196)"""

    """#Load in VGG16_BN
    NN = models.vgg16_bn(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.classifier[6] = nn.Linear(4096, 196)"""

    """#Load in Resnet18
    NN = models.resnet18(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.fc = nn.Linear(512, 196)"""

    """#Load in Resnet50
    NN = models.resnet50(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.fc = nn.Linear(2048, 196)"""

    """#Load in Resnet152
    NN = models.resnet152(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.fc = nn.Linear(2048, 196)"""

    """#Load in DenseNet161
    NN = models.densenet161(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #Reshape last layer as 196 classes and not 1000
    NN.fc = nn.Linear(1024, 196)"""

    #Load AlexNet and reshape last layer as only 196 classes
    NN = models.alexnet(pretrained=True)
    #Used to determine fine-tuning vs feature extraction only
    set_parameter_requires_grad(NN, feature_extract)
    #NN = models.alexnet(pretrained=False)
    NN.classifier[6] = nn.Linear(4096, 196)

    trainModel(trainCars, testCars, maxEpochs, batchSize1, batchSize2, resizeShape, trainLabels, testLabels, NN)
