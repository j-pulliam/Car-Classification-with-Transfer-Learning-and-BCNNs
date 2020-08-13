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

from bcnnModel import BCNN448, BCNN224, BCNN


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


def testModel(trainDataset, testDataset, resizeShape, trainLabels, testLabels, NN):
    trainDatasetLength = len(trainDataset)
    testDatasetLength = len(testDataset)

    #NN Parameters
    criterion = nn.CrossEntropyLoss()

    GPU = None
    #Send our network and criterion to GPU
    if torch.cuda.is_available():
        print("Using GPU")
        GPU = torch.device("cuda")
        NNgpu = NN.to(GPU)
        criterion = criterion.to(GPU)
    else:
        print("Using CPU")

    #Test model on training set
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
            NNgpu.eval()
            batchDataValues = batchDataValues.to(GPU)
            batchLabels = batchLabels.to(GPU)
            output = NNgpu(batchDataValues/255)
            loss = criterion(output, batchLabels)
            epochTotalLoss = epochTotalLoss + loss.item()
            trainCorrectPredictions = trainCorrectPredictions + calcCorrectPredictions(output, batchLabels)
            trainTop5CorrectPredictions = trainTop5CorrectPredictions + calcCorrectPredictions5(output, batchLabels)

        #No GPU avaliable
        else:
            NN.eval()
            output = NN(batchDataValues/255)
            loss = criterion(output, batchLabels)
            epochTotalLoss = epochTotalLoss + loss.item()
            trainCorrectPredictions = trainCorrectPredictions + calcCorrectPredictions(output, batchLabels)
            trainTop5CorrectPredictions = trainTop5CorrectPredictions + calcCorrectPredictions5(output, batchLabels)

    print("Training Average Loss: ", epochTotalLoss/trainDatasetLength)
    print("Training Accuracy: ", trainCorrectPredictions/trainDatasetLength)
    print("Training Accuracy Top 5: ", trainTop5CorrectPredictions/trainDatasetLength)

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

    print("Validation Average Loss: ", epochTotalLoss/testDatasetLength)
    print("Validation Accuracy: ", valCorrectPredictions/testDatasetLength)
    print("Validation Accuracy Top 5: ", valTop5CorrectPredictions/testDatasetLength)
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



#Main function
if __name__ == '__main__':
    trainLabels = getLabels("carDevkit/devkit/cars_train_annos.mat")
    trainLabels[:,1] = trainLabels[:,1] - 1
    trainLabels = buildLabelDictionary(trainLabels)
    testLabels = getLabels("carDevkit/devkit/cars_test_annos_withlabels.mat")
    testLabels[:,1] = testLabels[:,1] - 1
    testLabels = buildLabelDictionary(testLabels)

    resizeShape = [224, 224]
    #resizeShape = [448, 448]

    dataTransforms = {transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

    trainCars = CarDataset("carsTrain", resizeShape)
    dataloader1 = DataLoader(dataset=trainCars, batch_size=1, shuffle=False, num_workers=4)

    testCars = CarDataset("carsTest", resizeShape)
    dataloader2 = DataLoader(dataset=testCars, batch_size=1, shuffle=False, num_workers=4)

    #For VGG11
    #NN = models.vgg11()
    #NN.classifier[6] = nn.Linear(4096, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For VGG11_BN
    #NN = models.vgg11_bn()
    #NN.classifier[6] = nn.Linear(4096, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For VGG16
    #NN = models.vgg16()
    #NN.classifier[6] = nn.Linear(4096, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For VGG16_BN
    #NN = models.vgg16_bn()
    #NN.classifier[6] = nn.Linear(4096, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For Resnet18
    #NN = models.resnet18()
    #NN.fc = nn.Linear(512, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For Resnet50
    #NN = models.resnet50()
    #NN.fc = nn.Linear(2048, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For Resnet152
    #NN = models.resnet152()
    #NN.fc = nn.Linear(2048, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For DenseNet161
    #NN = models.densenet161()
    #NN.fc = nn.Linear(1024, 196)
    #NN.load_state_dict(torch.load("NN"))

    #For BCNN
    #NN = BCNN224()
    #NN = BCNN448()
    #NN = BCNN()
    #NN.load_state_dict(torch.load("BCNN"))

    #For AlexNet
    NN = models.alexnet()
    NN.classifier[6] = nn.Linear(4096, 196)
    NN.load_state_dict(torch.load("NN"))

    testModel(trainCars, testCars, resizeShape, trainLabels, testLabels, NN)
