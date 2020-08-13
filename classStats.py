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

def getLabelNames():
    annos = sio.loadmat('carDevkit/devkit/cars_meta.mat')
    _, total_size = annos["class_names"].shape
    labelNames = np.ndarray(shape=(total_size, 2), dtype=object)
    for i in range(total_size):
        labelName = annos["class_names"][0][i][0]
        labelNames[i,0] = i+1
        labelNames[i,1] = labelName
    return labelNames


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


def testModel(testDataset, resizeShape, testLabels, NN, carStats):
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

    #Test model on test set
    for batchNumber, (batchDataValues, batchImageNames) in enumerate(dataloader2):
        print("Batch Number: ", batchNumber, end='\r')
        batchLabels = getImageLabels(batchImageNames, testLabels)
        batchDataValues = batchDataValues.float()
        batchDataValues = np.swapaxes(batchDataValues, 2, 3)
        batchDataValues = np.swapaxes(batchDataValues, 1, 2)

        carStats[batchLabels.item(), 4] = carStats[batchLabels.item(), 4] + 1

        #Do heavy computations on the GPU if avaliable
        if torch.cuda.is_available():
            NNgpu.eval()
            batchDataValues = batchDataValues.to(GPU)
            batchLabels = batchLabels.to(GPU)
            output = NNgpu(batchDataValues/255)
            loss = criterion(output, batchLabels)
            topCorrect = calcCorrectPredictions(output, batchLabels)
            top5Correct = calcCorrectPredictions5(output, batchLabels)
            carStats[batchLabels.item(), 2] = carStats[batchLabels.item(), 2] + topCorrect
            carStats[batchLabels.item(), 3] = carStats[batchLabels.item(), 3] + top5Correct

        #No GPU avaliable
        else:
            NN.eval()
            output = NN(batchDataValues/255)
            loss = criterion(output, batchLabels)
            topCorrect = calcCorrectPredictions(output, batchLabels)
            top5Correct = calcCorrectPredictions5(output, batchLabels)
            carStats[batchLabels.item(), 2] = carStats[batchLabels.item(), 2] + topCorrect
            carStats[batchLabels.item(), 3] = carStats[batchLabels.item(), 3] + top5Correct
    return carStats


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


def printClassStats(topPredictedAccuracy, top5PredictedAccuracy, printCount):
    length = len(topPredictedAccuracy)
    print("Network least accurate classes are: ")
    print(topPredictedAccuracy[0:printCount])
    print()

    print("Network most accurate classes are: ")
    print(topPredictedAccuracy[length-printCount:])
    print()

    print("Network least accurate top 5 prediction classes are: ")
    print(top5PredictedAccuracy[0:printCount])
    print()

    print("Network most accurate top 5 prediction classes are: ")
    print(top5PredictedAccuracy[length-printCount:])
    print()
    return



#Main function
if __name__ == '__main__':
    labelNames = getLabelNames()
    labelNames[:,0] = labelNames[:,0] - 1
    #carStats data structure = [label, labelName, correctTop1, correctTop5, total]
    carStats = np.ndarray(shape=(len(labelNames), 5), dtype=object)
    carStats[:,0:2] = labelNames
    carStats[:,2:] = 0.0

    testLabels = getLabels("carDevkit/devkit/cars_test_annos_withlabels.mat")
    testLabels[:,1] = testLabels[:,1] - 1
    testLabels = buildLabelDictionary(testLabels)

    resizeShape = [224, 224]

    dataTransforms = {transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

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

    #For AlexNet
    NN = models.alexnet()
    NN.classifier[6] = nn.Linear(4096, 196)
    NN.load_state_dict(torch.load("NN"))

    carStats = testModel(testCars, resizeShape, testLabels, NN, carStats)

    #carStats data structure now = [label, labelName, accuracyTop1, accuracyTop5, total]
    carStats[:,2] = carStats[:,2] / carStats[:,4]
    carStats[:,3] = carStats[:,3] / carStats[:,4]

    #Sort by ascending accuracy in terms of order
    topPredictedAccuracy = carStats[carStats[:,2].argsort()][:,1:3]
    top5PredictedAccuracy = carStats[carStats[:,3].argsort()][:,[1,3]]

    printCount=10
    printClassStats(topPredictedAccuracy, top5PredictedAccuracy, printCount)
