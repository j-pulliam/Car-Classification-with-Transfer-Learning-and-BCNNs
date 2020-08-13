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


def testEnsemble(trainDataset, testDataset, resizeShape, trainLabels, testLabels, model1, model2, model3):
    trainDatasetLength = len(trainDataset)
    testDatasetLength = len(testDataset)

    GPU = None
    #Send our network and criterion to GPU
    if torch.cuda.is_available():
        print("Using GPU")
        GPU = torch.device("cuda")
        model1gpu = model1.to(GPU)
        model2gpu = model2.to(GPU)
        model3gpu = model3.to(GPU)
    else:
        print("Using CPU")

    #Test model on training set
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
            model1gpu.eval()
            model2gpu.eval()
            model3gpu.eval()
            batchDataValues = batchDataValues.to(GPU)
            batchLabels = batchLabels.to(GPU)
            output1 = model1gpu(batchDataValues/255)
            output2 = model2gpu(batchDataValues/255)
            output3 = model3gpu(batchDataValues/255)
            trainCorrectPredictions = trainCorrectPredictions + calcCorrectPredictions(output1, output2, output3, batchLabels)
            trainTop5CorrectPredictions = trainTop5CorrectPredictions + calcCorrectPredictions5(output1, output2, output3, batchLabels)

        #No GPU avaliable
        else:
            model1gpu.eval()
            model2gpu.eval()
            model3gpu.eval()
            output1 = model1gpu(batchDataValues/255)
            output2 = model2gpu(batchDataValues/255)
            output3 = model3gpu(batchDataValues/255)
            trainCorrectPredictions = trainCorrectPredictions + calcCorrectPredictions(output1, output2, output3, batchLabels)
            trainTop5CorrectPredictions = trainTop5CorrectPredictions + calcCorrectPredictions5(output1, output2, output3, batchLabels)

    print("Training Accuracy: ", trainCorrectPredictions/trainDatasetLength)
    print("Training Accuracy Top 5: ", trainTop5CorrectPredictions/trainDatasetLength)

    #Test model on test set
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
            model1gpu.eval()
            model2gpu.eval()
            model3gpu.eval()
            batchDataValues = batchDataValues.to(GPU)
            batchLabels = batchLabels.to(GPU)
            output1 = model1gpu(batchDataValues/255)
            output2 = model2gpu(batchDataValues/255)
            output3 = model3gpu(batchDataValues/255)
            valCorrectPredictions = valCorrectPredictions + calcCorrectPredictions(output1, output2, output3, batchLabels)
            valTop5CorrectPredictions = valTop5CorrectPredictions + calcCorrectPredictions5(output1, output2, output3, batchLabels)

        #No GPU avaliable
        else:
            model1gpu.eval()
            model2gpu.eval()
            model3gpu.eval()
            output1 = model1gpu(batchDataValues/255)
            output2 = model2gpu(batchDataValues/255)
            output3 = model3gpu(batchDataValues/255)
            valCorrectPredictions = valCorrectPredictions + calcCorrectPredictions(output1, output2, output3, batchLabels)
            valTop5CorrectPredictions = valTop5CorrectPredictions + calcCorrectPredictions5(output1, output2, output3, batchLabels)

    print("Test Accuracy: ", valCorrectPredictions/testDatasetLength)
    print("Test Accuracy Top 5: ", valTop5CorrectPredictions/testDatasetLength)
    return


def calcCorrectPredictions(output1, output2, output3, labels):
    outputs = output1 + output2 + output3
    correct = 0
    predictedLabels = torch.argmax(outputs, dim=1)
    index = 0
    while(index < len(labels)):
        if(predictedLabels[index] == labels[index]):
            correct += 1
        index += 1
    return correct


def calcCorrectPredictions5(output1, output2, output3, labels):
    outputs = output1 + output2 + output3
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

    dataTransforms = {transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

    trainCars = CarDataset("carsTrain", resizeShape)
    dataloader1 = DataLoader(dataset=trainCars, batch_size=1, shuffle=False, num_workers=4)

    testCars = CarDataset("carsTest", resizeShape)
    dataloader2 = DataLoader(dataset=testCars, batch_size=1, shuffle=False, num_workers=4)

    #For VGG11
    #Vgg11 = models.vgg11()
    #Vgg11.classifier[6] = nn.Linear(4096, 196)
    #Vgg11.load_state_dict(torch.load("VGG11"))

    #For VGG11_BN
    Vgg11_BN = models.vgg11_bn()
    Vgg11_BN.classifier[6] = nn.Linear(4096, 196)
    Vgg11_BN.load_state_dict(torch.load("VGG11_BN"))

    #For VGG16
    #Vgg16 = models.vgg16()
    #Vgg16.classifier[6] = nn.Linear(4096, 196)
    #Vgg16.load_state_dict(torch.load("VGG16"))

    #For VGG16_BN
    Vgg16_BN = models.vgg16_bn()
    Vgg16_BN.classifier[6] = nn.Linear(4096, 196)
    Vgg16_BN.load_state_dict(torch.load("VGG16_BN"))

    #For Resnet18
    #Resnet18 = models.resnet18()
    #Resnet18.fc = nn.Linear(512, 196)
    #Resnet18.load_state_dict(torch.load("Resnet18"))

    #For Resnet50
    #Resnet50 = models.resnet50()
    #Resnet50.fc = nn.Linear(2048, 196)
    #Resnet50.load_state_dict(torch.load("Resnet50"))

    #For Resnet152 (1)
    #Resnet152 = models.resnet152()
    #Resnet152.fc = nn.Linear(2048, 196)
    #Resnet152.load_state_dict(torch.load("Resnet152"))

    #For Resnet152 (2)
    #Resnet152_2 = models.resnet152()
    #Resnet152_2.fc = nn.Linear(2048, 196)
    #Resnet152_2.load_state_dict(torch.load("Resnet152_2"))

    #For Densenet161
    #Densenet161 = models.densenet161()
    #Densenet161.fc = nn.Linear(1024, 196)
    #Densenet161.load_state_dict(torch.load("DenseNet161"))

    #For AlexNet
    AlexNet = models.alexnet()
    AlexNet.classifier[6] = nn.Linear(4096, 196)
    AlexNet.load_state_dict(torch.load("AlexNet"))

    testEnsemble(trainCars, testCars, resizeShape, trainLabels, testLabels, Vgg11_BN, Vgg16_BN, AlexNet)
