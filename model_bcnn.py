#Developer: Dillon Pulliam
#Date: 9/6/2020
#Purpose: The purpose of this file is to give the class definition for a BCNN model


#Libraries needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#Name:          createNetworkBCNN
#Purpose:       initialize the network and reshape the final layer
#Input:         model -> integer corresponding to the type of CNN to build
#Output:        NN -> PyTorch neural network class
def createNetworkBCNN(model):
    #Create a variable to store the NN
    NN = None

    #Initialize the network
    #Load in VGG11
    if(model == 1):
        NN = models.vgg11(pretrained=True)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in VGG11_BN
    elif(model == 2):
        NN = models.vgg11_bn(pretrained=True)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in VGG16
    elif(model == 3):
        NN = models.vgg16(pretrained=True)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in VGG16_BN
    elif(model == 4):
        NN = models.vgg16_bn(pretrained=True)
        NN.classifier[6] = nn.Linear(4096, 196)
    #Load in Resnet50
    elif(model == 5):
        NN = models.resnet50(pretrained=True)
        NN.fc = nn.Linear(2048, 196)
    #Load in Resnet152
    elif(model == 6):
        NN = models.resnet152(pretrained=True)
        NN.fc = nn.Linear(2048, 196)
    else:
        print("Error! Model options are as follows:")
        print("1 -> VGG11")
        print("2 -> VGG11_BN")
        print("3 -> VGG16")
        print("4 -> VGG16_BN")
        print("5 -> Resnet50")
        print("6 -> Resnet152")
        exit()
    return NN


#Name:          getFeatures
#Purpose:       get the portion of the CNN that will be used as features in the BCNN
#Inputs:        CNN -> CNN to extract features from
#               model -> integer corresponding to the type of CNN
#Output:        features -> features to extract from the CNN
def getFeatures(CNN, model):
    #Create a variable to store the features
    features = None

    #For VGG style networks
    if((model >= 1) and (model <= 4)):
        features = CNN.features
        features = nn.Sequential(*list(features.children())[:-1])  # Remove pool5.
    #For Resnet style networks
    elif((model >= 5) and (model <= 6)):
        del CNN.fc
        features = CNN
        features = nn.Sequential(*list(features.children())[:-1])  # Remove fc
    else:
        #Block should never occur
        exit()
    return features


#BCNN model class definition
class BCNN(nn.Module):
    #Name:          __init__
    #Purpose:       initialize and create the BCNN
    #Inputs:        model1 -> integer corresponding to the type of CNN for model #1 of the BCNN
    #               model2 -> integer corresponding to the type of CNN for model #2 of the BCNN
    #               resizeShape -> image resize shape
    #               featureExtract -> whether to perform feature extraction or fine-tune the entire network
    #Output:        none -> just creates class variables and the BCNN network
    def __init__(self, model1, model2, resizeShape, featureExtract):
        nn.Module.__init__(self)

        #Store the resize shape
        self.resize_shape = resizeShape

        #For CNN1 features
        CNN1 = createNetworkBCNN(model1)
        self.features1 = getFeatures(CNN1, model1)

        #For CNN2 features
        CNN2 = createNetworkBCNN(model2)
        self.features2 = getFeatures(CNN2, model2)

        #Linear classifier
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512**2, 196)
        self.softmax = nn.Softmax(dim=1)

        #For feature extraction versus fine-tuning
        if(featureExtract):
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False

        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)
        return


    #Name:          forward
    #Purpose:       forward pass through the BCNN
    #Inputs:        X -> network input
    #Output:        X -> network output
    def forward(self, X):
        #Get the batch size being used
        batchSize = X.size()[0]
        #Pass the image through individual CNNs extarcting features
        X1 = self.features1(X)
        X2 = self.features2(X)
        #Reshape for batch matrix multiplication
        if(self.resize_shape == 224):
            X1 = X1.view(batchSize, 512, 14**2)
            X2 = X2.view(batchSize, 512, 14**2)
            X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (14**2) # Bilinear
        else:
            X1 = X1.view(batchSize, 512, 28**2)
            X2 = X2.view(batchSize, 512, 28**2)
            X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (28**2) # Bilinear
        #Reshape and perform final BCNN computations
        X = X.view(batchSize, 512**2)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X)) #Signed Square Root
        X = self.dropout(X)
        X = self.fc(X)
        return X
