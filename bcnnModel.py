import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

#Source: https://github.com/HaoMood/bilinear-cnn
class BCNN224(nn.Module):
    """B-CNN for Stanford Cars."""
    def __init__(self):
        nn.Module.__init__(self)
        #CNN1 = models.vgg11()
        #CNN1.classifier[6] = nn.Linear(4096, 196)
        #CNN1.load_state_dict(torch.load("VGG11"))
        CNN1 = models.vgg11_bn()
        CNN1.classifier[6] = nn.Linear(4096, 196)
        CNN1.load_state_dict(torch.load("VGG11_BN"))
        self.features1 = CNN1.features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])  # Remove pool5.

        #CNN2 = models.vgg16()
        #CNN2.classifier[6] = nn.Linear(4096, 196)
        #CNN2.load_state_dict(torch.load("VGG16"))
        CNN2 = models.vgg16_bn()
        CNN2.classifier[6] = nn.Linear(4096, 196)
        CNN2.load_state_dict(torch.load("VGG16_BN"))
        self.features2 = CNN2.features
        self.features2 = nn.Sequential(*list(self.features2.children())[:-1])  # Remove pool5.

        # Linear classifier.
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512**2, 196)
        self.softmax = nn.Softmax(dim=1)

        """for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False"""
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, X):
        N = X.size()[0]
        X1 = self.features1(X)
        X1 = X1.view(N, 512, 14**2)
        X2 = self.features2(X)
        X2 = X2.view(N, 512, 14**2)
        #bmm = batch matrix multiplication
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (14**2) # Bilinear
        X = X.view(N, 512**2)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X)) #Signed Square Root
        X = self.dropout(X)
        X = self.fc(X)
        return X


class BCNN448(nn.Module):
    """B-CNN for Stanford Cars."""
    def __init__(self):
        nn.Module.__init__(self)
        CNN1 = models.vgg11_bn()
        CNN1.classifier[6] = nn.Linear(4096, 196)
        CNN1.load_state_dict(torch.load("VGG11_BN"))
        self.features1 = CNN1.features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])  # Remove pool5.

        CNN2 = models.vgg16_bn()
        CNN2.classifier[6] = nn.Linear(4096, 196)
        CNN2.load_state_dict(torch.load("VGG16_BN"))
        self.features2 = CNN2.features
        self.features2 = nn.Sequential(*list(self.features2.children())[:-1])  # Remove pool5.

        # Linear classifier.
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512**2, 196)
        self.softmax = nn.Softmax(dim=1)

        """for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False"""
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        N = X.size()[0]
        X1 = self.features1(X)
        X1 = X1.view(N, 512, 28**2)
        X2 = self.features2(X)
        X2 = X2.view(N, 512, 28**2)
        #bmm = batch matrix multiplication
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (28**2) # Bilinear
        X = X.view(N, 512**2)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X)) #Signed Square Root
        X = self.dropout(X)
        X = self.fc(X)
        return X


class BCNN(nn.Module):
    """B-CNN for Stanford Cars."""
    def __init__(self):
        nn.Module.__init__(self)
        CNN1 = models.resnet152()
        CNN1.fc = nn.Linear(2048, 196)
        CNN1.load_state_dict(torch.load("Resnet152"))
        del CNN1.fc
        self.features1 = CNN1
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])  # Remove fc

        CNN2 = models.resnet50()
        CNN2.fc = nn.Linear(2048, 196)
        CNN2.load_state_dict(torch.load("Resnet50"))
        del CNN2.fc
        self.features2 = CNN2
        self.features2 = nn.Sequential(*list(self.features2.children())[:-1])  # Remove fc.

        # Linear classifier.
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512**2, 196)
        self.softmax = nn.Softmax(dim=1)

        """for param in self.features1.parameters():
            param.requires_grad = False
        for param in self.features2.parameters():
            param.requires_grad = False"""
        # Initialize the fc layers.
        torch.nn.init.kaiming_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)


    def forward(self, X):
        N = X.size()[0]
        X1 = self.features1(X)
        X1 = X1.view(N, 512, 14**2)
        X2 = self.features2(X)
        X2 = X2.view(N, 512, 14**2)
        #bmm = batch matrix multiplication
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (14**2) # Bilinear
        X = X.view(N, 512**2)
        X = torch.sqrt(F.relu(X)) - torch.sqrt(F.relu(-X)) #Signed Square Root
        X = self.dropout(X)
        X = self.fc(X)
        return X
