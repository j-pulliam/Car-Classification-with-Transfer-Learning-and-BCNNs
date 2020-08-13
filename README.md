# Fine-Grained_Car_Make_and_Model_Classification_with_Transfer_Learning_and_BCNNs
In this project we explore fine-grained car make and model classification on the Stanford Cars Data Set. We first experiment with fine-tuning some of the more famous CNN architectures such as VGG, Resnet, and Densenet. After doing this analysis we build various structured ensembles of these fine-tuned models and analyze how they are able to support each other during classification. Finally we explore the concept of bilinear convolutional neural networks (BCNNs) which take into consideration not only spatial locality within images but also feature location.

Virtual environment creation steps:
conda create -n car_classification python=3.7
conda activate car_classification 


README

Developer: Dillon Pulliam & Hashim Saeed
Course: 10-707 Topics in Deep Learning
Date: 3/8/2019

Note that for this code to work a carDevkitfolder must contain the following files:
  -A folder named devkit containing:
    1. cars_meta.mat
    2. cars_test_annos.mat
    3. cars_test_annos_withlabels.mat
    4. cars_train_annos.mat
    5. eval_train.m
    6. train_perfect_preds.txt
  -A folder named carsTest containing all jpeg test images
  -A folder named carsTrain containing all jpeg train images

Dataset can downloaded from the following site: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

To view all labels in the dataset run the following command:
  python3 labelNames.py

To view an image from the training set along with its corresponding class label run the following:
  python3 viewImage.py
  -Note to change the image viewed edit line 44 and the value of image number
  -Note that this code was based off of development found at the following site: https://www.kaggle.com/king9528/data-preprocessing-in-python

To run our main code to train a model and get accuracy and loss values run the following command:
  python3 main.py
  -Choose the architecture to train (AlexNet, VGG, Resnet, or DenseNet) by un-commenting the
  block corresponding to the specific model and commenting out all other blocks. These blocks
  are located from lines 263-324. Model choices are as follow:
    VGG11
    VGG11 BN
    VGG16
    VGG16 BN
    Resnet18
    Resnet50
    Resnet152
    Densenet161
    AlexNet
  -This code fine-tunes a pre-trained version of VGG, Resnet, DenseNet, or AlexNet
  -To train a "from-scratch" version set the pretrained variable within the model to "false"
  -To fine-tune the entire model edit the "feature_extract" variable on line 253
    Set "false" to fine-tune the entire model, "true" to train the final fully-connected layer only
  -To edit the max number of epochs, training set batch size, test set batch size, and image
  re-size shape edits lines 248-251
    1. Image re-size shape should be 224x224 unless architectural changes being made
  -Training can be done using either SGD or Adam as the optimizer, adjust lines 63-64 and 67-68
  accordingly
  -Running this command will produce 7 output files:
  1. NN.pickle: saved version of model
  2. trainLoss.npy: Keeps track of the training set loss each epoch
  3. trainAccuracy.npy: Keeps track of the training set accuracy each epoch
  4. trainAccuracyTop5.npy: Keeps track of the training set top 5 accuracy each epoch
  5. testLoss.npy: Keeps track of the test set loss each epoch
  6. testAccuracy.npy: Keeps track of the test set accuracy each epoch
  7. testAccuracyTop5.npy: Keeps track of the test set top 5 accuracy each epoch

To display a graphic of training stats per epoch run the following code:
  python3 displayLoss.py
  -This will create a graph plotting the training and validation set losses and accuracy over each epoch

To get class stats on the test set for a trained model (best / worst classes) run the following:
  python3 classStats.py
  -Note: Adjust lines 191-234 based on the model type being loaded in. Un-comment the specific model
  type trained and comment out all other model types

To test either a trained fine-tuned model or trained BCNN model and get accuracy stats
on both the training and test sets run the following command:
  python3 testModel.py
  -Adjust lines 191-192 based on the re-size shape of images (only use 448x448 for BCNN448)
  -Adjust lines 202-251 based on the model type being tested. Comment out all models except for the
  version being tested
  -If testing a BCNN model adjust lines 243-245 based on the specific architecture and model type
    1. When testing a BCNN the 2 feature extractors (CNNs) that make it up must also be saved in
    the current folder

To test an ensemble of 3 fine-tuned networks run the following command:
  python3 testEnsemble.py
  -Adjust lines 219-272 based on the 3 model architectures being used. 3 sets of architectures
  should be left un-commented while all others are commented out
    1. May also have to adjust model names being loaded in accordingly
    2. If loading 2 / 3 models of the same type may have to add proper source code (see Resnet152)
  -Adjust the model names input to the "testEnsemble" function on line 269 based on the models
  in the ensemble

To run our BCNN main code to train a BCNN model and get accuracy and loss values run the following command:
  python3 bcnnMain.py
  -Choose the architecture to train (BCNN, BCNN224, BCNN) by un-commenting the
  line corresponding to the specific model and commenting out all other lines (259-261)
  -To edit the max number of epochs, training set batch size, test set batch size, and image
  re-size shape edits lines 245-249
    1. Image re-size shape should be 224x224 unless dealing with BCNN448 model
  -Training can be done using either SGD or Adam as the optimizer, adjust lines 64-65 and 68-69
  accordingly
  -When running this code the 2 feature extractors (CNNs) that make up the BCNN must also be saved in
  the current folder
  -To use a BCNN with Resnet model type use the BCNN class
    1. To set feature extractors edit lines 109-111 and 116-118 accordingly in the "bcnnModel.py" file
    2. To fine-tune only the fully-connected layer un-comment lines 128-131; otherwise entire model is updated
  -To use a BCNN with VGG model type use the BCNN224 class
    1. To set feature extractors edit lines 12-17 and 21-26 accordingly in the "bcnnModel.py" file
    2. To fine-tune only the fully-connected layer un-comment lines 35-38; otherwise entire model is updated
  -To use a BCNN with 448x448 sized images use the BCNN448 class
    1. To set feature extractors edit lines 64-66 and 70-72 accordingly in the "bcnnModel.py" file
    2. To fine-tune only the fully-connected layer un-comment lines 81-84; otherwise entire model is updated
  -Running this command will produce 7 output files:
    1. BCNN.pickle: saved version of model
    2. trainLoss.npy: Keeps track of the training set loss each epoch
    3. trainAccuracy.npy: Keeps track of the training set accuracy each epoch
    4. trainAccuracyTop5.npy: Keeps track of the training set top 5 accuracy each epoch
    5. testLoss.npy: Keeps track of the test set loss each epoch
    6. testAccuracy.npy: Keeps track of the test set accuracy each epoch
    7. testAccuracyTop5.npy: Keeps track of the test set top 5 accuracy each epoch
