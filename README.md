# Fine-Grained Car Make and Model Classification with Transfer Learning and BCNNs
In this project we explore fine-grained car make and model classification on the Stanford Cars Dataset. We first experiment with fine-tuning some of the more famous CNN architectures such as VGG, Resnet, and Densenet. After doing this analysis we build various structured ensembles of these fine-tuned models and analyze how they are able to support each other during classification. Finally we explore the concept of bilinear convolutional neural networks (BCNNs) which take into consideration not only spatial locality within images but also feature location.

This project was created as the final for [10-707 Topics in Deep Learning](https://deeplearning-cmu-10707.github.io/) at [Carnegie Mellon University](https://www.cmu.edu/) under the instruction of [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/) in collaboration with Hashim Saeed.  

## Dataset Download
Due to space constraints the dataset has not been included in this repository as it is ~2GB in size. The Stanford Cars Dataset can be downloaded at the following [link](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). To download this data please scroll to the "Download" section and click on the following series of links:
* Link at the end of "Training images can be downloaded here" (Line 1)
* Link at the end of "Testing images can be downloaded here" (Line 2)
* Link at the end of "A devkit, including class labels for training images..." (Line 3)

The three links above should download the following files:
* cars_train.tgz
* cars_test.tgz
* car_devkit.tgz

After downloading these files please create a 'data' folder within this repository and move the three files above to it. Extract the folder contained within each compressed download. The code is setup to work over the data in this way. The result of performing the actions above should result in the following being contained in this repo:
* data/car_devkit
* data/cars_test
* data/cars_train

For testing purposes download the test annotations file with labels. This can be downloaded by selecting the final link in the "Update" subsection underneath "Download" where it reads "you can use test annotations here" (Line 3). After downloading this file (named "cars_test_annos_withlabels.mat") please move it to the following folder:
* data/car_devkit/devkit

The Stanford Cars Dataset should now be ready for processing with the code contained in this repository.


## Virtual environment creation:
Assuming [Anaconda](https://www.anaconda.com/) is installed the following steps can be taken to create a conda virtual environment for this project. Note that the local system used for development had access to a GeForce GTX 1060 GPU with CUDA version 10.2, thus the PyTorch install command may vary based on CUDA version. Please see [PyTorch installation](https://pytorch.org/) for more details.
```
conda create -n car_classification python=3.7
conda activate car_classification
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install scipy
conda install opencv
conda install matplotlib
```

## Repo breakdown
This repository consists of 9 files and 1 folder. The 9 files are as follows:
* bcnnMain.py:
* bcnnModel.py:
* dataset.py: PyTorch dataset class definition
* display_training_stats.py: Displays the stats from training in a graph
* label_names.py: Prints all car labels in the dataset
* main_cnn.py: Trains a CNN over the dataset for a set number of epochs evaluating performance over the test set after each
* test_cnn.py: Evaluates the final performance of a trained CNN over both the training and test sets
* test_ensemble.py: Evaluates the performance of an ensemble of three trained CNNs over both the training and test sets
* utils.py: Utility functions called by numerous files
* view_image.py: Views a random image from the training dataset and its corresponding label

The 1 folder is:
* report: contains the paper written covering the project, "Fine Grained Car Make and Model Classification with Transfer Learning and BCNNs"  

## Viewing all labels in the dataset
To view all dataset labels please run the following command:
```
python label_names.py
```

## Viewing a random image from the training dataset
To view a random image from the training dataset run the following command:
```
python view_image.py
```
