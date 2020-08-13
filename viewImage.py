#Libraries used
import numpy as np
import scipy.io as sio
import os
import cv2
import matplotlib.pyplot as plt


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


def viewImage(path, idx, labels, resizeShape):
    image_names = os.listdir(path)
    im = cv2.imread(path + "/" + image_names[idx])[:,:,::-1]
    print("image is", image_names[idx])
    w, h, ch = im.shape
    print("orignal shape:" , w, h)
    im = cv2.resize(im,(resizeShape[1],resizeShape[0]),interpolation=cv2.INTER_LINEAR)
    w, h, ch = im.shape
    print("resized shape:" , w, h)
    index = np.where(labels == image_names[idx])
    index = index[0]
    print("the label is ", labels[index][0][1])
    plt.imshow(im)
    plt.show()



#Main function
if __name__ == '__main__':
    trainLabels = getLabels("carDevkit/devkit/cars_train_annos.mat")
    trainLabels[:,1] = trainLabels[:,1] - 1

    resizeShape = [224, 224]
    imageNumber = 2216

    viewImage("carsTrain", imageNumber, trainLabels, resizeShape)
