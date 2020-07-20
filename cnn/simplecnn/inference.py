
import numpy as np
import tensorflow as tf
import os, sys
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D, UpSampling2D, Add, BatchNormalization
from keras.utils import np_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import scipy.misc
import matplotlib.pyplot as plt
from scipy import misc
from matplotlib import pyplot
import scipy.ndimage
from tensorflow.keras.optimizers import Adam
import random
from keras.utils.np_utils import to_categorical  
from PIL import Image
import cv2 
from sklearn.model_selection import train_test_split
import csv


IMAGE_ORDERING = 'channels_last'

image_size= 512

def SimpleCNN(input_height, input_width):
    img_input = Input(shape=(input_height, input_width, 1))

    x = Conv2D(36, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv1', data_format=IMAGE_ORDERING)(
        img_input)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool1', data_format=IMAGE_ORDERING)(x)



    x = SeparableConv2D(36, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv31',
               data_format=IMAGE_ORDERING)(x)
    x = SeparableConv2D(36, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv32',
               data_format=IMAGE_ORDERING)(x)
    x = SeparableConv2D(36, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv33',
               data_format=IMAGE_ORDERING)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)



    x = SeparableConv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv40',
               data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)



    x = SeparableConv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool4', data_format=IMAGE_ORDERING)(x)


    x = SeparableConv2D(36, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv5', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool5', data_format=IMAGE_ORDERING)(x)

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(img_input, x)
    return model

dataDir='/home/cvip/deep_learning/datasets/DAGM_KaggleUpload/'


dirList = os.listdir(dataDir)

images = []
class_image = []
for dir in dirList:
    curDirTrain = dataDir + dir + '/Test/'

    curDirTrain_Label = dataDir + dir + '/Test/Label'

    curDirTrain_LabelFile = dataDir + dir + '/Test/Label/Labels.txt'

    with open(curDirTrain_LabelFile) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 1:
                continue
            images.append(curDirTrain+row[2])
            if row[1] =='1':
                class_image.append(1)
            elif row[1] =='0':
                class_image.append(0)


class_image = to_categorical(class_image, num_classes=2)


images = np.array(images)
class_image = np.array(class_image)


model = SimpleCNN(image_size, image_size)
model.load_weights('./model/pretrained.h5')
model.summary()



model.save('test.h5')
total_num_images = images.shape[0]
n_good_result = 0
x = np.zeros((1, image_size, image_size))

n_good = 0
n_defect = 0
res_defect = 0
res_good = 0

for index in range(total_num_images):
    cur_img = plt.imread(images[index])

    x[0]= cur_img


    y = model.predict(x)


    if class_image[index,0] == 1:
        n_good += 1
        if y[0, 0] >= y[0, 1]:
            res_good += 1

    if class_image[index,1] == 1:
        n_defect += 1
        if y[0,0] < y[0,1]:
            res_defect += 1

print(res_defect/n_defect, res_good/n_good, n_defect, n_good)
