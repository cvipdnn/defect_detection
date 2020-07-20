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
n_class = 2

def SimpleFCN(input_height, input_width):
    img_input = Input(shape=(input_height, input_width, 1))

    x = Conv2D(48, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool1', data_format=IMAGE_ORDERING)(x)

    f0 = x;
    x = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv31',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv32',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv33',
               data_format=IMAGE_ORDERING)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)
    f1 = x;

    x = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv34',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv35',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv36',
               data_format=IMAGE_ORDERING)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)

    f3 = x
    x = Conv2D(n_class, (3, 3), kernel_initializer='he_normal', activation='linear', padding='same', name='fc2',
               data_format=IMAGE_ORDERING)(x)
    x = UpSampling2D(size=(2, 2))(x)

    f1 = Conv2D(n_class, (3, 3), kernel_initializer='he_normal', activation='linear', padding='same', name='fc6',
                data_format=IMAGE_ORDERING)(f1)

    x = Add()([f1, x]);
    x = UpSampling2D(size=(2, 2))(x);

    f1 = Conv2D(n_class, (3, 3), kernel_initializer='he_normal', activation='linear', padding='same', name='fc7',
                data_format=IMAGE_ORDERING)(f0)

    x = Add()([f1, x]);

    x = UpSampling2D(size=(4, 4))(x)
    x = Activation('softmax')(x);

    model = Model(img_input, x)
    return model


dataDir='/home/cvip/deep_learning/datasets/DAGM_KaggleUpload/'



# read train image and labels
dirList = os.listdir(dataDir)

images = []
mask_images = []
for dir in dirList:
    curDirTest = dataDir + dir + '/Test/'

    curDirTest_Label = dataDir + dir + '/Test/Label'

    curDirTest_LabelFile = dataDir + dir + '/Test/Label/Labels.txt'

    with open(curDirTest_LabelFile) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 1:
                continue
            images.append(curDirTest+row[2])
            if (row[4]!='0'):
                mask_images.append(curDirTest_Label+'/'+row[4])
            else:
                # no defect
                mask_images.append(row[4])

images = np.array(images)
mask_images = np.array(mask_images)

model = SimpleFCN(image_size, image_size)
model.load_weights('./model/pretrained.h5')
model.summary()

num_images = images.shape[0]

n_good = 0
n_defect = 0
res_good = 0
res_defect = 0

for index in range(num_images):
    # inference only
    cur_img = plt.imread(images[index])

    x = np.reshape(cur_img, (1, image_size, image_size))
    label = model.predict(x)
    label = label[0, :, :, 0] <= label[0, :, :, 1]
    n_pixel_defect = np.sum(label)
    if mask_images[index] != '0':
       n_defect = n_defect + 1
       if n_pixel_defect > 4096:
           res_defect = res_defect + 1

    else:
        n_good = n_good + 1
        if n_pixel_defect <= 4096:
            res_good = res_good + 1

print(res_defect/n_defect, res_good/n_good, n_defect, n_good)
c = 0
plt.subplots(tight_layout=True)
# display the first 8 images with defect
for index in range(num_images):
    # inference only
    cur_img = plt.imread(images[index])

    if mask_images[index] != '0':
        x=np.reshape(cur_img, (1,image_size,image_size))
        label = model.predict(x)
        plt.subplot(8,6,c+1)
        plt.imshow(cur_img)
        plt.axis('off')
        plt.gray()
        plt.subplot(8,6, c+2)
        plt.imshow(plt.imread(mask_images[index]))
        plt.axis('off')
        plt.gray()
        plt.subplot(8,6,c+3)
        plt.imshow(label[0,:,:,1])
        plt.axis('off')
        plt.gray()
        c= c+ 3
    if c  == 48:
        plt.show()
        break
# display the first 8 images without defect
c = 0
plt.subplots(tight_layout=True)
for index in range(num_images):
    # inference only
    cur_img = plt.imread(images[index])

    if mask_images[index] == '0':
        x=np.reshape(cur_img, (1,image_size,image_size))
        label = model.predict(x)
        plt.subplot(8,6,c+1)
        plt.imshow(cur_img)
        plt.axis('off')
        plt.gray()
        plt.subplot(8,6,c+2)
        plt.imshow(label[0,:,:,1])
        plt.axis('off')
        plt.gray()
        c= c+ 2
    if c  == 48:
        plt.show()
        break