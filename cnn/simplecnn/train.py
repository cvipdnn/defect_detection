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

    x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(img_input, x)
    return model


def imageGenerator(images, y, batch_size=32):
    image_num = images.shape[0]
    print(image_num)
    c = 0
    index_shuf = list(range(image_num))
    while (True):
        img = np.zeros((batch_size, image_size, image_size))
        y_label = np.zeros((batch_size, 2))

        for i in range(c, c + batch_size):
            index = index_shuf[i]
            imageObj = images[index]

            ### Extract Image ###
            cur_img = plt.imread(imageObj)

            img[i - c] = cur_img

            y_label [ i -c ] = y[index]

        c = c + batch_size
        if (c + batch_size >= image_num):
            c = 0
            random.shuffle(index_shuf)

        yield img, y_label


dataDir='/home/cvip/deep_learning/datasets/DAGM_KaggleUpload/'



# read train image and labels
dirList = os.listdir(dataDir)

images = []
class_image = []

for dir in dirList:
    curDirTrain = dataDir + dir + '/Train/'

    curDirTrain_Label = dataDir + dir + '/Train/Label'

    curDirTrain_LabelFile = dataDir + dir + '/Train/Label/Labels.txt'

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
#model.load_weights('./model/pretrained.h5')
model.summary()


x_train, x_valid, y_train, y_valid = train_test_split(images, class_image, test_size=0.2, random_state = 32)

batchsize = 16
train_gen = imageGenerator(x_train, y_train, batch_size = batchsize)
val_gen = imageGenerator(x_valid, y_valid, batch_size = batchsize)


train_image_num = x_train.shape[0]

val_image_num = x_valid.shape[0]

epochs = 256
learning_rate = 0.0001
decay_rate = learning_rate/ epochs
opt = Adam(lr= learning_rate, decay = decay_rate)


steps_per_epoch = train_image_num // batchsize


validation_steps = val_image_num // batchsize


model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics= ['accuracy'])


mc = ModelCheckpoint('wt{epoch:05d}.h5', save_weights_only=True, save_freq=steps_per_epoch)


history = model.fit(x=train_gen,
                validation_data=val_gen,
                steps_per_epoch=steps_per_epoch,
                validation_steps = validation_steps,
                callbacks = [mc],
                class_weight = {0: 1.,1: 6.6},
                epochs = epochs,
                verbose = True)


