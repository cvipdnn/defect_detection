
import numpy as np
import tensorflow as tf
import os, sys
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, ReLU, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D, UpSampling2D, Add, BatchNormalization
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
from tensorflow import keras

IMAGE_ORDERING = 'channels_last'

image_size= 512

# https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
def conv_block_mobilenetv2(x, n_filter=36):
    f1 = Conv2D(n_filter, (1, 1), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(x)
    f2 = ReLU(max_value=6.0)(f1)
    # apply relu6
    f3 = SeparableConv2D(n_filter, (3, 3), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(f2)
    f4 = ReLU(max_value=6.0)(f3)

    f5 = Conv2D(n_filter, (1, 1), strides=(1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(f4)

    f6 = Add()([f5,x])
    return f6

def conv_block_mobilenetv2_stride2(x, n_filter=36):
    f1 = Conv2D(n_filter, (1, 1), strides=(1, 1),  padding='same', data_format=IMAGE_ORDERING)(x)
    f2 = ReLU(max_value=6.0)(f1)
    # apply relu6
    f3 = SeparableConv2D(n_filter, (3, 3), strides=(2,2), padding='same', data_format=IMAGE_ORDERING)(f2)
    f4 = ReLU(max_value=6.0)(f3)

    f5 = Conv2D(n_filter, (1, 1), strides=(1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(f4)
    return f5

#https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
#Single Residual block
def conv_block_single_residual(x, n_filter=36):
    f1 = SeparableConv2D(n_filter, (3, 3), strides=(1, 1), activation='relu', padding='same', data_format=IMAGE_ORDERING)(x)

    f2 = SeparableConv2D(n_filter, (3, 3), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(f1)


    f3 = Add()([f2, x])
    f4 = ReLU()(f3)

    return f4

def conv_block_residual_bn(x, n_filter=36):
    f1 = SeparableConv2D(n_filter, (3, 3), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(x)
    f2 = BatchNormalization()(f1)
    f3 = ReLU()(f2)

    f4 = SeparableConv2D(n_filter, (3, 3), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(f3)
    f5 = BatchNormalization()(f4)

    f6 = Add()([f5, x])
    f7 = ReLU()(f6)

    return f7

def generateCNN(input_height, input_width, n_layers):
    img_input = Input(shape=(input_height, input_width, 1))

    x = img_input
    n_downsample = 0
    steps2maxpool =  n_layers//5
    steps2maxpool = max(steps2maxpool, 2)

    # the first layer is always the same: the same as the first layers of mobilnetv2
    x = Conv2D(36, (3, 3), strides=(2, 2), padding='same', data_format=IMAGE_ORDERING)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # my design is to keep the final output is about (16, 16, n)
    for index in range(n_layers):
        if index == n_layers - 1:
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', data_format=IMAGE_ORDERING)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
        else:
            n_filter = 36
            if n_downsample <= 3:
                layer_type = np.random.randint(4)
            else:
                layer_type = np.random.randint(3)

            if layer_type==0:
                x = conv_block_mobilenetv2(x, n_filter)
            if layer_type==3:
                x = conv_block_mobilenetv2_stride2(x, n_filter)
                n_downsample += 1
                #print(n_downsample)
            if layer_type==2:
                x = conv_block_single_residual(x, n_filter)
            if layer_type==1:
                x = conv_block_residual_bn(x, n_filter)
        if index % steps2maxpool == (steps2maxpool-1) and n_downsample <= 3 :
            x = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(x)
            n_downsample += 1
            #print(n_downsample)

    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    y = Dense(2, activation='softmax')(x)

    model = Model(img_input, y)
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


x_train, x_valid, y_train, y_valid = train_test_split(images, class_image, test_size=0.2, random_state = 32)

batchsize = 8
train_gen = imageGenerator(x_train, y_train, batch_size = batchsize)
val_gen = imageGenerator(x_valid, y_valid, batch_size = batchsize)


train_image_num = x_train.shape[0]

val_image_num = x_valid.shape[0]

epochs = 6
learning_rate = 0.0001
decay_rate = learning_rate/ epochs
opt = Adam(lr= learning_rate, decay = decay_rate)


steps_per_epoch = train_image_num // batchsize


validation_steps = val_image_num // batchsize

model = keras.Sequential()

## search the number of layers to get at least 95% accuracy for validation set and training set.
max_layers = 0
for layers in range(6,18):
# each case tries three different networks
    print('trying'  + str(layers) + 'layers')
    for tries in range (3):
        #print(tries)
        model = generateCNN(image_size, image_size, layers)
        #model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps = validation_steps,
                        class_weight = {0: 1.,1: 6.6},
                        epochs = epochs,
                        verbose = True)

        combined_acc = history.history['accuracy'][epochs-1] + history.history['val_accuracy'][epochs-1]
        combined_acc = combined_acc/2
        print(combined_acc)
        if combined_acc > 0.95:
            max_layers = layers
            model.save('best_model.h5')
            break
    if max_layers>0:
        break
# search networks
print( max_layers)

combined_acc = 0
epochs = 32
if max_layers > 0:
    while True:
        model = generateCNN(image_size, image_size, max_layers)

        history = model.fit(x=train_gen,
                        validation_data=val_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps = validation_steps,
                        class_weight = {0: 1.,1: 6.6},
                        epochs = epochs,
                        verbose = False)

        current_score = history.history['accuracy'][epochs - 1] + history.history['val_accuracy'][epochs - 1]

        current_score = current_score / 2

        if combined_acc < current_score:
            model.save('best_model.h5')
            print(combined_acc)
            combined_acc = current_score
else:
    print("search failure!")