import numpy as np 
import tensorflow as tf
import os, sys
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, ReLU, Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D, UpSampling2D, Add, BatchNormalization
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
sys.path.insert(0, '../../utils')
# You can add the path into PYTHONPATH
import nn_summary as nn_calc


IMAGE_ORDERING = 'channels_last'

image_size= 512


def imageGenerator(images, y, batch_size=32):
    image_num = images.shape[0]
    #print(image_num)
    c = 0
    index_shuf = list(range(image_num))
    while (True):
        img = np.zeros((batch_size, image_size, image_size,3))
        y_label = np.zeros((batch_size, 2))

        for i in range(c, c + batch_size):
            index = index_shuf[i]
            imageObj = images[index]

            ### Extract Image ###
            cur_img = plt.imread(imageObj)
            cur_img = np.stack((cur_img,) * 3, axis=-1)
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

feature_model = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")

#feature_model.summary()

# try smaller network based on mobilenetv2
featurelayers=['out_relu', 'block_15_depthwise_relu', 'block_13_depthwise_relu', 'block_13_expand_relu', 'block_10_depthwise_relu', 'block_8_depthwise_relu', 'block_6_depthwise_relu', 'block_5_depthwise_relu', 'block_3_depthwise_relu','block_1_depthwise_relu']

x_train, x_valid, y_train, y_valid = train_test_split(images, class_image, test_size=0.2, random_state = 32)
batchsize = 16
train_gen = imageGenerator(x_train, y_train, batch_size=batchsize)
val_gen = imageGenerator(x_valid, y_valid, batch_size=batchsize)


def conv_block_mobilenetv2_stride2(x, n_filter=36):
    f1 = Conv2D(n_filter, (1, 1), strides=(1, 1),  padding='same', data_format=IMAGE_ORDERING)(x)
    f2 = ReLU(max_value=6.0)(f1)
    # apply relu6
    f3 = SeparableConv2D(n_filter, (3, 3), strides=(2,2), padding='same', data_format=IMAGE_ORDERING)(f2)
    f4 = ReLU(max_value=6.0)(f3)

    f5 = Conv2D(n_filter, (1, 1), strides=(1, 1), activation='linear', padding='same', data_format=IMAGE_ORDERING)(f4)
    return f5


for layername in featurelayers:
    layer = feature_model.get_layer(layername).output

    # Create the feature extraction model
    lastlayer = tf.keras.Model(inputs=feature_model.input, outputs=layer)
    # not trained
    #lastlayer.trainable = False

    inputs = tf.keras.layers.Input(shape=[image_size, image_size, 3])
    x = inputs

    feature_layer = lastlayer(x)

    # check the output size, trying to keep the output size is 16x16 , add single block with stride = 2 or maxpooling.
    # https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c
    outputsize = feature_layer.shape[1]

    if outputsize == 32:
        n_filters = feature_layer.shape[3]
        feature_layer = conv_block_mobilenetv2_stride2(feature_layer, n_filters)

    elif outputsize == 64:
        n_filters = feature_layer.shape[3]
        feature_layer = conv_block_mobilenetv2_stride2(feature_layer, n_filters)

        feature_layer = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(feature_layer)

    elif outputsize == 128:
        n_filters = feature_layer.shape[3]

        feature_layer = conv_block_mobilenetv2_stride2(feature_layer, n_filters)
        feature_layer = conv_block_mobilenetv2_stride2(feature_layer, n_filters)

        feature_layer = MaxPooling2D((2, 2), strides=(2, 2), data_format=IMAGE_ORDERING)(feature_layer)
    elif outputsize != 16:
        continue

    final_output = GlobalAveragePooling2D()(feature_layer)
    final_output = Dense(128, activation = 'relu')(final_output)
    final_output = Dense(32, activation = 'relu')(final_output)
    final_output = Dense(2, activation = 'softmax')(final_output)
    model = tf.keras.Model(inputs=inputs, outputs=final_output)
    #model.summary()

    total_calulations = nn_calc.calcutate_mul(lastlayer, showDetails = False) + nn_calc.calcutate_mul(model, showDetails = False)


    train_image_num = x_train.shape[0]

    val_image_num = x_valid.shape[0]

    epochs = 36
    learning_rate = 0.0001
    decay_rate = learning_rate/ epochs
    opt = Adam(lr= learning_rate, decay = decay_rate)


    steps_per_epoch = train_image_num // batchsize


    validation_steps = val_image_num // batchsize


    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics= ['accuracy'])

    mc = ModelCheckpoint('wt{epoch:05d}.h5', save_weights_only=False, save_freq=steps_per_epoch)

    history = model.fit(x=train_gen,
                    validation_data=val_gen,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps = validation_steps,
                    class_weight = {0: 1.,1: 6.6},
                    callbacks=[mc],
                    epochs = epochs,
                    verbose = True)

    print(total_calulations, history.history['accuracy'][epochs-1] , history.history['val_accuracy'][epochs-1], (history.history['accuracy'][epochs-1] + history.history['val_accuracy'][epochs-1])/2)
