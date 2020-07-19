
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


testX = np.array(images)
testY = np.array(class_image)

del images
del class_image


feature_model = tf.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")

feature_model.summary()

layer = feature_model.get_layer('out_relu').output

# Create the feature extraction model
lastlayer = tf.keras.Model(inputs=feature_model.input, outputs=layer)
# not trained
#lastlayer.trainable = False



inputs = tf.keras.layers.Input(shape=[image_size, image_size, 3])
x = inputs

feature_layer = lastlayer(x)

final_output = GlobalAveragePooling2D()(feature_layer)
final_output = Dense(128, activation = 'relu')(final_output)
final_output = Dense(32, activation = 'relu')(final_output)
final_output = Dense(2, activation = 'softmax')(final_output)
model = tf.keras.Model(inputs=inputs, outputs=final_output)
model.load_weights('./model/pretrained.h5')
model.summary()


total_num_images = testX.shape[0]
n_good_result = 0
x = np.zeros((1, image_size, image_size, 3))

for index in range(total_num_images):
    # inference only

    cur_img = plt.imread(testX[index]) * 255

    # normalized to (-1,1)
    cur_img = tf.keras.applications.mobilenet_v2.preprocess_input(cur_img)

    # Add to respective batch sized arrays
    cur_img = np.stack((cur_img,) * 3, axis=-1)

    x[0]= cur_img


    y = model.predict(x)

    if y[0,0] >= y[0,1] and testY[index,0] == 1:
        n_good_result += 1

    if y[0,0] < y[0,1] and testY[index,1] == 1:
        n_good_result += 1

print(n_good_result*100/total_num_images)
