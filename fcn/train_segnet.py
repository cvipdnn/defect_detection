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
    # 1/4 resolution now

    res_1_4 = x;
    # residual layer
    x = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv31',
               data_format=IMAGE_ORDERING)(x)
    x = SeparableConv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv32',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(48, (1, 1), strides=(1, 1), padding='same', name='conv33',
               data_format=IMAGE_ORDERING)(x)

    #x = Add()([res_1_4, x])
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)

    #1/8 resolution now
    res_1_8 = x;

    x = Conv2D(48, (1, 1), strides=(1, 1), activation='relu', padding='same', name='conv34',
               data_format=IMAGE_ORDERING)(x)
    x = SeparableConv2D(48, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv35',
               data_format=IMAGE_ORDERING)(x)
    x = Conv2D(48, (1, 1), strides=(1, 1), padding='same', name='conv36',
               data_format=IMAGE_ORDERING)(x)

    #x = Add()([res_1_8, x])
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)

    # 1/16 resolution now
    res_1_16 = x


    x = Conv2D(n_class, (3, 3), kernel_initializer='he_normal', activation='linear', padding='same', name='fc2',
               data_format=IMAGE_ORDERING)(x)
    x = UpSampling2D(size=(2, 2))(x)

    # 1/8 resolution
    class_res_1_8 = Conv2D(n_class, (3, 3), kernel_initializer='he_normal', activation='linear', padding='same', name='fc6',
                data_format=IMAGE_ORDERING)(res_1_8)

    x = Add()([class_res_1_8, x]);
    x = UpSampling2D(size=(2, 2))(x);

    class_res_1_4 = Conv2D(n_class, (3, 3), kernel_initializer='he_normal', activation='linear', padding='same', name='fc7',
                data_format=IMAGE_ORDERING)(res_1_4)

    x = Add()([class_res_1_4, x]);

    x = UpSampling2D(size=(4, 4))(x)
    x = Activation('softmax')(x);

    model = Model(img_input, x)
    return model



def calSamplenum(images, y):
    num_images = len(images)

    num_background = 0
    num_foreground = 0

    for i in range(num_images):

        imageObj = images[i]
        current_defect_pixels = 0
        if y[i] != '0':
            mask_img= plt.imread(y[i])

            current_defect_pixels = np.sum(mask_img)

        num_background += (image_size*image_size-current_defect_pixels)
        num_foreground += current_defect_pixels


    return num_background, num_foreground

def imageGenerator(images, y, batch_size=32):
    image_num = images.shape[0]
    print(image_num)
    c = 0
    index_shuf = list(range(image_num))
    while (True):
        img = np.zeros((batch_size, image_size, image_size))

        mask_img = np.zeros((batch_size, image_size, image_size, n_class))

        mask_img[:,:,:,0]=1

        for i in range(c, c + batch_size):
            index = index_shuf[i]
            imageObj = images[index]

            ### Extract Image ###
            cur_img = plt.imread(imageObj)

            img[i - c] = cur_img
            if y[index] != '0':
                mask_img [ i -c ,:,:,1]= plt.imread(y[index])

        c = c + batch_size
        if (c + batch_size >= image_num):
            c = 0
            random.shuffle(index_shuf)

        yield img, mask_img


dataDir='/home/cvip/deep_learning/datasets/DAGM_KaggleUpload/'



# read train image and labels
dirList = os.listdir(dataDir)

images = []
mask_images = []
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
            if (row[4]!='0'):
                mask_images.append(curDirTrain_Label+'/'+row[4])
            else:
                # no defect
                mask_images.append(row[4])

images = np.array(images)
mask_images = np.array(mask_images)

model = SimpleFCN(image_size, image_size)
model.load_weights('./model/pretrained.h5')
model.summary()


# 80% used for training , 20% used for validation
x_train, x_valid, y_train, y_valid = train_test_split(images, mask_images, test_size=0.2, random_state = 32)

batchsize = 4
train_gen = imageGenerator(x_train, y_train, batch_size = batchsize)
val_gen = imageGenerator(x_valid, y_valid, batch_size = batchsize)


train_image_num = x_train.shape[0]

val_image_num = x_valid.shape[0]

epochs = 2560
learning_rate = 0.0001
decay_rate = learning_rate/ epochs
opt = Adam(lr= learning_rate, decay = decay_rate)


steps_per_epoch = train_image_num // batchsize


validation_steps = val_image_num // batchsize

# handle sample imbalance : it is about 231, below I use 256 instead
#n_bg, n_fg = calSamplenum(images, mask_images)
#print(n_bg, n_fg, n_bg/n_fg)

def w_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        # avoid log(0) and log(1)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

model.compile(loss = w_categorical_crossentropy([1,256]), optimizer = opt, metrics= ['accuracy'])


mc = ModelCheckpoint('wt{epoch:05d}.h5', save_weights_only=True, save_freq=steps_per_epoch)


history = model.fit(x=train_gen,
                validation_data=val_gen,
                steps_per_epoch=steps_per_epoch,
                validation_steps = validation_steps,
                callbacks = [mc],
                epochs = epochs,
                verbose = True)
model.save('./model/fullModel.h5')
#print(history.history.keys())
# display history
plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.subplot(212)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.show()
