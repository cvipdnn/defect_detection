import numpy as np
import tensorflow as tf
import os, sys
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D, UpSampling2D, Add, BatchNormalization
from keras.utils import np_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow import keras
import pydot


# the return unit is GFLOPs

def calcutate_mul(model, showDetails = True ):
    total_mul = 0

    input_size = 1
    for i in range(1,len(model.input.shape)):
        input_size *= model.input.shape[i]

    # the input size , it is not used now
    #print(input_size)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D ) == True:
            #print(layer.kernel_size)
            #print(layer.filters)
            #print(layer.input.shape)
            #print(layer.output.shape)

            # output size * kernelsize * the number of input channels
            current_layer_flops = layer.output.shape[1] * layer.output.shape[2] * layer.kernel_size[0] * \
            layer.kernel_size[1] * layer.input.shape[3]

            # the number of output channels
            current_layer_flops = layer.output.shape[3]*current_layer_flops

            total_mul += current_layer_flops
            if showDetails:
                print(layer.name, current_layer_flops/1000/1000/1000)
        if isinstance(layer, tf.keras.layers.SeparableConv2D ) == True:
            #print(layer.kernel_size)
            #print(layer.filters)
            #print(layer.input.shape)
            #print(layer.output.shape)

            # output size * kernelsize
            current_layer_flops = layer.output.shape[1] * layer.output.shape[2] * layer.kernel_size[0] * \
            layer.kernel_size[1]

            # output size * the number of input channels
            current_layer_flops += layer.output.shape[1] * layer.output.shape[2] * layer.input.shape[3]

            # the number of output channels
            current_layer_flops = layer.output.shape[3]*current_layer_flops

            total_mul += current_layer_flops

            if showDetails:
                print(layer.name, current_layer_flops/1000/1000/1000)

        if isinstance(layer, tf.keras.layers.Dense ) == True:
            # output size * input size
            current_layer_flops = layer.output.shape[1] * layer.input.shape[1]


            total_mul += current_layer_flops

            if showDetails:
                print(layer.name, current_layer_flops/1000/1000/1000)

    #print("Summary\n=====================")
    #print(total_mul/1000/1000)
    return total_mul/1000/1000/1000


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input the model file: python nn_summary.py xxxx.h5')
        exit(-1)
    modelfile= sys.argv[1]
    model=load_model(modelfile)
    model.summary()


    total_flops = calcutate_mul(model, True)
    print( total_flops )








