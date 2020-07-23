import os
import random

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from skimage import io

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


def getModel():
    """Creates the architecture of the neural network

    Returns
    -------
    model : `keras.Model`
        Neural network model

    Notes
    -----
    J-net architecture
    \
     \
      \   /
       \_/
    """

    img_shape=(None, None, 1)     #input shape of images

    # Create U-net architecture
    inputs = keras.layers.Input(shape = img_shape)  # input layer

    # Layer 0
    conv0 = keras.layers.Conv2D(32, 4, activation='relu', padding='same',
                                kernel_initializer='he_normal')(inputs)
    pool0 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv0)

    # Layer 1
    conv1 = keras.layers.Conv2D(64, 4, activation='relu', padding='same',
                                kernel_initializer='he_normal')(pool0)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Layer 2
    conv2 = keras.layers.Conv2D(128, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal')(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Layer 3
    conv3 = keras.layers.Conv2D(256, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal')(pool2)
    pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Layer 4
    conv4 = keras.layers.Conv2D(512, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal')(pool3)
    pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Layer 5
    conv5 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal')(pool4)
    pool5 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv5)

    # Layer 6
    conv6 = keras.layers.Conv2D(1024, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal')(pool5)
    up6 = keras.layers.UpSampling2D(size=(2, 2))(conv6)

    # Layer 7
    merge7 = keras.layers.concatenate([conv5, up6], axis=3)
    conv7 = keras.layers.Conv2DTranspose(1024, 3, activation='relu', padding='same',
                                         kernel_initializer='he_normal')(merge7)

    # Layer 8
    conv8 = keras.layers.Conv2DTranspose(1024, 3, activation='relu', padding='same',
                                         kernel_initializer='he_normal')(conv7)

    conv9 = keras.layers.Conv2DTranspose(1, 1, activation= 'sigmoid')(conv7)

    # Create model
    model = keras.Model(inputs=inputs, outputs=conv9)
    return model 
