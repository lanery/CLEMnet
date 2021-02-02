from pathlib import Path

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


__all__ = ['get_model',
           'get_unet',
           'get_ogish_clemnet']


def get_model(input_shape=(256, 256), kernel_initializer=None):
    """U-net-like convolutional neural network

    Parameters
    ----------
    input_shape : tuple
        Shape of input image data

    Returns
    -------
    model : `keras.Model`
        The model (duh)

    References
    ----------
    [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    [2] Adapted from
        https://github.com/zhixuhao/unet/blob/master/model.py
    """
    # Create input layer
    input_shape = (*input_shape, 1) if len(input_shape) < 3 else input_shape
    inputs = layers.Input(shape=input_shape)

    # Set up keyword arguments for convolutional layers
    ki = 'he_normal' if kernel_initializer is None \
                                           else kernel_initializer
    kwargs = {
        'activation': 'relu',
        'padding': 'same',
        'kernel_initializer': ki
    }

    # Downsampling arm
    # ----------------
    # Block 1
    conv1 = layers.Conv2D(64, 3, **kwargs)(inputs)
    conv1 = layers.Conv2D(64, 3, **kwargs)(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)
    # Block 2
    conv2 = layers.Conv2D(128, 3, **kwargs)(pool1)
    conv2 = layers.Conv2D(128, 3, **kwargs)(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)
    # Block 3
    conv3 = layers.Conv2D(256, 3, **kwargs)(pool2)
    conv3 = layers.Conv2D(256, 3, **kwargs)(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)
    # Block 4
    conv4 = layers.Conv2D(512, 3, **kwargs)(pool3)
    conv4 = layers.Conv2D(512, 3, **kwargs)(conv4)
    drop4 = layers.Dropout(0.5, seed=345)(conv4)
    pool4 = layers.MaxPooling2D(2)(drop4)
    # Block 5 (bottom of the U)
    conv5 = layers.Conv2D(1024, 3, **kwargs)(pool4)
    conv5 = layers.Conv2D(1024, 3, **kwargs)(conv5)
    drop5 = layers.Dropout(0.5, seed=345)(conv5)

    # Upsampling arm
    # --------------
    # Block 6
    up6 = layers.Conv2D(512, 3, **kwargs)(layers.UpSampling2D(2)(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, **kwargs)(merge6)
    conv6 = layers.Conv2D(512, 3, **kwargs)(conv6)
    # Block 7
    up7 = layers.Conv2D(256, 3, **kwargs)(layers.UpSampling2D(2)(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, **kwargs)(merge7)
    conv7 = layers.Conv2D(256, 3, **kwargs)(conv7)
    conv7 = layers.Conv2D(2, 3, **kwargs)(conv7)

    # Output layer
    conv8 = layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    model = keras.Model(inputs=inputs, outputs=conv8)

    return model


def get_unet(input_shape=(256, 256)):
    """Convolutional network architecture for fast segmentation of images

    Parameters
    ----------
    input_shape : tuple
        Image data input shape (optional)

    Returns
    -------
    model : `keras.Model`
        The model (duh)

    References
    ----------
    [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
    [2] Adapted from
        https://github.com/zhixuhao/unet/blob/master/model.py
    """
    # Create input layer
    input_shape = (*input_shape, 1) if len(input_shape) < 3 else input_shape
    inputs = layers.Input(shape=input_shape)

    # Set up keyword arguments for convolutional layers
    kwargs = {
        'activation': 'relu',
        'padding': 'same',
        'kernel_initializer': 'he_normal'
    }

    # Downsampling arm
    conv1 = layers.Conv2D(64, 3, **kwargs)(inputs)
    conv1 = layers.Conv2D(64, 3, **kwargs)(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)
    conv2 = layers.Conv2D(128, 3, **kwargs)(pool1)
    conv2 = layers.Conv2D(128, 3, **kwargs)(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)
    conv3 = layers.Conv2D(256, 3, **kwargs)(pool2)
    conv3 = layers.Conv2D(256, 3, **kwargs)(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)
    conv4 = layers.Conv2D(512, 3, **kwargs)(pool3)
    conv4 = layers.Conv2D(512, 3, **kwargs)(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(2)(drop4)

    conv5 = layers.Conv2D(1024, 3, **kwargs)(pool4)
    conv5 = layers.Conv2D(1024, 3, **kwargs)(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # Upsampling arm
    up6 = layers.Conv2D(512, 2, **kwargs)(layers.UpSampling2D(2)(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, **kwargs)(merge6)
    conv6 = layers.Conv2D(512, 3, **kwargs)(conv6)

    up7 = layers.Conv2D(256, 2, **kwargs)(layers.UpSampling2D(2)(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, **kwargs)(merge7)
    conv7 = layers.Conv2D(256, 3, **kwargs)(conv7)

    up8 = layers.Conv2D(128, 2, **kwargs)(layers.UpSampling2D(2)(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, **kwargs)(merge8)
    conv8 = layers.Conv2D(128, 3, **kwargs)(conv8)

    up9 = layers.Conv2D(64, 2, **kwargs)(layers.UpSampling2D(2)(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, **kwargs)(merge9)
    conv9 = layers.Conv2D(64, 3, **kwargs)(conv9)
    conv9 = layers.Conv2D(2, 3, **kwargs)(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = keras.Model(inputs=inputs, outputs=conv10)

    return model


def get_ogish_clemnet(input_shape=(1024, 1024)):
    """
    """
    # Create input layer
    input_shape = (*input_shape, 1) if len(input_shape) < 3 else input_shape
    inputs = layers.Input(shape=input_shape)

    # Set up keyword arguments for convolutional layers
    kwargs = {
        'activation': 'relu',
        'padding': 'same',
        'kernel_initializer': 'he_normal'
    }

    # Downsampling layers
    conv0 = keras.layers.Conv2D(32, 3, **kwargs)(inputs)
    pool0 = keras.layers.MaxPooling2D(2)(conv0)
    conv1 = keras.layers.Conv2D(64, 3, **kwargs)(pool0)
    pool1 = keras.layers.MaxPooling2D(2)(conv1)
    conv2 = keras.layers.Conv2D(128, 3, **kwargs)(pool1)
    pool2 = keras.layers.MaxPooling2D(2)(conv2)
    conv3 = keras.layers.Conv2D(256, 3, **kwargs)(pool2)
    pool3 = keras.layers.MaxPooling2D(2)(conv3)
    conv4 = keras.layers.Conv2D(512, 3, **kwargs)(pool3)
    pool4 = keras.layers.MaxPooling2D(2)(conv4)
    conv5 = keras.layers.Conv2D(1024, 3, **kwargs)(pool4)
    pool5 = keras.layers.MaxPooling2D(2)(conv5)
    drop5 = keras.layers.Dropout(0.5)(pool5)

    # Upsampling layers
    conv6 = keras.layers.Conv2D(512, 3, **kwargs)(drop5)
    up6 = keras.layers.UpSampling2D(2)(conv6)
    merge7 = keras.layers.concatenate([conv5, up6], axis=3)
    conv7 = keras.layers.Conv2D(256, 3, **kwargs)(merge7)
    up7 = keras.layers.UpSampling2D(2)(conv7)
    merge8 = keras.layers.concatenate([conv4, up7], axis=3)
    conv8 = keras.layers.Conv2D(128, 3, **kwargs)(merge8)
    up8 = keras.layers.UpSampling2D(2)(conv8)
    merge9 = keras.layers.concatenate([conv3, up8], axis=3)
    conv9 = keras.layers.Conv2D(64, 3, **kwargs)(merge9)
    up9 = keras.layers.UpSampling2D(2)(conv9)

    # Output layer
    conv10 = keras.layers.Conv2D(1, 1, activation= 'sigmoid')(up9)

    # Build model
    model = keras.Model(inputs=inputs, outputs=conv10)
    return model
