from pathlib import Path

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


__all__ = ['get_model',
           'get_unet',
           'get_dummy']


def get_model(input_shape=(1024, 1024), crop=False, crop_width=None):
    """U-net-like convolutional neural network
    Parameters
    ----------
    input_shape : tuple
        Shape of input image data
    crop : bool
        Whether to include a cropping layer
    crop_width : int
        Number of pixels to crop from each border

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
    # ----------------
    # Block 1
    conv1 = layers.Conv2D(32, 3, **kwargs)(inputs)
    pool1 = layers.MaxPooling2D(2)(conv1)
    # Block 2
    conv2 = layers.Conv2D(64, 3, **kwargs)(pool1)
    pool2 = layers.MaxPooling2D(2)(conv2)
    # Block 3
    conv3 = layers.Conv2D(128, 3, **kwargs)(pool2)
    pool3 = layers.MaxPooling2D(2)(conv3)
    # Block 4
    conv4 = layers.Conv2D(256, 3, **kwargs)(pool3)
    pool4 = layers.MaxPooling2D(2)(conv4)
    # Block 5
    conv5 = layers.Conv2D(512, 3, **kwargs)(pool4)
    pool5 = layers.MaxPooling2D(2)(conv5)
    # Block 6
    conv6 = layers.Conv2D(1024, 3, **kwargs)(pool5)
    pool6 = layers.MaxPooling2D(2)(conv6)

    # Upsampling arm
    # --------------
    # Block 7
    conv7 = layers.Conv2D(1024, 3, **kwargs)(pool6)
    uppp7 = layers.UpSampling2D(2)(conv7)
    # Block 8
    merg8 = layers.concatenate([conv6, uppp7], axis=3)
    conv8 = layers.Conv2D(1024, 3, **kwargs)(merg8)
    uppp8 = layers.UpSampling2D(2)(conv8)
    # Block 9
    merg9 = layers.concatenate([conv5, uppp8], axis=3)
    conv9 = layers.Conv2D(512, 3, **kwargs)(merg9)
    uppp9 = layers.UpSampling2D(2)(conv9)
    # Block 10
    merg10 = layers.concatenate([conv4, uppp9], axis=3)
    conv10 = layers.Conv2D(256, 3, **kwargs)(merg10)
    conv10 = layers.Conv2D(2, 3, **kwargs)(conv10)

    # Additional upsampling
    uppp11 = layers.UpSampling2D(2)(conv10)

    # Cropping layer
    if crop is not None:
        cropping = ((crop_width, crop_width), (crop_width, crop_width))
        uppp11 = layers.Cropping2D(cropping=cropping)(uppp11)

    # Output layer
    conv11 = layers.Conv2D(1, 1, activation='sigmoid')(uppp11)
    model = keras.Model(inputs=inputs, outputs=conv11)
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


def get_dummy(input_shape=(256, 256), filter_depth=3, seed=123):
    """Dummy U-net-like convolutional neural network for testing purposes

    Parameters
    ----------
    input_shape : tuple
        Shape of input image data

    Returns
    -------
    model : `keras.Model`
        The model (duh)
    """
    # Model setup
    # -----------
    # Create input layer
    input_shape = (*input_shape, 1) if len(input_shape) < 3 else input_shape
    inputs = layers.Input(shape=input_shape)
    # Keyword arguments for convolutional layers
    ki = keras.initializers.he_normal(seed=seed)
    kwargs = {
        'activation': 'relu',
        'padding': 'same',
        'kernel_initializer': ki
    }
    # Filters
    filters = [2**(5+n) for n in range(filter_depth)]

    # Entry block
    # -----------
    x = layers.Conv2D(16, 3, **kwargs)(inputs)
    previous_block_activation = x  # Set aside residual

    # Downsampling arm
    # ----------------
    for f in filters:
        # Convolution
        x = layers.Conv2D(f, 3, **kwargs)(x)
        x = layers.Conv2D(f, 3, **kwargs)(x)
        x = layers.MaxPooling2D(2)(x)

        # Project residual
        residual = layers.Conv2D(f, 1, strides=2, kernel_initializer=ki)(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Upsampling arm
    # --------------
    for filters in filters[::-1]:
        x = layers.Conv2D(f, 3, **kwargs)(x)
        x = layers.Conv2D(f, 3, **kwargs)(x)
        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(f, 1, kernel_initializer=ki)(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Output per-pixel classification layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer=ki)(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model
