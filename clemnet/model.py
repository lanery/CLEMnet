from pathlib import Path

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


__all__ = ['get_model']


def get_model(input_shape):
    """
    U-net-like convolutional neural network

    Parameters
    ----------
    input_shape : tuple
        Image data input shape (optional)
        Default: (1024, 1024, 1)

    Returns
    -------
    model : `keras.Model`
    """
    inputs = layers.Input(shape=(*shape, 1))

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
#     x = layers.BatchNormalization()(x)

    for filters in [64, 128]:

        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

    for filters in [128, 64]:

        x = layers.Conv2DTranspose(filters, 3, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(filters, 3, activation='relu', padding='same')(x)
#         x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    return model
