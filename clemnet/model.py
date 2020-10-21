from pathlib import Path

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


__all__ = ['get_model']


def get_model(input_shape):
    """
    U-net-like convolutional neural network
    """
    inputs = layers.Input()

    conv0 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    pool0 = layers.MaxPooling2D(2)(conv0)

    conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')(pool0)

    model = keras.Model(inputs=inputs,
                        outputs=conv1)
    return model
