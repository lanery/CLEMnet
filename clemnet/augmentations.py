import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf

__all__ = ['augment',
           'elastic_transform',
           'DEFAULT_AUGMENTATIONS']


# Default augmentations
DEFAULT_AUGMENTATIONS = {
    'flips': True,
    'rotation': True,
    'translation': False,
    'scale': False,
    'contrast': False,
    'brightness': False
}


def augment(image, flips=True, rotation=True, translation=True,
            scale=True, contrast=True, brightness=True):
    """Apply image augmentation

    Parameters
    ----------
    image : (M, N, d) array
        Input image to be augmented

    Returns
    -------
    image : (M, N, d) array
        Augmented image
    """
    kwargs = {'row_axis': 0,
              'col_axis': 1,
              'channel_axis': 2,
              'fill_mode': 'reflect'}
    # Flips
    if flips:
        image = tf.image.random_flip_left_right(image).numpy()
        image = tf.image.random_flip_up_down(image).numpy()
    # Rotation
    if rotation:
        image = tf.keras.preprocessing.image\
                  .random_rotation(image, rg=30, **kwargs)
    # Translation
    if translation:
        image = tf.keras.preprocessing.image\
                  .random_shift(image, wrg=0.2, hrg=0.2, **kwargs)
    # Scale
    if scale:
        kwargs['fill_mode'] = 'constant'
        image = tf.keras.preprocessing.image\
                  .random_zoom(image, zoom_range=(0.8, 1.2), **kwargs)
    # Contrast / brightness
    if contrast:
        image = tf.image.random_contrast(image, 0.75, 1.5).numpy()
    if brightness:
        image = tf.image.random_brightness(image, 0.2).numpy()

    return image


def elastic_transform(image, alpha=400, sigma=20):
    """Apply a randomly generated elastic tranform

    Parameters
    ----------
    image : (M, N, d) array
        Input image to be transformed
    alpha : scalar
        Amplitude of Gaussian filtered noise
    sigma : scalar
        Smoothing factor, standard deviation for Gaussian kernel

    Returns
    -------
    transformed : (M, N, d) array
        Elastically transformed image    
    """

    # Gaussian filter some noise
    dx = gaussian_filter((np.random.rand(*image.shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*image.shape) * 2 - 1), sigma) * alpha

    # Create distortion grid
    x, y, z = np.meshgrid(np.arange(image.shape[1]),
                          np.arange(image.shape[0]),
                          np.arange(image.shape[2]))
    indices = (np.reshape(y+dy, (-1, 1)),
               np.reshape(x+dx, (-1, 1)),
               np.reshape(z, (-1, 1)))
    transformed = map_coordinates(image, indices, order=1, mode='reflect')

    return transformed.reshape(image.shape)
