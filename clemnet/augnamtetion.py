import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf

__all__ = ['augment',
           'elastic_transform']


# Default augmentations
DEFAULT_AUGMENTATIONS = {
    'flips': True,
    'rotation': True,
    'translation': False,
    'scale': False,
    'contrast': False,
    'brightness': False
}


def augment(x, y, flip=0, rotation=0, translation=0,
            crop=0, contrast=0, brightness=0, noise=0):
    """Apply various image augmentations

    Parameters
    ----------
    x : `tf.Tensor`
        EM image tensor
    y : `tf.Tensor`
        FM image tensor
    flip : scalar (0, 1)
        Probability of applying flip augmentation
    rotation : scalar (0, 1)
        Probability of applying rotation augmentation

    Returns
    -------
    image : (M, N, d) array
        Augmented image
    """
    # Concatenate tensors
    xy = tf.concat([x, y], axis=2)

    # Flips
    if flip:
        # Give a `flip`% chance of flipping in each direction
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        # Probabilities on probabilities since `random_flip_`
        # already has a built-in 50/50% chance of flipping
        xy = tf.cond(flip > u, lambda: tf.image.random_flip_left_right(xy), lambda: xy)
        xy = tf.cond(flip > u, lambda: tf.image.random_flip_up_down(xy), lambda: xy)

    # Rotation
    if rotation:
        # Give a `rotation`% chance of rotating a multiple of 90degs in a random direction
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        d = tf.random.uniform([], 0, 4, dtype=tf.int32)
        xy = tf.cond(rotation > u, lambda: tf.image.rot90(xy, d), lambda: xy)

    # Crop
    if crop:
        # Give a `crop`% chance of cropping the image 1-20%
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        xy = tf.cond(rotation > u, lambda: crop__(xy), lambda: xy)

#     # Contrast / brightness
#     if contrast:
#         xy = tf.image.random_contrast(xy, 0.75, 1.5)
#     if brightness:
#         xy = tf.image.random_brightness(xy, 0.2)
#     # Noise
#     if noise:
#         raise NotImplementedError("noise not implemented yet.")
# #         xy = 

    # Split back into separate tensors
    x, y = tf.split(xy, 2, axis=2)

    return x, y


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
