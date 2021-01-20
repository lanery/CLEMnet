import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf
import tensorflow_addons as tfa


__all__ = ['apply_augmentations',
           'DEFAULT_AUGMENTATIONS']


# Default augmentations
DEFAULT_AUGMENTATIONS = {
    'flip': 0.9,
    'rotation': 0.9,
    'crop': 0.5,
    'elastic': 0.5,
    'contrast': 0.5,
    'brightness': 0.5,
    'noise': 0.5,
}


def apply_augmentations(x, y, flip=0, rotation=0, translation=0, crop=0,
                        elastic=0, contrast=0, brightness=0, noise=0):
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
        # Probabilities on probabilities since `random_flip_a_b`
        # already has a built-in 50/50% chance of flipping
        xy = tf.cond(u > flip, lambda: xy, 
                               lambda: tf.image.random_flip_left_right(xy))
        xy = tf.cond(u > flip, lambda: xy,
                               lambda: tf.image.random_flip_up_down(xy))
    # Rotation
    if rotation:
        # Give a `rotation`% chance of rotating 0, 90, 180, or 270deg
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        d = tf.random.uniform([], 0, 4, dtype=tf.int32)
        xy = tf.cond(u > rotation, lambda: xy,
                                   lambda: tf.image.rot90(xy, d))
    # Crop
    if crop:
        # Give a `crop`% chance of cropping the image 1-20%
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        xy = tf.cond(u > crop, lambda: xy,
                               lambda: crop_augmentation(xy))
    # Elastic deformation
    if elastic:
        # Give an `elastic`% chance of applying an elastic deformation
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        flow = tf.random.normal([1, 256, 256, 2]) * (1+2*u)
        xy_ = tf.expand_dims(xy, axis=0)
        xy = tf.cond(u > elastic, lambda: xy,
                                  lambda: tf.squeeze(tfa.image.dense_image_warp(xy_, flow), axis=0))

    # Split back into separate tensors for remaining augmentations
    # which are only applied to the EM images
    x, y = tf.split(xy, 2, axis=2)

    # Contrast
    if contrast:
        # Give a `contrast`% chance of adjusting the contrast 75-150%
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        x = tf.cond(u > contrast, lambda: x,
                                  lambda: tf.image.random_contrast(x, 0.75, 1.5))
    # Brightness
    if brightness:
        # Give a `brightness`% chance of adjusting the brightness +/-20%
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        x = tf.cond(u > brightness, lambda: x,
                                    lambda: tf.image.random_brightness(x, 0.2))
    # Noise
    if noise:
        # Give a `noise`% chance of applying Poissonian noise
        u = tf.random.uniform([], 0, 1, dtype=tf.float32)
        lam = tf.random.uniform([], 4, 8, dtype=tf.float32)
        x = tf.cond(u > noise, lambda: x,
                               lambda: tf.random.poisson([], x*lam)/lam)

    return x, y


def crop_augmentation(x):
    """Apply a random crop augmentation

    References
    ----------
    [1] https://www.wouterbulten.nl/blog/tech/
        data-augmentation-using-tensorflow-data-dataset/
    """
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes,
                                         box_indices=np.zeros(len(scales)),
                                         crop_size=(512, 512))
        # Return a random crop
        return crops[tf.random.uniform([], 0, len(scales), dtype=tf.int32)]

    return random_crop(x)
