import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean

import tensorflow as tf
from tensorflow import keras


__all__ = ['TilePairGenerator',
           'augment']

# Augmentations
AUGMENTATIONS = {
    'flips': True,
    'rotation': True,
    'translation': False,
    'scale': False,
    'contrast': False,
    'brightness': False
}


class TilePairGenerator(keras.utils.Sequence):
    """Generates batches of EM, FM tile pairs to facilitate training & testing

    Parameters
    ----------
    batch_size : int
        Batch size
    img_size : tuple
        Image shape, typically (1024, 1024)
    fps_src : list
        List of input EM filepaths
    fps_tgt : list
        List of target FM filepaths
    """

    def __init__(self, batch_size, fps_src, fps_tgt,
                 augmentations=None):
        self.batch_size = batch_size
        self.fps_src = fps_src
        self.fps_tgt = fps_tgt
        # Set up default augmentations
        self.augment = augment
        if augmentations is None:
            self.augmentations = augmentations

    def __len__(self):
        return len(self.fps_tgt) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple of images (source, target) corresponding to batch index"""
        i = idx * self.batch_size

        # Get batch of EM, FM filepaths
        fps_src_batch = self.fps_src[i: (i+self.batch_size)]
        fps_tgt_batch = self.fps_tgt[i: (i+self.batch_size)]

        # Create batches of EM and FM images
        batch_EM = []
        batch_FM = []
        for fp_EM, fp_FM in zip(fps_src_batch, fps_tgt_batch):
            image_EM, image_FM = self.fetch_image_pairs(fp_EM, fp_FM)
            batch_EM.append(image_EM)
            batch_FM.append(image_FM)

        return np.array(batch_EM), np.array(batch_FM)

    def fetch_image_pairs(self, fp_EM, fp_FM):
        """"""
        image_EM = imread(fp_EM) / 255.
        image_FM = imread(fp_FM) / 255.

        # Apply augmentations
        if self.augment:
            # Augmentation functions in tf.keras.preprocessing.image
            # require 3 channel (RGB) input images
            image = np.stack([image_EM, image_FM], axis=2)
            image = augment(image, self.augmentations)
            image_EM = image[:,:,0]
            image_FM = image[:,:,1]

        # Downscale FM (1024, 1024) --> (256, 256)
        image_FM = downscale_local_mean(image_FM, factors=(4, 4))

        # Add dummy axis to make tensorflow happy
        # and convert to float16 to save on memory
        image_EM = image_EM[..., np.newaxis].astype(np.float16)
        image_FM = image_FM[..., np.newaxis].astype(np.float16)

        return image_EM, image_FM


def augment(image, flips=True, rotation=True, translation=True,
            scale=True, contrast=True, brightness=True):
    """Apply image augmentation

    Parameters
    ----------
    images : list
        Input images to be augmented

    Returns
    -------
    images : list
        Augmented images
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

#     # Translation
#     if translation:
#         image = tf.keras.preprocessing.image\
#                   .random_shift(image, wrg=0.2, hrg=0.2, **kwargs)

#     # Scale
#     if scale:
#         kwargs['fill_mode'] = 'constant'
#         image = tf.keras.preprocessing.image\
#                   .random_zoom(image, zoom_range=(0.8, 1.2), **kwargs)

    # Contrast / brightness
    if contrast:
        image = tf.image.random_contrast(image, 0.75, 1.5).numpy()
    if brightness:
        image = tf.image.random_brightness(image, 0.2).numpy()

    return image
