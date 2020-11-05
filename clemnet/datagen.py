import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean

import tensorflow as tf
from tensorflow import keras

import augmentations


__all__ = ['TilePairGenerator']


class TilePairGenerator(keras.utils.Sequence):
    """Generates batches of EM, FM tile pairs to facilitate training & testing

    Parameters
    ----------
    batch_size : scalar
        Batch size
    img_size : tuple
        Image shape, typically (1024, 1024)
    fps_src : list
        List of input EM filepaths
    fps_tgt : list
        List of target FM filepaths
    augment : bool
        Whether to apply image augmentations
    augmentations_set : dict
        Mapping of augmentations passed to `augmentations.augment`
    """

    def __init__(self, batch_size, fps_src, fps_tgt, augment=False,
                 augmentations_set=None):
        self.batch_size = batch_size
        self.fps_src = fps_src
        self.fps_tgt = fps_tgt
        self.augment = augment
        self.augmentations_set = augmentations.DEFAULT_AUGMENTATIONS \
                                 if augmentations_set is None \
                                 else augmentations_set        

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
            image_EM, image_FM = self.fetch_image_pairs(fp_EM, fp_FM,
                                                        self.augment)
            batch_EM.append(image_EM)
            batch_FM.append(image_FM)

        return np.array(batch_EM), np.array(batch_FM)


    def fetch_image_pairs(self, fp_EM, fp_FM, augment=False):
        """Fetch images for the data generator

        Parameters
        ----------
        fp_EM : str
            Filepath to EM image
        fp_FM : str
            Filepath to FM image
        augment : bool
            Whether to apply image augmentation

        Returns
        -------
        image_EM : (M, N, 1) array
            EM image as 16bit float
        image_FM : (M, N, 1) array
            FM image as 16bit float
        """
        image_EM = imread(fp_EM) / 255.
        image_FM = imread(fp_FM) / 255.

        # Apply augmentations
        if self.augment:
            # Augmentation functions in tf.keras.preprocessing.image
            # require 3 channel (RGB) input images
            image = np.stack([image_EM, image_FM], axis=2)
            image = augmentations.augment(image, **self.augmentations_set)
            image_EM = image[:,:,0]
            image_FM = image[:,:,1]

        # Downscale FM (1024, 1024) --> (256, 256)
        image_FM = downscale_local_mean(image_FM, factors=(4, 4))

        # Add dummy axis to make tensorflow happy
        # and convert to float16 to save on memory
        image_EM = image_EM[..., np.newaxis].astype(np.float16)
        image_FM = image_FM[..., np.newaxis].astype(np.float16)

        return image_EM, image_FM
