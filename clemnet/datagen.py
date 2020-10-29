import numpy as np
from skimage.transform import downscale_local_mean
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


__all__ = ['TilePairGenerator']


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

    def __init__(self, batch_size, img_size, fps_src, fps_tgt):
        self.batch_size = batch_size
        self.img_size = img_size
        self.fps_src = fps_src
        self.fps_tgt = fps_tgt

    def __len__(self):
        return len(self.fps_tgt) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple of images (source, target) corresponding to batch index"""
        i = idx * self.batch_size

        # Get batch of EM, FM filepaths
        fps_src_batch = self.fps_src[i: (i+self.batch_size)]
        fps_tgt_batch = self.fps_tgt[i: (i+self.batch_size)]

        # Create batch of EM images
        batch_EM = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='float16')
        for j, fp in enumerate(fps_src_batch):
            image = load_img(fp, target_size=self.img_size, color_mode='grayscale')
            batch_EM[j] = np.expand_dims(image, 2) / 255.

        # Create batch of FM images
        batch_FM = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='float16')
        for j, fp in enumerate(fps_tgt_batch):
            image = load_img(fp, target_size=self.img_size, color_mode='grayscale')
            batch_FM[j] = np.expand_dims(image, 2) / 255.
            batch_FM[j] = downscale_local_mean(batch_FM[j], (4, 4, 1)))

        return batch_EM, batch_FM
