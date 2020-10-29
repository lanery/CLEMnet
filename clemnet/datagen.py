import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean
from tensorflow import keras


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

    def __init__(self, batch_size, fps_src, fps_tgt):
        self.batch_size = batch_size
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
        batch_EM = []
        for fp in fps_src_batch:
            image = imread(fp, as_gray=True) / 255.
            image = image[..., np.newaxis].astype(np.float16)
            batch_EM.append(image)

        # Create batch of FM images
        batch_FM = []
        for fp in fps_tgt_batch:
            image = imread(fp, as_gray=True) / 255.
            # Downscale (1024, 1024) --> (256, 256)
            image = downscale_local_mean(image, factors=(4, 4))
            image = image[..., np.newaxis].astype(np.float16)
            batch_FM.append(image)

        return np.array(batch_EM), np.array(batch_FM)
