import tensorflow as tf

__all__ = ['load_images']


def load_images(fp_src, fp_tgt):
    """
    Parameters
    ----------
    fp_src : str
        Filepath to EM image
    fp_tgt : str
        Filepath to corresponding FM image

    Returns
    -------
    image_src : (M, N, 1) array
        EM image rescaled to (1024, 1024) float16 array
    image_tgt : (M, N, 1)
        FM image rescaled to (256, 256) float16 array

    Notes
    -----
    > `decod_image`
      * `expand_animations` is set to False so that the tensor returned
        by `decode_image` has a shape
      * automatically rescales intensity to (0, 1) range for dtype float32
    """
    # Read images as float32
    image_src = tf.io.decode_image(tf.io.read_file(fp_src),
                                   dtype='float32',
                                   expand_animations=False)
    image_tgt = tf.io.decode_image(tf.io.read_file(fp_tgt),
                                   dtype='float32',
                                   expand_animations=False)
    return image_src, image_tgt
