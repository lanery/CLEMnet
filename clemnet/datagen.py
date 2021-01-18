import tensorflow as tf

__all__ = ['load_images',
           'create_dataset']


AUTOTUNE = tf.data.experimental.AUTOTUNE


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
    > `decode_image`
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


def create_dataset(fps_src, fps_tgt, batch_size, augment=False,
                   shuffle=False, prefetch=False):
    """Create dataset from source and target filepaths

    Parameters
    ----------
    fps_src : list-like
        List of source filepaths for training, validation, or testing

    fps_tgt : list-like
        List of target filepaths for training, validation, or testing

    Returns
    -------
    ds : `tf.data.Dataset`
        Returns the (prefetched) `Dataset` object
    """
    # Load images
    ds_fps = tf.data.Dataset.from_tensor_slices((fps_src, fps_tgt))
    ds = ds_fps.map(load_images, num_parallel_calls=AUTOTUNE)

    # Apply augmentations
    if augment:
        raise NotImplementedError("augmentations not yet implemented.")

    # Resize FM
    ds = ds.map(lambda x, y: (x, tf.image.resize(y, size=[256, 256])),
                num_parallel_calls=AUTOTUNE)

    # Convert to float16 to save on GPU RAM
    ds = ds.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype='float16'),
                              tf.image.convert_image_dtype(y, dtype='float16')))

    # Shuffle
    if shuffle:
        ds = ds.shuffle(1000)
    # Batch
    if batch_size:
        ds = ds.batch(batch_size)
    # Prefetch
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
