import tensorflow as tf

from .augnamtetion import apply_augmentations
from .augnamtetion import DEFAULT_AUGMENTATIONS


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

    # Resize images to (256, 256)
    image_src = tf.image.resize(image_src, size=[256, 256])
    image_tgt = tf.image.resize(image_tgt, size=[256, 256])

    return image_src, image_tgt


def create_dataset(fps_src, fps_tgt, shuffle=True, buffer_size=None,
                   repeat=False, n_repetitions=None, augment=False,
                   augmentations=None, batch=False, batch_size=None,
                   prefetch=True, n_cores=None):
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

    References
    ----------
    [1] https://cs230.stanford.edu/blog/datapipeline/
    """
    # Choose number of cores if not provided
    if n_cores is None:
        n_cores = AUTOTUNE

    # Create dataset of filepaths
    ds_fps = tf.data.Dataset.from_tensor_slices((fps_src, fps_tgt))

    # Shuffle
    if shuffle:
        # Choose sufficiently high buffer size for proper shuffling
        buffer_size = len(fps_src) if buffer_size is None\
                                   else buffer_size
        ds_fps = ds_fps.shuffle(buffer_size=buffer_size)

    # Repeat
    if repeat:
        # Choose a reasonable(?) number of repetitions if not provided
        if n_repetitions is None:
            # TODO: choose n_repetitions intelligently
            n_repetitions = (17-6+5)//2
        ds_fps = ds_fps.repeat(count=n_repetitions)

    # Load images
    ds = ds_fps.map(load_images, num_parallel_calls=n_cores//2)

    # Augment images
    if augment:
        # Use default augmentations if not provided
        augmentations = DEFAULT_AUGMENTATIONS if augmentations is None\
                                              else augmentations
        # Apply image augmentations
        ds = ds.map(lambda x, y: apply_augmentations(x, y, **augmentations),
                    num_parallel_calls=n_cores//2)

    # Clip intensity values to 0 - 1 range
    ds = ds.map(lambda x, y: (tf.clip_by_value(x, 0, 1),
                              tf.clip_by_value(y, 0, 1)))

    # Convert to float16 to save on GPU RAM
    ds = ds.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype='float16'),
                              tf.image.convert_image_dtype(y, dtype='float16')))

    # Batch
    if batch:
        # Choose a reasonable(?) batch size
        if batch_size is None:
            # TODO: choose batch size intelligently
            batch_size = 16
        ds = ds.batch(batch_size)

    # Prefetch
    if prefetch:
        ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
