import numpy as np
from skimage import color, exposure


__all__ = ['colorize',
           'T_HOECHST',
           'T_INSULIN']


# Color transformations
T_HOECHST = [[0.2, 0.0, 0.0, 0.2],
             [0.0, 0.2, 0.0, 0.2],
             [0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0]]

T_INSULIN = [[1.0, 0.0, 0.0, 1.0],
             [0.0, 0.6, 0.0, 0.6],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]


def colorize(image, T):
    """Colorize image

    Parameters
    ----------
    image : numpy array

    Returns
    -------
    rescaled : rgba float array
    Image array after color transformation
    """
    # Convert to rgba
    rgba = color.grey2rgb(image, alpha=True)
    # Apply transform
    transformed = np.dot(rgba, T)
    rescaled = exposure.rescale_intensity(transformed)
    return rescaled


def get_n_tensors(model):
    """
    Parameters
    ----------
    model : `keras.Model`
        Model (duh)

    Returns
    -------
    n_tensors : int
        Number of tensors in model
    """
    # First get shape of tensors in each layer
    # (skipping the first index which is the batch size)
    tensor_shapes = [layer.input_shape[1:] if isinstance(layer.input_shape, tuple)\
                                           else layer.input_shape[0][1:]\
                                           for layer in model.layers]
    n_tensors = np.product(tensor_shapes, axis=1).sum()
    return n_tensors


def get_max_batch_size(gpu_ram, model):
    """
    Parameters
    ----------
    gpu_ram : float
        Size of RAM GPU in GB
    model : `keras.Model`
        Model (duh)

    Returns
    -------
    max_batch_size : int
        Maximum number of batches possible without exceeding memory limitations

    Notes
    -----
    Max batch size given by
                GPU RAM / 4
        ----------------------------------
        (N_tensors + N_trainable_parameters)
    Ref: https://stackoverflow.com/a/46656508/5285918
    Optimal number of batches should be the highest power of 2 < max batch size
    """
    n_tensors = get_n_tensors(model)
    max_batch_size = (1e9*gpu_ram / 4) / (n_tensors + model.count_params())
    return max_batch_size
