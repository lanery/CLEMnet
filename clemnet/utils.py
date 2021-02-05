from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, exposure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


__all__ = ['get_n_tensors',
           'get_max_batch_size',
           'parse_tensorboard_logs',
           'colorize',
           'T_HOECHST',
           'T_INSULIN',
           'T_RED',
           'T_GREEN',
           'T_BLUE']


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
    # Then add 'em up
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
    # Get number of tensors in model
    n_tensors = get_n_tensors(model)
    # Calculate max batch size
    max_batch_size = (1e9*gpu_ram / 4) / (n_tensors + model.count_params())
    return max_batch_size


def parse_tensorboard_logs(log_dir):
    """Parse tensorboard log directory data

    Parameters
    ----------
    log_dir : `path.Path`
        Log directory

    Returns
    -------
    df : `pd.DataFrame`
        DataFrame of tensorboard data
    """
    # Initialize lists for DataFrame
    runs = []
    datasets = []
    tags = []
    wall_times = []
    steps = []
    values = []

    # Loop through log directories
    for fp_run in log_dir.glob('[!.]*'):
        # Either '../train' or '../validation'
        for fp_trnval in fp_run.glob('*'):
            try:  # perhaps training didn't complete? \_0_/
                fp_event = list(fp_trnval.glob('*.v2'))[0]
            except IndexError as err:
                continue
            events = EventAccumulator(fp_event.as_posix()).Reload()
            # Loop through recorded events
            # e.g. ['epoch_loss', 'epoch_accuracy', 'epoch_pearson']
            for tag in events.Tags()['scalars']:
                runs.append(fp_run.name)
                datasets.append(fp_trnval.name)
                tags.append(tag)
                wall_times.append([event.wall_time for event in events.Scalars(tag)])
                steps.append([event.step for event in events.Scalars(tag)])
                values.append([event.value for event in events.Scalars(tag)])

    # Build DataFrame
    df = pd.DataFrame({
        'run': runs,
        'dataset': datasets,
        'tag': tags,
        'wall_time': wall_times,
        'step': steps,
        'value': values
    })
    # Explode events
    df = df.apply(lambda x: pd.Series.explode(x))\
           .reset_index(drop=True)
    return df


def histogram2d(X, Y, bins=64, **kwargs):
    """Wrapper for np.histogram2d to return bin centers"""
    H, x_edges, y_edges = np.histogram2d(X.ravel(), Y.ravel(),
                                         bins=bins, **kwargs)
    x_centers = (x_edges[1:] + x_edges[:-1]) / 2
    y_centers = (y_edges[1:] + y_edges[:-1]) / 2
    return H, x_centers, y_centers


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
    rgba = color.gray2rgb(image, alpha=True)
    # Apply transform
    transformed = np.dot(rgba, T)
    rescaled = exposure.rescale_intensity(transformed)
    return rescaled


# Color transformations
# ---------------------
# Labels
T_HOECHST = [[0.2, 0.0, 0.0, 0.2],  # blueish
             [0.0, 0.2, 0.0, 0.2],
             [0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0]]
T_INSULIN = [[1.0, 0.0, 0.0, 1.0],  # orangeish
             [0.0, 0.6, 0.0, 0.6],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]]
# Primary colors
T_RED = [[1.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0]]
T_GREEN = [[0.0, 0.0, 0.0, 0.0],
           [0.0, 1.0, 0.0, 1.0],
           [0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0]]
T_BLUE = [[0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0]]
