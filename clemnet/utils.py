import numpy as np
from skimage import color, exposure


# Color transformations
T_HOECHST = [[0.2, 0.0, 0.0, 0.2],
             [0.0, 0.2, 0.0, 0.2],
             [0.0, 0.0, 1.0, 1.0],
             [0.0, 0.0, 0.0, 0.0]]

T_INSULIN = [[1.0, 0.0, 0.0, 1.0],
             [0.0, 0.6, 0.0, 0.6],
             [0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0]],


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
