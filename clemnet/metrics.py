import numpy as np
from scipy import stats
from skimage.filters import threshold_otsu

from tensorflow.python.keras import backend as K


__all__ = [
    'accuracy'
    'accuracy_tf'
    'PCC'
    'PCC_tf'
    'MCC'
    'ICQ'
    'ICQ_tf'
    'spearman'
    'overlap'
    'intersection'
    'threshold'
    'costes'
]


def accuracy(X, Y):
    """Mean squared error flipped for the sake of accuracy"""
    mse = ((X - Y)**2).mean()
    return 1-mse

def accuracy_tf(X, Y):
    """Mean squared error for tensors flipped for the sake of accuracy"""
    mse = K.mean((X - Y)**2)
    return 1-mse


def PCC(X, Y, thresholding=None):
    """Pearson correlation coefficient"""
    # Apply thresholding
    X, Y = threshold(X, Y, method=thresholding)
    # Calculate Pearson correlation coefficient
    r = stats.pearsonr(X.ravel(), Y.ravel())[0]
    return r

def PCC_tf(X, Y):
    """Pearson correlation coefficient for tensors"""
    x0 = X - K.mean(X)
    y0 = Y - K.mean(Y)
    r = K.sum(x0*y0) / K.sqrt(K.sum(x0**2) * K.sum(y0**2))
    return r


def MCC(X, Y, thresholding='costes'):
    """Manders colocalization coefficients"""
    # Apply thresholding
    X, Y = threshold(X, Y, method=thresholding)
    # Calculate colocalization coefficients
    Xco = np.where(Y > 0, X, 0)
    Yco = np.where(X > 0, Y, 0)
    M1 = Xco.sum() / X.sum()
    M2 = Yco.sum() / Y.sum()
    return 2*(M1-0.5), 2*(M2-0.5)


def ICQ(X, Y, thresholding=None):
    """Intensity correlation quotient"""
    # Apply thresholding
    X, Y = threshold(X, Y, method=thresholding)
    # Calculate intensity quotient
    x0 = X - X.mean()
    y0 = Y - Y.mean()
    q = (x0*y0 > 0).mean()
    return 2*(q-0.5)

def ICQ_tf(X, Y):
    """Intensity correlation quotient for tensors"""
    x0 = X - K.mean(X)
    y0 = Y - K.mean(Y)
    q = K.mean(x0*y0 > 0)
    return 2*(q-0.5)


def spearman(X, Y, thresholding=None):
    """Spearman correlation coefficient"""
    # Apply thresholding
    X, Y = threshold(X, Y, method=thresholding)
    # Calculate Pearson correlation coefficient
    r = stats.spearmanr(X.ravel(), Y.ravel())[0]
    return r


def overlap(X, Y, thresholding=None):
    # Apply thresholding
    X, Y = threshold(X, Y, method=thresholding)
    # Calculate overlap coefficient
    o = (X*Y).sum() / np.sqrt((X**2).sum() * (Y**2).sum())
    return o


def intersection(X, Y, thresholding=None):
    # Apply thresholding
    X, Y = threshold(X, Y, method=thresholding)
    # Calculate intersection coefficient
    i = (X*Y).sum() / (X.sum() + Y.sum() - (X*Y).sum())
    return i


def threshold(X, Y, method=None, c_val=0):
    """Threshold images"""
    # Determine threshold values
    if method == 'costes':
        Tx, Ty = costes(X, Y)
    elif method == 'otsu':
        Tx = threshold_otsu(X)
        Ty = threshold_otsu(Y)
    elif method == 'constant':
        # Allow for independent threshold values
        if hasattr(c_val, '__len__'):
            Tx, Ty = c_val
        else:
            Tx, Ty = c_val, c_val
    else:  # no thresholding
        return X, Y
    # Apply threshold
    X = np.where(X > Tx, X, 0)
    Y = np.where(Y > Ty, Y, 0)
    return X, Y


def costes(X, Y, r_min=0.01):
    """Costes background estimation

    Iteratively descends along the regression line of the 2D histogram
    until PCC falls below 0 (or in this case, `r_min`)

    Parameters
    ----------
    X : (N,) or (M, N) array
        Input array for signal 1
    Y : (N,) or (M, N) array
        Input array for signal 2
    r_min : float, optional
        Threshold Pearson correlation coefficient

    Returns
    -------
    Tx : scalar
        Background threshold for signal 1
    Ty : scalar
        Background threshold for signal 2

    References
    ----------
    [1] https://doi.org/10.1529/biophysj.103.038422
    [2] https://svi.nl/BackgroundEstimation
    """
    # Regression line of 2d histogram
    m, b, *_ = stats.linregress(X.ravel(), Y.ravel())
    # Proper implementation would be to descend along
    # the regression line, but found this to take
    # waaaaaay tooooo loooong (up to 65535 iterations)
    # --> Txs = X.ravel()[np.argsort(X.ravel())[::-1]]
    Txs = np.linspace(X.max(), X.min(), 256)  # 8bit range
    Tys = m*Txs + b

    # Descend along "regression line" until PCC goes below 0
    i = 0
    r = 1
    try:
        while r > r_min:
            Tx = Txs[i]
            Ty = Tys[i]
            # Selectively turn off pixels above threshold
            X = np.where(X > Tx, 0, X)
            Y = np.where(Y > Ty, 0, Y)
            # Calculate PCC
            r = stats.pearsonr(X.ravel(), Y.ravel())[0]
            i += 1
    except IndexError:
        print(f"PCC did not fall below threshold ({r_min}).")
        return Tx, Ty

    return Tx, Ty
