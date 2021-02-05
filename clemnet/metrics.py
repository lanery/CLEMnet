import numpy as np
from scipy import stats
from tensorflow.python.keras import backend as K


def pearson(X, Y):
    """Pearson correlation coefficient"""
    x0 = X - K.mean(X)
    y0 = Y - K.mean(Y)
    r = K.sum(x0*y0) / K.sqrt(K.sum(x0**2) * K.sum(y0**2))
    return r


# def acc_pred(y_true, y_pred):
#     'accuracy metric as described in progress report'
#     a = (1-abs(y_true-y_pred))
#     b = K.sum(a)/(256**2)/batch_size
#     return b


def costes(X, Y, r_min=0.01):
    """Costes background estimation

    Iteratively descends along the regression line of the 2D histogram
    until PCC falls below 0 (or in this case, `r_min`)

    Parameters
    ----------
    X : 

    Returns
    -------

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
