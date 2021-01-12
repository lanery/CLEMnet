import numpy as np
from tensorflow.python.keras import backend as K


def pearson(y_true, y_pred):
    'pearson correlation coefficient'
    x0 = y_true-K.mean(y_true)
    y0 = y_pred-K.mean(y_pred) 
    return K.sum(x0*y0) / K.sqrt(K.sum((x0)**2)*K.sum((y0)**2))


# def acc_pred(y_true, y_pred):
#     'accuracy metric as described in progress report'
#     a = (1-abs(y_true-y_pred))
#     b = K.sum(a)/(256**2)/batch_size
#     return b
