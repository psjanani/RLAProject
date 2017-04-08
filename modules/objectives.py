"""Loss functions."""

import tensorflow as tf
import semver
import numpy as np
from keras import backend as K

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    
    # will be [batch_size, 6]
    # one-hot encodings ... only take max==sum
    y_true= K.sum(y_true, axis=1)
    y_pred = K.sum(y_pred, axis=1)

    # calculate error deltas
    delta_abs = K.abs(y_true - y_pred)
    sq = 0.5 * K.square(delta_abs)
    li = max_grad * (delta_abs - 0.5 * max_grad)
    # determine regions by gamma threshold
    #result = K.switch(delta_abs[0] < max_grad, lambda: 0.5 * K.square(delta_abs), lambda: max_grad * (delta_abs - 0.5 * max_grad))
    result = tf.where(delta_abs[0] < max_grad, sq, li)

    return result


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """

    return K.mean(huber_loss(y_true, y_pred, max_grad))
