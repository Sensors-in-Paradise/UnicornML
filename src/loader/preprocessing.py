from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

from utils import settings
from utils.Recording import Recording
from utils.typing import assert_type
import math
import tensorflow as tf
import numpy as np


def replaceNaN_ffill(recordings: "list[Recording]") -> "list[Recording]":
    """
    the recordings have None values, this function replaces them with the last non-NaN value of the feature
    """
    assert_type([(recordings[0], Recording)])
    fill_method = "ffill"
    for recording in recordings:
        recording.sensor_frame = recording.sensor_frame.fillna(
            method=fill_method)
        recording.sensor_frame = recording.sensor_frame.fillna(
            0)
    return recordings


def _replaceNaN_ffill2D_tf(t: tf.Tensor):
    """
    Using only tensorflow api functions:
    the recordings have None values, this function replaces them with the last non-NaN value of the feature

    t - tensor with feature vectors e.g. valid inputs would be `tf.constant([[NaN, 0.3, 0,2], [0.5, 0.7, 0.9]])`
    """
    # Find non-NaN values
    mask = ~tf.math.is_nan(t)
    # Take non-NaN values and precede them with a NaN
    values = tf.concat([[math.nan], tf.boolean_mask(t, mask)], axis=0)
    # Use cumsum over mask to find the index of the non-NaN value to pick
    idx = tf.cumsum(tf.cast(mask, tf.int64), axis=0)
    # Gather values
    result = tf.gather(values, idx)
    # Find non-NaN values (if row of tensor started with NaN)
    result = tf.where(tf.math.is_nan(result), tf.ones_like(result) * 0, result)
    return result


def replaceNaN_ffill_tf(t: tf.Tensor):
    """
    Using only tensorflow api functions:
    The recordings have None values, this function replaces them with the last non-NaN value of the feature.

    t - tensor with arbritrary deep feature vectors e.g. valid inputs would be `tf.constant([[NaN, 0.3, 0,2], [0.5, 0.7, 0.9]])`
        or `tf.constant([[[NaN, 0.3, 0,2], [0.5, 0.7, 0.9]]])`
    """
    if len(tf.shape(t)) > 2:
        return tf.map_fn(fn=replaceNaN_ffill_tf, elems=t)
    else:
        return tf.map_fn(fn=_replaceNaN_ffill2D_tf, elems=t)


def _replaceNaN_ffill_numpy2D(arr: np.ndarray):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    out = np.nan_to_num(out)
    return out


def replaceNaN_ffill_numpy(arr):
    return np.array(list(map(_replaceNaN_ffill_numpy2D, arr)))
