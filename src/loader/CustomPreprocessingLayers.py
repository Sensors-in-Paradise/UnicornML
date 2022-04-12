from typing import List

import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf


# currently expects recording as inputs
# TODO: after the windowizing has been moved to be executed before the preprocessing this has to be changed accordingly

class Interpolation(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_interpolate", **kwargs):
        super(Interpolation, self).__init__(name=name, **kwargs)

    def get_config(self):
        super(Interpolation, self).get_config()

    def call(self, inputs):
        for col in range(inputs.shape[2]):
            mean = tf.reduce_mean(inputs[0, :, col])
            tf.map_fn(lambda x: tf.where(tf.math.is_nan(inputs[0, :, col]),
                                         tf.cast(tf.fill(inputs[0, :, col].shape, mean), dtype=tf.float32),
                                         inputs[0, :, col]), inputs[0, :, col])
        return inputs


class FillNaN(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_fillNaN", **kwargs):
        super(FillNaN, self).__init__(name=name, **kwargs)

    def get_config(self):
        super(FillNaN, self).get_config()

    def call(self, inputs):
        print(tf.executing_eagerly())
        print(np.ndarray(inputs))
        then = time.time()
        tf.map_fn(
            lambda value: tf.math.multiply_no_nan(
                value, tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(value)), dtype=tf.float32)),
            inputs
        )
        now = time.time()
        print(f"execution time of FillNaN call: {now - then}")
        return inputs


class Normalization(tf.keras.layers.Layer):
    def __init__(self, scaler=MinMaxScaler(), name="preprocess_normalize", **kwargs):
        self.scaler = scaler
        super(Normalization, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        then = time.time()
        for col in range(inputs.shape[2]):
            column = inputs[0, :, col]
            max_val = tf.math.reduce_max(column)
            min_val = tf.math.reduce_min(column)
            tf.map_fn(lambda x: tf.divide(tf.subtract(x, min_val), tf.subtract(max_val, min_val)), column)
        now = time.time()
        print(f"execution time of Normalization call: {now - then}")
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'input_shape': self.input_shape,
        }
        base_config = super(Normalization, self).get_config()
        return dict(base_config.items() + config.items())


class ExpandDimensions(tf.keras.layers.Layer):
    def __init__(self, axis, name="preprocess_expand_dimensions", **kwargs):
        self.axis = axis
        super(ExpandDimensions, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
