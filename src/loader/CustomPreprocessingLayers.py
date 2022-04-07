from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf


class NormalizeInterpolate(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_normalize_interpolate", **kwargs):
        super(NormalizeInterpolate).__init__(name=name, **kwargs)

    def get_config(self):
        super(NormalizeInterpolate, self).get_config()

    def call(self, inputs, **kwargs):
        for recording in inputs:
            recording.sensor_frame.interpolate()
            