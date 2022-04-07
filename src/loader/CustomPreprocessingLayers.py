from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
# currently expects recording as inputs
# TODO: after the windowizing has been moved to be executed before the preprocessing this has to be changed accordingly

class InterpolationLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_interpolate", **kwargs):
        super(InterpolationLayer).__init__(name=name, **kwargs)

    def get_config(self):
        super(InterpolationLayer, self).get_config()

    def call(self, inputs, **kwargs):
        for recording in inputs:
            recording.sensor_frame.interpolate()


class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_normalize", **kwargs):
        super(NormalizationLayer, self).__init__(name=name, **kwargs)

    def get_config(self):
        super(NormalizationLayer, self).get_config()

    def call(self, inputs, **kwargs):
        scaler = MinMaxScaler()
        for recording in inputs:
            scaler.fit(recording.sensor_frame)

        # Then apply normalization on each recording_frame
        for recording in inputs:
            transformed_array = scaler.transform(recording.sensor_frame)
            recording.sensor_frame = pd.DataFrame(
                transformed_array, columns=recording.sensor_frame.columns
            )
