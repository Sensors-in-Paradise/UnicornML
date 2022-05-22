import numpy as np
from utils.typing import assert_type
from tensorflow.keras.utils import to_categorical  # type: ignore
from utils.Window import Window
import utils.settings as settings


class Converter:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def sonar_convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.sensor_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities), num_classes=settings.DATA_CONFIG.n_activities(),
        )

        return np.array(sensor_arrays), np.array(activity_vectors)
    
    def sonar_convert_with_context(self, windows_list: "list[list[Window]]") -> "tuple[np.ndarray, np.ndarray]":
        sensor_arrays_in_context = []
        activity_arrays_in_context = []

        for window_list in windows_list:
            sensor_arrays = list(map(lambda window: window.sensor_array, window_list))
            activities = list(map(lambda window: window.activity, window_list))

            # to_categorical converts the activity_array to the dimensions needed
            activity_vectors = to_categorical(
                np.array(activities), num_classes=settings.DATA_CONFIG.n_activities(),
            )

            sensor_arrays_in_context += np.array(sensor_arrays)
            activity_arrays_in_context += np.array(activity_vectors)
        return sensor_arrays_in_context, activity_arrays_in_context


    def jens_convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        X_train, y_train = self.sonar_convert(windows)
        return np.expand_dims(X_train, -1), y_train

