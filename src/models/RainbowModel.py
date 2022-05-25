# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value

from gc import callbacks
import os
from abc import ABC, abstractmethod
from random import shuffle
from typing import Any, Union

import numpy as np
import tensorflow as tf  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.python.saved_model.utils_impl import get_saved_model_pb_path  # type: ignore

import utils.settings as settings
from utils.Recording import Recording
from utils.Window import Window
from utils.array_operations import transform_to_subarrays
from utils.folder_operations import create_folders_in_path
from utils.typing import assert_type


class RainbowModel(tf.Module):

    # general
    model_name = None

    # variables that need to be implemented in the child class
    window_size: Union[int, None] = None
    stride_size: Union[int, None] = None
    class_weight = None

    model: Union[tf.keras.Model, None] = None
    batch_size: Union[int, None] = None
    verbose: Union[int, None] = None
    n_epochs: Union[int, None] = None
    n_features: Union[int, None] = None
    n_outputs: Union[int, None] = None
    kwargs = None
    callbacks = []
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Builds a model, assigns it to self.model = ...
        It can take hyper params as arguments that are intended to be varied in the future.
        If hyper params dont directly influence the model creation (e.g. meant for normalisation),
        they need to be stored as instance variable, that they can be accessed, when needed.
        """

        # self.model = None
        # assert (self.model is not None)
        self.kwargs = kwargs

    # @error_after_seconds(600) # after 10 minutes, something is wrong
    def windowize_convert_fit(self, recordings_train: "list[Recording]") -> None:
        """
        For a data efficient comparison between models, the preprocessed data for
        training and evaluation of the model only exists, while this method is running

        shuffles the windows
        """
        assert_type([(recordings_train[0], Recording)])
        X_train, y_train = self.windowize_convert(recordings_train)
        self.fit(X_train, y_train)
        return X_train, y_train

    # Preprocess ----------------------------------------------------------------------

    def windowize_convert(
        self, recordings_train: "list[Recording]", should_shuffle=True
    ) -> "tuple[np.ndarray,np.ndarray]":
        """
        shuffles the windows
        """
        windows_train = self.windowize(recordings_train)
        if should_shuffle:
            shuffle(
                windows_train
            )  # many running windows in a row?, one batch too homogenous?, lets shuffle
        X_train, y_train = self.convert(windows_train)
        return X_train, y_train

    def windowize(self, recordings: "list[Recording]") -> "list[Window]":
        """
        based on the hyper param for window size, windowizes the recording_frames
        convertion to numpy arrays
        """
        assert_type([(recordings[0], Recording)])

        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        assert (
            self.stride_size is not None
        ), "stride_size has to be set in the constructor of your concrete model class, please"

        windows: "list[Window]" = []
        for recording in recordings:
            sensor_array = recording.sensor_frame.to_numpy()
            sensor_subarrays = transform_to_subarrays(
                sensor_array, self.window_size, self.stride_size
            )
            recording_windows = list(
                map(
                    lambda sensor_subarray: Window(
                        sensor_subarray, recording.activity, recording.subject, recording.recording_index
                    ),
                    sensor_subarrays,
                )
            )
            windows.extend(recording_windows)
        return windows

    def convert(self, windows: "list[Window]") -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.sensor_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities),
            num_classes=settings.DATA_CONFIG.n_activities(),
        )

        return np.array(sensor_arrays), np.array(activity_vectors)

    # The 'train' function takes an input window and a label
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, window_size, n_features], dtype=tf.float32),
        tf.TensorSpec(shape=[None, n_outputs], dtype=tf.float32),  # Binary Classification
    ])
    def train(self, input_window, label):
        print("input_window", input_window)
        print("label", label)
        #assert (self.model.loss == "categorical_crossentropy")

        with tf.GradientTape() as tape:
            predictions = self.model(input_window)
            print(self.model)
            loss = self.model.loss(label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        result = {"loss": loss}
        return result
    
    # Fit ----------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the self.model to the data
        """
        assert_type(
            [(X_train, (np.ndarray, np.generic)), (y_train, (np.ndarray, np.generic))]
        )
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train have to have the same length"
        # print(f"Fitting with class weight: {self.class_weight}")
        history = self.model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
            callbacks= callbacks
        )
        self.history = history

    # Predict ------------------------------------------------------------------------
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, window_size, n_features], dtype=tf.float32),
    ])
    def infer(self, input_window):
        logits = self.model(input_window)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities,
            "logits": logits,
        }
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(X_test)

    def export(self, path: str) -> None:
        """
        will create an 'export' folder in the path, and save the model there in 3 different formats
        """
        print("Exporting model ...")

        # Define, create folder structure
        export_path = os.path.join(path, "export")
        export_path_raw_model = os.path.join(export_path, "raw_model")
        create_folders_in_path(export_path_raw_model)

        # 1/3 Export raw model ------------------------------------------------------------
        tf.saved_model.save(self.model,export_path_raw_model,  signatures={
            'train':
                self.train.get_concrete_function(),
            'infer':
                self.infer.get_concrete_function(),
            'save':
                self.save.get_concrete_function(),
            'restore':
                self.restore.get_concrete_function(),
        })

        # 2/3 Export .h5 model ------------------------------------------------------------
        # self.model.save(export_path + "/" + self.model_name + ".h5", save_format="h5")

        # 3/3 Export .h5 model ------------------------------------------------------------
        converter = tf.lite.TFLiteConverter.from_saved_model(export_path_raw_model)

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]  # Refactoring Idea: Optimizations for new tensorflow version
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        with open(os.path.join(export_path,f"{self.model_name}.tflite"), "wb") as f:
            f.write(tflite_model)

        print("Export finished")

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensors_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path,
        }
