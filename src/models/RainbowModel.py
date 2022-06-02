# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value

from abc import ABC, abstractmethod
from math import sqrt
from typing import Any, Union
from numpy.core.numeric import full
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical  # type: ignore
from gc import callbacks
import os
from abc import abstractmethod
from random import shuffle
from typing import Union
import numpy as np
import tensorflow as tf
from loader.preprocessing import replaceNaN_ffill_tf, replaceNaN_ffill_numpy
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.python.saved_model.utils_impl import get_saved_model_pb_path  # type: ignore
import utils.settings as settings
from utils.Recording import Recording
from utils.Window import Window
from utils.array_operations import transform_to_subarrays
from utils.folder_operations import create_folders_in_path
from utils.typing import assert_type
import wandb
from wandb.keras import WandbCallback


class RainbowModel(tf.Module):

    # general
    model_name = "model"
    class_weight = None
    model: Any = None

    # Input Params
    n_features: Union[int, None] = None
    n_outputs: Union[int, None] = None
    window_size: Union[int, None] = None
    stride_size: Union[int, None] = None
    class_weight = None

    model: Union[tf.keras.Model, None] = None
    batch_size: Union[int, None] = None
    verbose: Union[int, None] = None
    n_epochs: Union[int, None] = None
    n_features: Union[int, None] = None
    n_outputs: Union[int, None] = None
    learning_rate: Union[float, None] = None

    # Config
    wandb_project: Union[str, None] = None
    verbose: Union[int, None] = 1
    kwargs = None
    callbacks = []

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Child classes should build a model, assign it to self.model = ...
        It can take hyper params as arguments that are intended to be varied in the future.
        If hyper params dont directly influence the model creation (e.g. meant for normalisation),
        they need to be stored as instance variable, that they can be accessed, when needed.

        Base parameters:
        - `input_distribution_mean` - array of float values: mean of distribution of each feature
        - `input_distribution_variance` - array of float values: variance of distribution of each feature
        --> These parameters will be used by the normalization layer (accessible by the function _preprocessing_layer
            in the child classes _create_model method)
        """

        # per feature measures of input distribution
        self.input_distribution_mean = kwargs["input_distribution_mean"]
        self.input_distribution_variance = kwargs["input_distribution_variance"]

        # input size
        self.window_size = kwargs.get("window_size", None)
        self.n_features = kwargs.get("n_features", None)
        self.stride_size = kwargs.get("stride_size", self.window_size)

        # output size
        self.n_outputs = kwargs.get("n_outputs", None)

        # training
        self.batch_size = kwargs.get("batch_size", None)
        self.n_epochs = kwargs.get("n_epochs", None)
        self.learning_rate = kwargs.get("learning_rate", None)
        self.validation_split = kwargs.get("validation_split", 0.2)
        self.class_weight = kwargs.get("class_weight", None)

        # others
        self.wandb_config = kwargs.get("wandb_config", None)
        self.verbose = kwargs.get("verbose", 1)
        self.kwargs = kwargs

        # Important declarations
        assert self.window_size is not None, "window_size is not set"
        assert self.n_features is not None, "n_features is not set"
        assert self.n_outputs is not None, "n_outputs is not set"
        assert self.batch_size is not None, "batch_size is not set"
        assert self.n_epochs is not None, "n_epochs is not set"
        assert self.learning_rate is not None, "learning_rate is not set"
        self.model = self._create_model()
        self.model.summary()

    def _create_model(self) -> tf.keras.Model:
        """
        Subclass Responsibility:
        returns a keras model
        """
        raise NotImplementedError

    def _preprocessing_layer(self, input_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        x = tf.keras.layers.Normalization(
            axis=-1, variance=self.input_distribution_variance, mean=self.input_distribution_mean)(
            input_layer)
        return x

    # The 'train' function takes an input window and a label
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, window_size, n_features], dtype=tf.float32),
        # Binary Classification
        tf.TensorSpec(shape=[None, n_outputs], dtype=tf.float32),
    ])
    def train(self, input_windows, labels):
        replaceNaN_ffill_tf(input_windows)

        with tf.GradientTape() as tape:
            predictions = self.model(input_windows)
            print(self.model)
            loss = self.model.loss(labels, predictions)
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
            [(X_train, (np.ndarray, np.generic)),
             (y_train, (np.ndarray, np.generic))]
        )
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train have to have the same length"

        # Wandb
        callbacks = None
        if self.wandb_config is not None:
            assert (
                self.wandb_config["project"] is not None
            ), "Wandb project name is not set"
            assert (
                self.wandb_config["entity"] is not None
            ), "Wandb entity name is not set"
            assert self.wandb_config["name"] is not None, "Wandb name is not set"

            wandb.init(
                project=str(self.wandb_config["project"]),
                entity=self.wandb_config["entity"],
                name=str(self.wandb_config["name"]),
                settings=wandb.Settings(start_method='fork')
            )
            wandb.config = {
                "learning_rate": self.learning_rate,
                "epochs": self.n_epochs,
                "batch_size": self.batch_size,
            }
            callbacks = [wandb.keras.WandbCallback()]

        self.history = self.model.fit(
            replaceNaN_ffill_numpy(X_train),
            y_train,
            validation_split=self.validation_split,
            epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            class_weight=self.class_weight,
            callbacks=callbacks
        )

    # Predict ------------------------------------------------------------------------
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, window_size, n_features], dtype=tf.float32),
    ])
    def infer(self, input_window):
        replaceNaN_ffill_tf(input_window)
        logits = self.model(input_window)
        probabilities = tf.nn.softmax(logits, axis=-1)
        return {
            "output": probabilities
        }

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        gets a list of windows and returns a list of prediction_vectors
        """
        return self.model.predict(replaceNaN_ffill_numpy(X_test))

    def export(self, path: str) -> None:
        """
        will create an 'export' folder in the path, and save the model there in 3 different formats
        """
        print("Exporting model ...")

        # Define, create folder structure
        export_path = os.path.join(path, "export")
        export_path_raw_model = os.path.join(export_path, "raw_model")
        create_folders_in_path(export_path_raw_model)

        # 1/2 Export raw model ------------------------------------------------------------
        tf.saved_model.save(self.model, export_path_raw_model,  signatures={
            'train':
                self.train.get_concrete_function(),
            'infer':
                self.infer.get_concrete_function(),
            'save':
                self.save.get_concrete_function(),
            'restore':
                self.restore.get_concrete_function(),
        })

        # 2/2 Convert raw model to tflite model -------------------------------------------
        converter = tf.lite.TFLiteConverter.from_saved_model(
            export_path_raw_model)

        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        with open(os.path.join(export_path, f"{self.model_name}.tflite"), "wb") as f:
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
        tensors_to_save = [weight.read_value()
                           for weight in self.model.weights]
        tf.raw_ops.Save(
            filename=checkpoint_path, tensor_names=tensor_names,
            data=tensors_to_save, name='save')
        return {
            "checkpoint_path": checkpoint_path,
        }
