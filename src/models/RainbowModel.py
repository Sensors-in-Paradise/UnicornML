# pylint: disable=locally-disabled, multiple-statements, fixme, line-too-long, no-name-in-module, unused_import, wrong-import-order, bad-option-value

import os
from abc import abstractmethod
from typing import Any
from typing import Union

import flatbuffers
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.python.saved_model.utils_impl import get_saved_model_pb_path  # type: ignore
from tflite_support import metadata as _metadata
# pylint: disable=g-direct-tensorflow-import
from tflite_support import metadata_schema_py_generated as _metadata_fb

from loader.preprocessing import replaceNaN_ffill_tf, replaceNaN_ffill_numpy
from utils.folder_operations import create_folders_in_path
from utils.typing import assert_type


class TimeSeriesModelSpecificInfo(object):
    """Holds information that is specifically tied to a time series classifier"""

    def __init__(self, name, version, window_size, sampling_rate, features, device_tags,
                 author, description):
        self.name = name
        self.version = version
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.features = features
        self.device_tags = device_tags
        self.author = author
        self.description = description


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

        # input params
        self.features = kwargs.get("features", None)
        self.device_tags = kwargs.get("device_tags", None)

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
        self.author = kwargs.get("author", None)
        self.version = kwargs.get("version", None)
        self.description = self._create_description()
        self.model_info = self._build_model_info()
        self.wandb_config = kwargs.get("wandb_config", None)
        self.verbose = kwargs.get("verbose", 1)
        self.kwargs = kwargs

        # Important declarations
        assert self.features is not None, "features are not set"
        assert self.device_tags is not None, "device tags are not set"
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

    def _create_description(self) -> Union[str, None]:
        """
        Subclass Responsibility:
        returns a string describing the model
        """
        return None

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
        tf.saved_model.save(self.model, export_path_raw_model, signatures={
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

        self.populate_and_export_metadata(export_path)

    def _build_model_info(self):
        return {self.model_name: TimeSeriesModelSpecificInfo(
            name=self.model_name,
            version=self.version,
            window_size=self.window_size,
            sampling_rate=60,  # TODO: get this from the data timestamps
            features=self.features,
            device_tags=self.device_tags,
            author=self.author,
            description=self.description,
        )}

    def _write_associated_files(self, export_path: str):
        """
        writes the label file, the features file and the device_tags file
        """
        # Write label file
        label_file_name = "labels.txt"
        with open(os.path.join(export_path, label_file_name), "w") as f:
            for label in self.labels:
                f.write(f"{label}\n")

        # Write features file
        features_file_name = "features.txt"
        with open(os.path.join(export_path, features_file_name), "w") as f:
            for feature in self.features:
                f.write(f"{feature}\n")

        # Write device_tags file
        device_tags_file_name = "device_tags.txt"
        with open(os.path.join(export_path, device_tags_file_name), "w") as f:
            for device_tag in self.device_tags:
                f.write(f"{device_tag}\n")

        return label_file_name, features_file_name, device_tags_file_name

    def populate_and_export_metadata(self, export_path: str) -> None:
        label_file, feature_file, device_tags_file = self._write_associated_files(export_path)

        export_model_path = os.path.join(export_path, f"{self.model_name}.tflite")

        # Generate the metadata objects and put them in the model file
        populator = MetadataPopulatorForTimeSeriesClassifier(
            export_model_path, self.model_info, label_file,
            feature_file, device_tags_file)
        populator.populate()

        # Validate the output model file by reading the metadata and produce
        # a json file with the metadata under the export path
        displayer = _metadata.MetadataDisplayer.with_model_file(export_model_path)
        export_json_file = os.path.join(export_path,
                                        self.model_name + ".json")
        json_file = displayer.get_metadata_json()
        with open(export_json_file, "w") as f:
            f.write(json_file)

        print("Finished populating metadata and associated file to the model:")
        print(export_model_path)
        print("The metadata json file has been saved to:")
        print(export_json_file)
        print("The associated file that has been been packed to the model is:")
        print(displayer.get_packed_associated_file_list())

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


class MetadataPopulatorForTimeSeriesClassifier(object):
    """Populates the metadata for a time series classifier"""

    def __init__(self, model_file, model_info, label_file_path, sensor_data_file_path, sensor_file_path):
        self.model_info = model_info
        self.model_file = model_file
        self.label_file_path = label_file_path
        self.sensor_data_file_path = sensor_data_file_path
        self.sensor_file_path = sensor_file_path
        self.metadata_buf = None

    def populate(self):
        """Creates Metadata and the populates it for a time series classifier"""
        self._create_metadata()
        self._populate_metadata()

    def _create_metadata(self):
        """Creates the metadata for a time series classifier"""

        # Creates model info.
        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = self.model_info.name
        model_meta.description = self.model_info.description.format(self.model_info.num_classes)
        model_meta.author = self.model_info.author
        model_meta.version = self.model_info.version
        model_meta.license = "Apache License. Version 2.0 https://www.apache.org/licenses/LICENSE-2.0."

        # Packs associated file for sensor data inputs.
        label_input_file = _metadata_fb.AssociatedFileT()
        label_input_file.name = os.path.basename(self.sensor_data_file_path)
        label_input_file.description = "Names of sensor data inputs."
        label_input_file.type = _metadata_fb.AssociatedFileType.DESCRIPTIONS

        # Packs associated file for sensors.
        sensor_file = _metadata_fb.AssociatedFileT()
        sensor_file.name = os.path.basename(self.sensor_file_path)
        sensor_file.description = "Names of sensors, that data was collected on."
        sensor_file.type = _metadata_fb.AssociatedFileType.DESCRIPTIONS
        model_meta.associatedFiles = [label_input_file, sensor_file]

        # Creates input info.
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = "window"
        input_meta.description = ("Input window to be classified. The expected window has a size of {0} at a sampling "
                                  "rate of {1}. It gets data from the features: {2} and {3} IMU sensors at the "
                                  "positions: {4}".format(self.model_info.window_size, self.model_info.sampling_rate,
                                                          self.model_info.features, self.model_info.num_devices,
                                                          self.model_info.device_tags))
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
        input_meta.content.contentProperties.contentPropertiesType = (
            _metadata_fb.ContentProperties.FeatureProperties)
        input_meta.processUnits = []

        # Creates output info.
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "probability"
        output_meta.description = ("Probabilities of the {0} labels respectively.".format(self.model_info.num_classes))
        output_meta.content = _metadata_fb.ContentT()
        output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
        output_stats = _metadata_fb.StatsT()
        output_stats.max = [1.0]
        output_stats.min = [0.0]
        output_meta.stats = output_stats

        # Packs associated file for label outputs.
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = os.path.basename(self.label_file_path)
        label_file.description = "Labels for classification output."
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        output_meta.associatedFiles = [label_file]

        # Creates subgraph info.
        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        # Builds flatbuffer.
        b = flatbuffers.Builder(0)
        b.Finish(
            model_meta.Pack(b),
            _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
        )
        self.metadata_buf = b.Output()

    def _populate_metadata(self):
        """Populates metadata and label file to the model file."""
        populator = _metadata.MetadataPopulator.with_model_file(self.model_file)
        populator.load_metadata_buffer(self.metadata_buf)
        populator.load_associated_files([self.label_file_path, self.sensor_data_file_path, self.sensor_file_path])
        populator.populate()

    def _normalization_params(self, feature_norm):
        """Creates normalization process unit for each input feature."""
        input_normalization = _metadata_fb.ProcessUnitT()
        input_normalization.optionsType = (
            _metadata_fb.ProcessUnitOptions.NormalizationOptions)
        input_normalization.options = _metadata_fb.NormalizationOptionsT()
        input_normalization.options.mean = feature_norm["mean"]
        input_normalization.options.std = feature_norm["std"]
        return input_normalization
