from datetime import datetime
from utils.data_set import DataSet

from utils.Recording import Recording
from utils.cache_recordings import load_recordings
from utils.typing import assert_type
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.load_sonar_dataset import load_sonar_dataset
import itertools
import json
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
import tensorflow as tf
import math


@dataclass
class DataConfig:
    """
    Config Interface:
    (can be used generic)
        data_config.activity_idx_to_activity_name(activity_idx) (subclasses need to define the mapping that is required for that)
        data_config.load_dataset()
    """

    # Dataset Config (subclass responsibility) -----------

    raw_label_to_activity_idx_map = None
    raw_subject_to_subject_idx_map = None

    activity_idx_to_activity_name_map = None
    subject_idx_to_subject_name_map = None

    timestep_frequency = None  # Hz
    variance = None
    mean = None
    DATA_CONFIG_METADATA_FILE = "dataConfigMetadata.json"

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_dataset(self, **kwargs) -> DataSet:
        features = kwargs.get("features", None)
        if features != None:
            self.features = features
            del kwargs["features"]

        recordings = self._load_dataset(**kwargs)

        for recording in recordings:
            if features != None:
                recording.sensor_frame = recording.sensor_frame[features]

            recording.sensor_frame = recording.sensor_frame.fillna(
                method="ffill")

        variance, mean = self._loadDataSetMeasures()
        if variance is None or mean is None:
            print(
                "Calculating mean and variance of whole dataset once. This can take a while...")
            startTime = datetime.now()

            sensor_frames = tf.constant(np.concatenate(
                [recording.sensor_frame.to_numpy() for recording in recordings], axis=0))
            layer = tf.keras.layers.Normalization(axis=-1)
            layer.adapt(sensor_frames)
            self.variance = layer.variance
            self.mean = layer.mean
            endTime = datetime.now()
            print("Time spent for finding mean and variance: ",
                  str(endTime-startTime))
            self._saveDataSetMeasures(self.variance, self.mean)
        else:
            self.variance = variance
            self.mean = mean

        ds = DataSet(recordings, self)
        ds.replaceNaN_ffill()
        return ds

    # interface (subclass responsibility to define) ------------------------------------------------------------
    def _load_dataset(self, **kwargs) -> "list[Recording]":
        raise NotImplementedError(
            "init subclass of Config that defines the method activity_idx_to_activity_name"
        )

    # generic
    def raw_label_to_activity_idx(self, label: str) -> int:
        """
        from the label as it is saved in the dataset, to the activity index
        (Relabeling)
        """
        assert (
            self.raw_label_to_activity_idx_map is not None
        ), "A subclass of Config which initializes the var raw_label_to_activity_idx_map should be used to access activity mapping."
        return self.raw_label_to_activity_idx_map[label]

    def raw_subject_to_subject_idx(self, subject: str) -> int:
        assert (
            self.raw_subject_to_subject_idx_map is not None
        ), "A subclass of Config which initializes the var raw_subject_to_subject_idx_map should be used to access subject mapping."
        return self.raw_subject_to_subject_idx_map[subject]

    def subject_idx_to_subject_name(self, subject_idx: int) -> str:
        assert (
            self.subject_idx_to_subject_name_map is not None
        ), "A subclass of Config which initializes the var subject_idx_to_subject_name_map should be used to access subject mapping."
        assert_type((subject_idx, int))
        return self.subject_idx_to_subject_name_map[subject_idx]

    def activity_idx_to_activity_name(self, activity_idx: int) -> str:
        assert (
            self.activity_idx_to_activity_name_map is not None
        ), "A subclass of Config which initializes the var activity_idx_to_activity_name_map should be used to access activity mapping."
        assert_type((activity_idx, int))
        return self.activity_idx_to_activity_name_map[activity_idx]

    def n_activities(self) -> int:
        assert (
            self.activity_idx_to_activity_name_map is not None
        ), "A subclass of Config which initializes the var activity_idx_to_activity_name_map should be used to access activity mapping."
        return len(self.activity_idx_to_activity_name_map)

    def _loadDataSetMeasures(self):
        """
        Returns feature wise variance and mean for the dataset of this data config if available, else Tuple of None, None
        """
        measures = self._loadMeasuresDict()
        if not measures is None:
            return np.array(measures["variance"]), np.array(measures["mean"])
        return None, None

    def _saveDataSetMeasures(self, variances, mean):
        metadata = {}
        identifier = self._getDataConfigIdentifier()

        measures = {
            "variance": np.array(variances).tolist(),
            "mean": np.array(mean).tolist(),
        }
        metadata = self._loadMetadataDict()
        metadata[identifier] = measures
        with open(
            DataConfig.DATA_CONFIG_METADATA_FILE, "w", encoding="utf8"
        ) as json_file:
            json.dump(metadata, json_file, indent=5)

    def _loadMeasuresDict(self):
        """
        Returns the metadata json of this data config's metadata as dict
        """
        metadata = self._loadMetadataDict()
        identifier = self._getDataConfigIdentifier()
        if identifier in metadata:
            return metadata[identifier]
        return None

    def _loadMetadataDict(self):
        """
        Returns the metadata json of this data config's metadata as dict
        """
        if os.path.isfile(DataConfig.DATA_CONFIG_METADATA_FILE):
            with open(
                DataConfig.DATA_CONFIG_METADATA_FILE, "r", encoding="utf8"
            ) as json_file:
                metadata = json.load(json_file)
                return metadata
        return {}

    def _getDataConfigIdentifier(self):
        return type(self).__name__ + self.dataset_path + ", ".join(self.features) if self.features != None else ""
