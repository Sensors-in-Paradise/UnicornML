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
        recordings = self._load_dataset(**kwargs)
        variance, mean = self._loadDataSetMeasures()
        if variance is None or mean is None:
            print(
            "Calculating mean and variance of whole dataset once. This can take a while...")
            startTime = datetime.now()
            for recording in recordings:
                recording.sensor_frame = recording.sensor_frame.fillna(
                    method="ffill")
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
        return recordings

    # interface (subclass responsibility to define) ------------------------------------------------------------
    def _load_dataset(self) -> DataSet:
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
        return type(self).__name__ + self.dataset_path


class OpportunityConfig(DataConfig):

    timestep_frequency = 30  # Hz

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.original_idx_to_activity_idx_map = {
            0: 0,
            101: 1,
            102: 2,
            103: 3,
            104: 4,
            105: 5,
        }

        self.raw_label_to_activity_idx_map = {
            "null": 0,
            "relaxing": 1,
            "coffee time": 2,
            "early morning": 3,
            "cleanup": 4,
            "sandwich time": 5,
        }

        self.activity_idx_to_activity_name_map = {
            0: "null",
            1: "relaxing",
            2: "coffee time",
            3: "early morning",
            4: "cleanup",
            5: "sandwich time",
        }

    def _load_dataset(self) -> DataSet:
        return load_opportunity_dataset(self.dataset_path)



class SonarConfig(DataConfig):

    timestep_frequency = 60  # Hz

    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        labels = list(
            itertools.chain.from_iterable(
                category["entries"] for category in self.category_labels
            )
        )
        self.raw_label_to_activity_idx_map = {
            label: i for i, label in enumerate(labels)
        }  # no relabeling applied
        activities = {k: v for v, k in enumerate(labels)}
        self.activity_idx_to_activity_name_map = {
            v: k for k, v in activities.items()}

        self.raw_subject_to_subject_idx_map = {
            key: value for value, key in enumerate(self.raw_subject_label)
        }
        self.subject_idx_to_subject_name_map = {
            v: k for k, v in self.raw_subject_to_subject_idx_map.items()
        }  # just the inverse, do relabeling here, if needed

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def _load_dataset(self, **args) -> DataSet:
        return load_sonar_dataset(self.dataset_path, **args)

    raw_subject_label = [
        "unknown",
        "christine",
        "aileen",
        "connie",
        "yvan",
        "brueggemann",
        "jenny",
        "mathias",
        "kathi",
        "anja",
    ]
    category_labels = [
        {
            "category": "Others",
            "entries": [
                "invalid",
                "null - activity",
                "aufräumen",
                "aufwischen (staub)",
                "blumen gießen",
                "corona test",
                "kaffee kochen",
                "schrank aufräumen",
                "wagen schieben",
                "wäsche umräumen",
                "wäsche zusammenlegen",
            ],
        },
        {
            "category": "Morgenpflege",
            "entries": [
                "accessoires (parfüm) anlegen",
                "bad vorbereiten",
                "bett machen",
                "bett beziehen",
                "haare kämmen",
                "hautpflege",
                "ikp-versorgung",
                "kateterleerung",
                "kateterwechsel",
                "medikamente geben",
                "mundpflege",
                "nägel schneiden",
                "rasieren",
                "umkleiden",
                "verband anlegen",
            ],
        },
        {
            "category": "Waschen",
            "entries": [
                "duschen",
                "föhnen",
                "gegenstand waschen",
                "gesamtwaschen im bett",
                "haare waschen",
                "rücken waschen",
                "waschen am waschbecken",
                "wasser holen",
            ],
        },
        {
            "category": "Mahlzeiten",
            "entries": [
                "essen auf teller geben",
                "essen austragen",
                "essen reichen",
                "geschirr austeilen",
                "geschirr einsammeln",
                "getränke ausschenken",
                "getränk geben",
                "küche aufräumen",
                "küchenvorbereitungen",
                "tablett tragen",
            ],
        },
        {
            "category": "Assistieren",
            "entries": [
                "arm halten",
                "assistieren - aufstehen",
                "assistieren - hinsetzen",
                "assistieren - laufen",
                "insulingabe",
                "patient umlagern (lagerung)",
                "pflastern",
                "rollstuhl modifizieren",
                "rollstuhl schieben",
                "rollstuhl transfer",
                "toilettengang",
            ],
        },
        {
            "category": "Organisation",
            "entries": [
                "arbeiten am computer",
                "dokumentation",
                "medikamente stellen",
                "telefonieren",
            ],
        },
    ]



class Sonar22CategoriesConfig(DataConfig):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

        self.raw_label_to_activity_idx_map = self.category_labels  # no relabeling applied
        self.activity_idx_to_activity_name_map = {
            k: v for v, k in self.raw_label_to_activity_idx_map.items()}

        # SONAR SPECIFIC VARS --------------------------------------------------

        self.sensor_suffix_order = ["LF", "LW", "ST", "RW", "RF"]
        self.csv_header_size = 8

    def _load_dataset(self, **args) -> DataSet:
        return load_recordings(self.dataset_path, self.raw_label_to_activity_idx_map, **args)

    category_labels = {'rollstuhl transfer': 0, 'essen reichen': 1, 'umkleiden': 2, 'bad vorbereiten': 3, 'bett machen': 4, 'gesamtwaschen im bett': 5, 'aufräumen': 6, 'geschirr einsammeln': 7, 'essen austragen': 8, 'getränke ausschenken': 9, 'küchenvorbereitung': 10,
                       'waschen am waschbecken': 11, 'rollstuhl schieben': 12, 'mundpflege': 13, 'haare kämmen': 14, 'essen auf teller geben': 15, 'dokumentation': 16, 'aufwischen (staub)': 17, 'haare waschen': 18, 'medikamente stellen': 19, 'accessoires anlegen': 20, 'föhnen': 21}

