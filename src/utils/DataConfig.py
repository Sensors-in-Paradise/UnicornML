from utils.typing import assert_type
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.load_sonar_dataset import load_sonar_dataset
import itertools
import json

import pandas as pd
from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    DataConfig Interface:
    (can be used generic)
        data_config.activity_idx_to_name(activity_idx) (subclasses need to define the mapping that is required for that)
        data_config.load_dataset()
    """

    # interface (subclass responsibility to define) ------------------------------------------------------------

    """
        Refactoring idea:
        find a better solution:
        - __init__ in DataConfig is never called! 
        - self.activity_idx_to_name_map defined for the typing
    """
    def __init__(self):
        self.activity_idx_to_name_map = None

    def load_dataset(self) -> 'list[Recording]':
        raise NotImplementedError("init subclass of DataConfig that defines the method activity_idx_to_name")

    # generic
    def activity_idx_to_name(self, activity_idx: int) -> str:
        assert self.activity_idx_to_name_map is not None, "init subclass of DataConfig that defines the var activity_idx_to_name_map"
        assert_type((activity_idx, int))
        return self.activity_idx_to_name_map[activity_idx]
    
    def n_activities(self) -> int:
        assert self.activity_idx_to_name_map is not None, "init subclass of DataConfig that defines the var activity_idx_to_name_map"
        return len(self.activity_idx_to_name_map)

class OpportunityDataConfig(DataConfig):

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.activity_idx_to_name_map = {
            0: "null",
            1: "relaxing",
            2: "coffee time",
            3: "early morning",
            4: "cleanup",
            5: "sandwich time",
        }

        # Custom vars
        self.initial_higher_level_label_to_activity_idx = {
            0: 0,
            101: 1,
            102: 2,
            103: 3,
            104: 4,
            105: 5,
        }


    def load_dataset(self) -> 'list[Recording]':
        return load_opportunity_dataset(self.dataset_path)

class SonarDataConfig(DataConfig):        

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

        labels = None
        with open("labels.json") as file:
            categories = json.load(file)["items"]
            labels = list(
                itertools.chain.from_iterable(
                    category["entries"] for category in categories
                )
            )
        activities = {k: v for v, k in enumerate(labels)}
        self.activity_idx_to_name_map =  {v: k for k, v in activities.items()}

        # SHOULD NOT BE NEEDED!!!! --------------------------------------------------

        # global SENSOR_SUFFIX_ORDER
        # SENSOR_SUFFIX_ORDER = ["LF", "LW", "ST", "RW", "RF"]

        # global CSV_HEADER_SIZE
        # CSV_HEADER_SIZE = 8

    def load_dataset(self) -> 'list[Recording]':
        return load_sonar_dataset(self.dataset_path)
