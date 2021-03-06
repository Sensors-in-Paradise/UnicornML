"""
Windowizer, Converter, new structure, working version
"""


import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from utils.DataConfig import OpportunityConfig
import utils.settings as settings
from utils.array_operations import split_list_by_percentage

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter

from models.JensModel import JensModel
from models.MultilaneConv import MultilaneConv
from models.BestPerformerConv import BestPerformerConv
from models.OldLSTM import OldLSTM
from models.SenselessDeepConvLSTM import SenselessDeepConvLSTM
from models.LeanderDeepConvLSTM import LeanderDeepConvLSTM
from utils.DataConfig import SonarConfig
from utils.data_set import DataSet

experiment_name = "sonar_template_exp"
currentDT = datetime.now()
currentDT_str = currentDT.strftime("%y-%m-%d_%H-%M-%S_%f")
experiment_name = experiment_name + "-" + currentDT_str

# Init
data_config = OpportunityConfig(dataset_path='../../data/opportunity-dataset')
settings.init(data_config)
window_size = 30 * 3
n_classes = len(data_config.activity_idx_to_activity_name_map)

# Lib -----------------------------------------------------------


def leave_recording_out_split(test_percentage): return lambda recordings: split_list_by_percentage(
    list_to_split=recordings, percentage_to_split=test_percentage
)
# leave_recording_out_split(test_percentage=0.3)(recordings)


def leave_person_out_split_idx(recordings, test_person_idx):
    def subset_from_condition(condition, recordings): return [
        recording for recording in recordings if condition(recording)
    ]
    recordings_train = subset_from_condition(
        lambda recording: recording.subject != test_person_idx, recordings
    )
    recordings_test = subset_from_condition(
        lambda recording: recording.subject == test_person_idx, recordings
    )
    return recordings_train, recordings_test


def leave_person_out_split(test_person_idx) -> "tuple[DataSet, DataSet]": return lambda recordings: leave_person_out_split_idx(
    recordings=recordings, test_person_idx=test_person_idx
)
# leave_person_out_split(test_person_idx=2)(recordings) # 1-4, TODO: could be random


# Config --------------------------------------------------------------------------------------------------------------


def convert(windows): return Converter(
    n_classes=n_classes).sonar_convert(windows)


def flatten(tuple_list): return [
    item for sublist in tuple_list for item in sublist]


def test_train_split(recordings)-> "tuple[DataSet, DataSet]": return leave_person_out_split(test_person_idx=2)(
    recordings
)


# Load data
recordings = data_config.load_dataset()

random.seed(1678978086101)
random.shuffle(recordings)

# Test Train Split
recordings_train, recordings_test = test_train_split(recordings)

# Windowize
windows_train, windows_test = recordings_train.windowize(window_size),recordings_test.windowize(window_size)

# Convert
X_train, y_train, X_test, y_test = tuple(
    flatten(map(convert, [windows_train, windows_test]))
)

# or JensModel
model = LeanderDeepConvLSTM(
    window_size=window_size,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=n_classes,
    n_epochs=5,
    learning_rate=0.001,
    batch_size=32,
    wandb_config={
       "project": "all_experiments_project",
       "entity": "tfiedlerdev",
       "name": experiment_name,
    },
    input_distribution_mean=data_config.mean,
    input_distribution_variance=data_config.variance,
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    experiment_name
)  # create folder to store results

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(
    experiment_folder_path, y_test_pred, y_test, [accuracy]
)  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
