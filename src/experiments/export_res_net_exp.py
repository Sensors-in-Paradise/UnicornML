"""
Windowizer, Converter, new structure, working version
"""


import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from utils.DataConfig import OpportunityConfig, Sonar22CategoriesConfig
import utils.settings as settings
from utils.array_operations import split_list_by_percentage

from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score
from utils.Windowizer import Windowizer
from sklearn.model_selection import KFold
from utils.Converter import Converter

from models.ResNetModel import ResNetModel
from utils.DataConfig import SonarConfig
from utils.data_set import DataSet

# Init
# OpportunityConfig(dataset_path='../../data/opportunity-dataset')
data_config = Sonar22CategoriesConfig(
    dataset_path='../../data/filtered_dataset_without_null')
settings.init(data_config)
window_size = 30 * 3
n_classes = len(data_config.activity_idx_to_activity_name_map)


experiment_folder_path = new_saved_experiment_folder(
    "export_resnet_exp"
)


def leave_recording_out_split(test_percentage): return lambda recordings: split_list_by_percentage(
    list_to_split=recordings, percentage_to_split=test_percentage
)


# Config --------------------------------------------------------------------------------------------------------------
def convert(windows): return Converter(
    n_classes=n_classes).sonar_convert(windows)


def flatten(tuple_list): return [
    item for sublist in tuple_list for item in sublist]


def test_train_split(recordings: DataSet) -> DataSet: return leave_recording_out_split(test_percentage=.2)(
    recordings
)


# Load data
recordings = data_config.load_dataset(limit=10)

random.seed(1678978086101)
random.shuffle(recordings)

# Test Train Split
recordings_train, recordings_test = test_train_split(recordings)
print(recordings_train)
# Windowize
windows_train, windows_test = recordings_train.windowize(window_size), recordings_test.windowize(window_size)

# Convert
X_train, y_train, X_test, y_test = tuple(
    flatten(map(convert, [windows_train, windows_test]))
)

# or JensModel
model = ResNetModel(
    window_size=window_size,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=n_classes,
    n_epochs=1,
    learning_rate=0.001,
    batch_size=32,
    input_distribution_mean=data_config.mean,
    input_distribution_variance=data_config.variance,
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(
    experiment_folder_path, y_test_pred, y_test, [accuracy]
)  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
model.export(experiment_folder_path)