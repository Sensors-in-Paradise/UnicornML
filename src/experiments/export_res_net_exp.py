"""
Windowizer, Converter, new structure, working version
"""

import random

import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from models.ResNetModel import ResNetModel
from utils.Converter import Converter
from utils.DataConfig import Sonar22CategoriesConfig
from utils.Windowizer import Windowizer
from utils.array_operations import split_list_by_percentage
from utils.folder_operations import new_saved_experiment_folder

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


def leave_person_out_split(test_person_idx): return lambda recordings: leave_person_out_split_idx(
    recordings=recordings, test_person_idx=test_person_idx
)


# leave_person_out_split(test_person_idx=2)(recordings) # 1-4, TODO: could be random


# Config --------------------------------------------------------------------------------------------------------------
def windowize(recordings): return Windowizer(window_size=window_size).jens_windowize(
    recordings
)


def convert(windows): return Converter(
    n_classes=n_classes).sonar_convert(windows)


def flatten(tuple_list): return [
    item for sublist in tuple_list for item in sublist]


def test_train_split(recordings): return leave_recording_out_split(test_percentage=.2)(
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
windows_train, windows_test = windowize(
    recordings_train), windowize(recordings_test)

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
    author="TobiUndFelix",
    version="0.1",
    description="ResNet Model for Sonar22 Dataset",
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(
    experiment_folder_path, y_test_pred, y_test, [accuracy]
)  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
model.export(experiment_folder_path)
