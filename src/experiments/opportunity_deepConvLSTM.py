"""
    Model Architecture from the Paper:
        'Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition', 2015
    It's a pretty basic approach for the combination of CNN and LSTM layers.
    Most likely there already exists an enhaced version of this approach.

    Opportunity activities
        0: "null",
        1: "relaxing",
        2: "coffee time",
        3: "early morning",
        4: "cleanup",
        5: "sandwich time"
"""

import os
import random
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
from models.DeepConvLSTM import DeepConvLSTMModel
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score


settings.init()

# Load data
recordings = load_opportunity_dataset(
    settings.opportunity_dataset_path
)  # Refactoring idea: load_dataset(x_sens_reader_func, path_to_dataset)
random.seed(1678978086101)
random.shuffle(recordings)

# TODO: apply recording label filter functions

# Preprocessing
recordings = Preprocessor().jens_preprocess(recordings)

# TODO: save/ load preprocessed data

# Test Train Split
test_percentage = 0.3
recordings_train, recordings_test = split_list_by_percentage(
    recordings, test_percentage
)

# Init, Train
model = DeepConvLSTMModel(
    window_size=24,
    n_features=recordings[0].sensor_frame.shape[1],
    n_outputs=6,
    verbose=1,
    n_epochs=10,
)
model.windowize_convert_fit(recordings_train)

# Test, Evaluate
# labels are always in vector format
X_test, y_test_true = model.windowize_convert(recordings_test)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "opportunity_deepConvLSTM"
)  # create folder to store results

model.export(experiment_folder_path)  # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(
    experiment_folder_path, y_test_pred, y_test_true, [accuracy]
)  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
