
"""
- this experiment does test_train_split on windows
- labels change on average every 2 min, jens uses <1sec windows -> for example "sandwich time" will have many windows that are hard to predict

1 sec 82 %
2 sec 83 %
3 sec 85 %
4 sec 85 %

10 sec 89 %
"""
import os
import random
from loader.load_opportunity_dataset import load_opportunity_dataset
from loader.Preprocessor import Preprocessor
import utils.settings as settings
from utils.array_operations import split_list_by_percentage
from models.JensModel import JensModel
from utils.folder_operations import new_saved_experiment_folder
from evaluation.conf_matrix import create_conf_matrix
from evaluation.text_metrics import create_text_metrics
from evaluation.metrics import accuracy, f1_score


settings.init()

# Load data
recordings = load_opportunity_dataset(settings.opportunity_dataset_path) # Refactoring idea: load_dataset(x_sens_reader_func, path_to_dataset)
random.seed(1678978086101)
random.shuffle(recordings)

# Preprocessing
recordings = Preprocessor().jens_preprocess(recordings)

# Init, Train
model = JensModel(window_size=300, n_features=recordings[0].sensor_frame.shape[1], n_outputs=6, verbose=1, n_epochs=10)

windows = model.windowize(recordings)

test_percentage = 0.3
windows_train, windows_test = split_list_by_percentage(windows, test_percentage)
random.shuffle(windows_train)

X_train, y_train = model.convert(windows_train)
model.fit(X_train, y_train)


# Test, Evaluate
# labels are always in vector format
X_test, y_test = model.convert(windows_test)

X_test, y_test_true = model.convert(windows_train)
y_test_pred = model.predict(X_test)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder('window_size_experiment') # create folder to store results

model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true)
create_text_metrics(experiment_folder_path, y_test_pred, y_test_true, [accuracy]) # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
