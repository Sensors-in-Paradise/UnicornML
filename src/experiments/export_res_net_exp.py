"""
Windowizer, Converter, new structure, working version
"""
import random
import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from models.ResNetModel import ResNetModel
from utils.DataConfig import SonarConfig,Sonar22CategoriesConfig
from utils.data_set import DataSet
from utils.DataConfig import Sonar22CategoriesConfig
from utils.folder_operations import new_saved_experiment_folder

# Init
# OpportunityConfig(dataset_path='../../data/opportunity-dataset')
data_config = Sonar22CategoriesConfig(
    dataset_path='../../data/filtered_dataset_without_null')
settings.init(data_config)
window_size = 30 * 3
n_classes = data_config.n_activities()

experiment_folder_path = new_saved_experiment_folder(
    "export_resnet_exp"
)

# Load data
recordings = data_config.load_dataset(limit=10)

random.seed(1678978086101)
random.shuffle(recordings)

# Test Train Split
recordings_train, recordings_test = recordings.split_by_percentage(test_percentage=0.2)
# Windowize
windows_train, windows_test = recordings_train.windowize(window_size), recordings_test.windowize(window_size)

# Convert
X_train, y_train = DataSet.convert_windows_sonar(windows_train)
X_test, y_test = DataSet.convert_windows_sonar(windows_test)

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
    description="ResNet Model for Sonar22 Dataset"   
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

# model.export(experiment_folder_path) # opt: export model to folder
create_conf_matrix(experiment_folder_path, y_test_pred, y_test)
create_text_metrics(
    experiment_folder_path, y_test_pred, y_test, [accuracy]
)  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions
model.export(experiment_folder_path,  features=[],
    device_tags=[],
    class_labels=[])
