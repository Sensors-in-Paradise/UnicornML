"""
Windowizer, Converter, new structure, working version
"""
import utils.settings as settings
from evaluation.conf_matrix import create_conf_matrix
from evaluation.metrics import accuracy
from evaluation.text_metrics import create_text_metrics
from models.GaitAnalysisTLModel import GaitAnalysisTLModel
from data_configs.gait_analysis_config import GaitAnalysisConfig
from utils.data_set import DataSet
from utils.folder_operations import new_saved_experiment_folder
from sklearn.utils import shuffle
import tensorflow as tf
import os

# Init
data_config = GaitAnalysisConfig(dataset_path="../../data/fatigue_dual_task")
features = [
    "GYR_X_LF",
    "GYR_Y_LF",
    "GYR_Z_LF",
    "ACC_X_LF",
    "ACC_Y_LF",
    "ACC_Z_LF",
    "GYR_X_RF",
    "GYR_Y_RF",
    "GYR_Z_RF",
    "ACC_X_RF",
    "ACC_Y_RF",
    "ACC_Z_RF",
    "GYR_X_SA",
    "GYR_Y_SA",
    "GYR_Z_SA",
    "ACC_X_SA",
    "ACC_Y_SA",
    "ACC_Z_SA",
]

settings.init(data_config)
window_size = 1000
n_classes = data_config.n_activities()

experiment_folder_path = new_saved_experiment_folder("export_GaitAnalysisTL_exp")
subs = [
    "sub_01",
    "sub_02",
    "sub_03",
    "sub_05",
    "sub_06",
    "sub_07",
    "sub_08",
    "sub_09",
    "sub_10",
    "sub_11",
    "sub_12",
    "sub_13",
    "sub_14",
    "sub_15",
    "sub_17",
    "sub_18",
]

transfer_learning_subject = "sub_02"

subs.remove(transfer_learning_subject)
# Load data
recordings = data_config.load_dataset(subs=subs, features=features)
recordings_tl = data_config.load_dataset(
    subs=[transfer_learning_subject], features=features
)

for lopo_sub in subs:
    experiment_folder_path_lopo = os.path.join(experiment_folder_path, lopo_sub)
    os.makedirs(experiment_folder_path_lopo, exist_ok=True)
    # Test Train Split
    recordings_train, recordings_test = recordings.split_leave_subject_out(
        test_subject=lopo_sub
    )
    # Windowize
    windows_train, windows_test = recordings_train.windowize(
        window_size
    ), recordings_test.windowize(window_size)

    # Convert
    X_train, y_train = DataSet.convert_windows_sonar(
        windows_train, data_config.n_activities()
    )
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = DataSet.convert_windows_sonar(
        windows_test, data_config.n_activities()
    )

    # build model
    base_model = GaitAnalysisTLModel(
        window_size=window_size,
        n_features=len(features),
        n_outputs=n_classes,
        n_epochs=10,
        learning_rate=0.0001,
        batch_size=32,
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description="GaitAnalysis transfer learning model for gait dataset",
    )

    base_model.model_name += "_lopo" + lopo_sub

    print("MEan", data_config.mean)
    print("Variance", data_config.variance)

    base_model.fit(X_train, y_train)

    y_test_pred = base_model.predict(X_test)

    create_conf_matrix(
        experiment_folder_path_lopo,
        y_test_pred,
        y_test,
        file_name="base_model_conf_matrix_lopo",
    )
    create_text_metrics(
        experiment_folder_path_lopo,
        y_test_pred,
        y_test,
        [accuracy],
        file_name="base_model_metrics_lopo",
    )  # TODO: at the moment only with one function working! data gets changed (passed by reference) - refactor metric functions

    # Test Train Split of tl data set
    recordings_tl_train, recordings_tl_test = recordings_tl.split_by_percentage(0.2)
    # Windowize
    windows_tl_train, windows_tl_test = recordings_tl_train.windowize(
        window_size
    ), recordings_tl_test.windowize(window_size)

    # Convert
    X_tl_train, y_tl_train = DataSet.convert_windows_sonar(
        windows_tl_train, data_config.n_activities()
    )
    X_tl_train, y_tl_train = shuffle(X_tl_train, y_tl_train, random_state=42)
    X_tl_test, y_tl_test = DataSet.convert_windows_sonar(
        windows_tl_test, data_config.n_activities()
    )

    # save weights of base model
    base_model.save_weights(
        os.path.join(experiment_folder_path_lopo, "base_model_weights.h5")
    )

    # create tl_model and load weights
    tl_model = GaitAnalysisTLModel(
        window_size=window_size,
        n_features=len(features),
        n_outputs=n_classes,
        n_epochs=10,
        learning_rate=0.0001,
        batch_size=32,
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance,
        author="Felix Treykorn",
        version="0.1",
        description="GaitAnalysis transfer learning model for gait dataset",
    )
    tl_model.load_weights(
        os.path.join(experiment_folder_path_lopo, "base_model_weights.h5")
    )
    tl_model.model_name += "_lopo" + lopo_sub + "_tl" + transfer_learning_subject

    # freeze inner layers of tl model
    tl_model.inner_model.trainable = False

    # train inner layers of tl model
    tl_model.fit(X_tl_train, y_tl_train)

    y_test_pred_tl_model = tl_model.predict(X_test)
    y_tl_test_pred_tl_model = tl_model.predict(X_tl_test)
    y_tl_test_pred_base_model = base_model.predict(X_tl_test)

    # create confusion matrix for tl model and base model
    create_conf_matrix(
        experiment_folder_path_lopo,
        y_tl_test_pred_tl_model,
        y_tl_test,
        file_name="tl_model_conf_matrix_tl",
    )
    create_conf_matrix(
        experiment_folder_path_lopo,
        y_test_pred_tl_model,
        y_test,
        file_name="tl_model_conf_matrix_lopo",
    )
    create_conf_matrix(
        experiment_folder_path_lopo,
        y_tl_test_pred_base_model,
        y_tl_test,
        file_name="base_model_conf_matrix_tl",
    )
    # create text metrics for tl model and base model
    create_text_metrics(
        experiment_folder_path_lopo,
        y_tl_test_pred_tl_model,
        y_tl_test,
        [accuracy],
        file_name="tl_model_metrics_tl",
    )
    create_text_metrics(
        experiment_folder_path_lopo,
        y_tl_test_pred_base_model,
        y_tl_test,
        [accuracy],
        file_name="base_model_metrics_tl",
    )
    create_text_metrics(
        experiment_folder_path_lopo,
        y_test_pred_tl_model,
        y_test,
        [accuracy],
        file_name="tl_model_metrics_lopo",
    )

    # export base model and tl model
    base_model.export(
        experiment_folder_path_lopo,
        features=features,
        device_tags=data_config.sensor_suffix_order,
        class_labels=list(data_config.category_labels.keys()),
    )
    print(f"Saved base model to {experiment_folder_path}")

    tl_model.export(
        experiment_folder_path_lopo,
        features=features,
        device_tags=data_config.sensor_suffix_order,
        class_labels=list(data_config.category_labels.keys()),
    )
    print(f"Saved transfer learning model to {experiment_folder_path}")
