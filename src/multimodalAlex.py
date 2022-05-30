from fileinput import filename
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from utils import settings as settings
import pandas as pd
from evaluation.metrics import accuracy
from evaluation.conf_matrix import create_conf_matrix
from loader.Preprocessor import Preprocessor
from models.JensModel import JensModel
from models.RainbowModel import RainbowModel
from models.ResNetModel import ResNetModel
from models.ResNetModel_Multimodal import ResNetModelMultimodal
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder
from utils.DataConfig import Sonar22CategoriesConfig, OpportunityConfig, SonarConfig, LabPoseConfig
from tensorflow.keras.layers import (Dense)
from utils.Recording import Recording
import matplotlib.pyplot as plt
import utils.DataConfig
from pose_sequence_loader import *
from multiprocessing import Pool

""" 
Number of recordings per person
{'connie.csv': 6, 'alex.csv': 38, 'trapp.csv': 9, 'anja.csv': 13, 'aileen.csv': 52, 'florian.csv': 16, 'brueggemann.csv': 36, 'oli.csv': 20, 'rauche.csv': 9, 'b2.csv': 6, 'yvan.csv': 8, 'christine.csv': 7, 'mathias.csv': 2, 'kathi.csv': 17}
"""

# DEFINE MACROS
data_config = LabPoseConfig(dataset_path='/dhc/groups/bp2021ba1/data/lab_data_filtered_without_null')#OpportunityConfig(dataset_path='/dhc/groups/bp2021ba1/data/opportunity-dataset')
#data_config = SonarConfig(dataset_path='/dhc/groups/bp2021ba1/data/lab_data')#OpportunityConfig(dataset_path='/dhc/groups/bp2021ba1/data/opportunity-dataset')
settings.init(data_config)
random.seed(1678978086101)

k_fold_splits = 3
numEpochs = 10

# LOAD DATA
recordings = settings.DATA_CONFIG.load_dataset(limit=40)
print("==> LOADING OF RECORDINGS DONE")

# APPEND POSE FRAMES 
def process_recording(recording_with_index):
    i = recording_with_index[0]
    recording = recording_with_index[1]
    pose_frame = get_poseframe(recording, "/dhc/groups/bp2021ba1/data/lab_data")
    
    print(f"Pose Frame added to Recording {i+1}/{len(recordings)}  ")
    return pose_frame

# pose_frames = []
# for i, k in list(enumerate(recordings)):
#     pose_frames.append(process_recording((i,k)))

pool = Pool()
pose_frames = pool.map(process_recording, list(enumerate(recordings)), 1)
pool.close()
pool.join()

for i, pose_frame in enumerate(pose_frames):
    recordings[i].pose_frame = pose_frame

initialLength = len(recordings)
recordings = list(filter(
    lambda recording: not recording.pose_frame.empty, 
    recordings
))
print(f"Filtered out {initialLength - len(recordings)} Recordings (!)")

print("==> APPENDING POSE FRAMES DONE")


# CONFIG TRAINING
window_size = 100
n_sensor_features = recordings[0].sensor_frame.shape[1]
n_pose_features = recordings[0].pose_frame.shape[1]
print(f"Sensor features: {n_sensor_features},  Pose features: {n_pose_features}")
n_outputs = settings.DATA_CONFIG.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "multimodalAlex"
)


def evaluate(y_test_pred: np.ndarray, y_test_true: np.ndarray, confusionMatrixFileName=None, confusionMatrixTitle="") -> tuple[float, float,float, np.ndarray]:
    acc = accuracy(y_test_pred, y_test_true)
    if confusionMatrixFileName:
        create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name = confusionMatrixFileName, title=confusionMatrixTitle+", acc:"+str(int(acc*10000)/100)+"%") 
    f1_macro = f1_score(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1), average="macro")   
    f1_weighted = f1_score(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1), average="weighted")    
    return acc, f1_macro, f1_weighted, y_test_true

def instanciateModel(use_sensor_frame = True, use_pose_frame = False) -> ResNetModelMultimodal:
    n_features = 0
    n_features += n_sensor_features if (use_sensor_frame) else 0
    n_features += n_pose_features if (use_pose_frame) else 0

    return ResNetModelMultimodal(
        n_epochs=numEpochs,
        window_size=100,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
        use_sensor_frame=use_sensor_frame,
        use_pose_frame=use_pose_frame
    )


# TRAIN AND PREDICT
multiModals = [(True, False), (True, True), (False, True)]
for modality_index, (use_sensor_frame, use_pose_frame) in enumerate(multiModals):
    model = instanciateModel(use_sensor_frame=use_sensor_frame, use_pose_frame=use_pose_frame)
    model.n_epochs = numEpochs

    #k_fold = StratifiedKFold(n_splits=k_fold_splits, random_state=None)
    split_index = int(0.8 * len(recordings))
    recordingsTrain = recordings[:split_index]
    recordingsTest = recordings[split_index+1:]

    x_train, y_train = model.windowize_convert_fit(recordingsTrain)

    x_test, y_test = model.windowize_convert(recordingsTest)
    y_test_pred = model.predict(x_test)
    acc, f1_macro, f1_weighted, _ = evaluate(y_test_pred, y_test)

    print(f"==== Sensor Features: {use_sensor_frame}   Pose Features: {use_pose_frame} ====")
    print(f"Accuracy: {acc}")
    print(f"F1 Macro: {f1_macro}")
    print(f"F1 Weighted: {f1_weighted}")

