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
#data_config = LabPoseConfig(dataset_path='/dhc/groups/bp2021ba1/data/lab_data')#OpportunityConfig(dataset_path='/dhc/groups/bp2021ba1/data/opportunity-dataset')
data_config = SonarConfig(dataset_path='/dhc/groups/bp2021ba1/data/lab_data')#OpportunityConfig(dataset_path='/dhc/groups/bp2021ba1/data/opportunity-dataset')
settings.init(data_config)
random.seed(1678978086101)

k_fold_splits = 3
numEpochsBeforeTL = 10
numEpochsForTL = 3
minimumRecordingsPerLeftOutPerson = 5

# LOAD DATA
recordings = settings.DATA_CONFIG.load_dataset()#limit=75

# APPEND POSE FRAMES 
def process_recording(recording_with_index):
    i = recording_with_index[0]
    recording = recording_with_index[1]
    pose_frame = get_poseframe(recording)
    
    print(f"Pose Frame added to Recording {i+1}/{len(recordings)}  ")
    #print(recording.recording_folder, recording.pose_frame)
    return pose_frame

pool = Pool()
pose_frames = pool.map(process_recording, list(enumerate(recordings)), 1)
pool.close()
pool.join()

for i, pose_frame in enumerate(pose_frames):
    recordings[i].pose_frame = pose_frame

print("==> APPENDING POSE FRAMES DONE")


# DEFINE TRAINING
def split_list_by_people(recordings: "list[Recording]", peopleForListA: "list[str]") -> tuple[np.ndarray, np.ndarray]:
    """ Splits the recordings into a tuple of a sublist of recordings of people in peopleForListA and the recordings of other people"""
    return np.array(list(filter(lambda recording: recording.subject in peopleForListA,recordings))), np.array(list(filter(lambda recording: recording.subject not in peopleForListA, recordings)))
window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
print(n_features)
n_outputs = settings.DATA_CONFIG.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "transferLearningTobi"
)
