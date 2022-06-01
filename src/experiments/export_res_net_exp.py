"""
test with new config

"""

from fileinput import filename
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from models.RainbowModel import RainbowModel
import pandas as pd
import utils.settings as settings
from evaluation.metrics import accuracy
from evaluation.conf_matrix import create_conf_matrix
from models.JensModel import JensModel
from models.RainbowModel import RainbowModel
from models.ResNetModel import ResNetModel
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder
from utils.DataConfig import Sonar22CategoriesConfig, OpportunityConfig
from utils.Recording import Recording
import matplotlib.pyplot as plt

# Init
# TODO: refactor, find out why confusion matrix sometimes shows different results than accuracy
# TODO: - make train and test datasets evenly distributed
#       - make
""" 
Number of recordings per person
{'connie.csv': 6, 'alex.csv': 38, 'trapp.csv': 9, 'anja.csv': 13, 'aileen.csv': 52, 'florian.csv': 16, 'brueggemann.csv': 36, 'oli.csv': 20, 'rauche.csv': 9, 'b2.csv': 6, 'yvan.csv': 8, 'christine.csv': 7, 'mathias.csv': 2, 'kathi.csv': 17}
"""

# Sonar22CategoriesConfig(dataset_path='/dhc/groups/bp2021ba1/data/filtered_dataset_without_null')#
data_config = OpportunityConfig(
    dataset_path='../../data/opportunity-dataset')
settings.init(data_config)

random.seed(1678978086101)
# Load dataset
recordings = data_config.load_dataset()
window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
n_outputs = data_config.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "transferLearningTobi"
)


def count_activities_per_person(recordings: "list[Recording]"):
    values = pd.DataFrame(
        {recordings[0].subject: recordings[0].activities.value_counts()})
    for rec in recordings[1:]:
        values = values.add(pd.DataFrame(
            {rec.subject: rec.activities.value_counts()}), fill_value=0)

    return values


def plot_activities_per_person(recordings: "list[Recording]", fileName: str, title: str = ""):
    values = count_activities_per_person(recordings)

    values.plot.bar(figsize=(22, 16))
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(os.path.join(experiment_folder_path, fileName))


def save_pie_chart_from_dict(labelsAndFrequencyDict: dict, fileName: str, title: str = None, subtitle: str = None) -> None:
    plt.cla()
    plt.clf()
    print(labelsAndFrequencyDict)
    print(settings.DATA_CONFIG.raw_label_to_activity_idx_map)
    data = [labelsAndFrequencyDict[label]
            for label in labelsAndFrequencyDict.keys()]
    labels = [
        f"{settings.DATA_CONFIG.raw_label_to_activity_idx_map[label]} {label} {int(labelsAndFrequencyDict[label]/60)} secs" for label in labelsAndFrequencyDict.keys()]
    plt.pie(data, labels=labels)
    if title:
        plt.suptitle(title, y=1.05, fontsize=18)
    if subtitle:
        plt.title(subtitle, fontsize=10)

    plt.savefig(os.path.join(experiment_folder_path, fileName))


def getActivityCountsFromRecordings(recordings: "list[Recording]") -> dict:
    resultDict = {}
    for recording in recordings:
        counts = recording.activities.value_counts()
        for activity_id, count in counts.items():
            if activity_id in resultDict:
                resultDict[activity_id] += count
            else:
                resultDict[activity_id] = count
    for activity in settings.DATA_CONFIG.raw_label_to_activity_idx_map:
        if not activity in resultDict:
            resultDict[activity] = 0
    return resultDict


def getActivityCounts(yTrue):
    unique, counts = np.unique(np.argmax(yTrue, axis=1), return_counts=True)
    countsDict = dict(zip(
        [settings.DATA_CONFIG.activity_idx_to_activity_name_map[item] for item in unique], counts))
    return countsDict


def save_activity_distribution_pie_chart(yTrue, fileName: str, title: str = None, subtitle: str = None) -> None:
    unique, counts = np.unique(np.argmax(yTrue, axis=1), return_counts=True)
    countsDict = dict(zip(
        [settings.DATA_CONFIG.activity_idx_to_activity_name_map[item] for item in unique], counts))
    subtitleSuffix = f"Mean activity duration difference from mean activity duration: {int(getMeanCountDifferenceFromMeanActivityCount(yTrue)/60)}s"
    if subtitle != None:
        subtitle += "\n"+subtitleSuffix
    else:
        subtitle = subtitleSuffix
    save_pie_chart_from_dict(countsDict, fileName, title, subtitle)


def count_recordings_of_people(recordings: "list[Recording]") -> dict:
    peopleCount = {}
    for recording in recordings:
        if recording.subject in peopleCount:
            peopleCount[recording.subject] += 1
        else:
            peopleCount[recording.subject] = 1
    return peopleCount


def evaluate(model: "RainbowModel", X_test: np.ndarray, y_test_true: np.ndarray, confusionMatrixFileName=None, confusionMatrixTitle="") -> "tuple[float, float, float, np.ndarray]":
    y_test_pred = model.predict(X_test)
    acc = accuracy(y_test_pred, y_test_true)
    if confusionMatrixFileName:
        create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name=confusionMatrixFileName,
                           title=confusionMatrixTitle+", acc:"+str(int(acc*10000)/100)+"%")
    f1_macro = f1_score(np.argmax(y_test_true, axis=1),
                        np.argmax(y_test_pred, axis=1), average="macro")
    f1_weighted = f1_score(np.argmax(y_test_true, axis=1), np.argmax(
        y_test_pred, axis=1), average="weighted")
    return acc, f1_macro, f1_weighted, y_test_true


def instanciateModel():
    return ResNetModel(
        n_epochs=5,
        window_size=100,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
        input_distribution_mean=data_config.mean,
        input_distribution_variance=data_config.variance
    )


model = instanciateModel()
xTrain, yTrainTrue = model.windowize_convert_fit(recordings)

model.export(os.path.join(experiment_folder_path, "model"))
