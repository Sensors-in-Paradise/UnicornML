"""
test with new config

"""

import os
import random

import numpy as np
from sklearn.model_selection import KFold

import utils.settings as settings
from evaluation.metrics import accuracy
from evaluation.conf_matrix import create_conf_matrix
from loader.Preprocessor import Preprocessor
from models.JensModel import JensModel
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder
from utils.DataConfig import Sonar22CategoriesConfig
from utils.save_all_recordings import save_all_recordings, load_all_recordings
from utils.array_operations import split_list_by_percentage
from tensorflow.keras.layers import (Dense)
# Init
""" 
Number of recordings per person
{'connie.csv': 6, 'alex.csv': 38, 'trapp.csv': 9, 'anja.csv': 13, 'aileen.csv': 52, 'florian.csv': 16, 'brueggemann.csv': 36, 'oli.csv': 20, 'rauche.csv': 9, 'b2.csv': 6, 'yvan.csv': 8, 'christine.csv': 7, 'mathias.csv': 2, 'kathi.csv': 17}
"""
data_config = Sonar22CategoriesConfig(dataset_path='/dhc/groups/bp2021ba1/data/filtered_dataset_without_null')
settings.init(data_config)
random.seed(1678978086101)

# Load dataset
recordings = settings.DATA_CONFIG.load_dataset()

# Preprocess
recordings = Preprocessor().our_preprocess(recordings)

# TODO: split recordings by person , e.g. leave one person out 
# (repeat e.g. 5 times with different people, then split recordings of remaining person into transfer-learning-train and transfer-learning-test data
# --> test accuracy on test data of remaining person and compare with accuracy after doing transfer learning on transfer-learning-train data)
def split_list_by_people(recordings: "list[Recording]", peopleForListA: "list[str]"):
    """ Splits the recordings into a tuple of a sublist of recordings of people in peopleForListA and the recordings of other people"""
    return np.array(list(filter(lambda recording: recording.subject in peopleForListA,recordings))), np.array(list(filter(lambda recording: recording.subject not in peopleForListA, recordings)))
window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
n_outputs = settings.DATA_CONFIG.n_activities()

def count_recordings_of_people(recordings: "list[Recording]"):
    peopleCount = {}
    for recording in recordings:
        if recording.subject in peopleCount:
            peopleCount[recording.subject] += 1
        else:
            peopleCount[recording.subject] = 1
    return peopleCount

def get_people_in_recordings(recordings: "list[Recording]"):
    people = set()
    for recording in recordings:
        people.add(recording.subject)
    list(people)

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "transferLearningTobi"
)

def getAccuracyOnRecordings(model: "JensModel",recordings: "list[Recording]", confusionMatrixFileName=None):
    X_test, y_test_true = model.windowize_convert(recordings)
    y_test_pred = model.predict(X_test)
    if confusionMatrixFileName:
        create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name = confusionMatrixFileName) 
    return accuracy(y_test_pred, y_test_true)

people = get_people_in_recordings(recordings)
peopleToLeaveOutPerExpirement = ["anja.csv", "florian.csv"]#, "oli.csv", "rauche.csv"

k_fold_splits = 2
result = [["Left out person"]+[i for i in range(k_fold_splits)]+["Average"]]

for personToLeaveOut in peopleToLeaveOutPerExpirement:
    print(f"Leaving person {personToLeaveOut} out")
    model = JensModel(
        n_epochs=2,
        window_size=100,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
    )
    recordingsOfLeftOutPerson, recordingsTrain = split_list_by_people(recordings, [personToLeaveOut])
    model.windowize_convert_fit(recordingsTrain)
    

    resultWithoutTLCol = [personToLeaveOut + " without TL"]
    resultWithoutTLVals = []
    resultWithTLCol = [personToLeaveOut + " with TL"]
    resultWithTLVals = []

    model.model.save_weights("ckpt")
    # Evaluate on left out person
    k_fold = KFold(n_splits=k_fold_splits, random_state=None)
    for (index, (train_index, test_index)) in enumerate(k_fold.split(recordingsOfLeftOutPerson)):
        model.model.load_weights("ckpt")
        recordingsOfLeftOutPerson_train =  recordingsOfLeftOutPerson[train_index]
        recordingsOfLeftOutPerson_test = recordingsOfLeftOutPerson[test_index]

        confMatrixWithoutTLFileName = f"{personToLeaveOut}_kfold{index}_withoutTL_conf_matrix.png"
        # Evaluate without transfer learning
        accuracyWithoutTransferLearning = getAccuracyOnRecordings(model, recordingsOfLeftOutPerson_test, confMatrixWithoutTLFileName)
        print(f"Accuracy on test data of left out person {accuracyWithoutTransferLearning}")

        # Set non dense layers to not trainable (freezing them)
        for index, layer in enumerate(model.model.layers):
            layer.trainable = type(layer) == Dense
            print(f"Layer {index} trainable: {layer.trainable}")

        # Do transfer learning
        model.windowize_convert_fit(recordingsOfLeftOutPerson_train)
        confMatrixWithTLFileName = f"{personToLeaveOut}_kfold{index}_withTL_conf_matrix.png"
        accuracyWithTransferLearning = getAccuracyOnRecordings(model, recordingsOfLeftOutPerson_test,confMatrixWithTLFileName)
        print(f"Accuracy on test data of left out person {accuracyWithTransferLearning}")
        resultWithoutTLCol.append(str(accuracyWithoutTransferLearning)+f" ![confusion matrix]({confMatrixWithoutTLFileName})")
        resultWithTLCol.append(str(accuracyWithTransferLearning) +f" ![confusion matrix]({confMatrixWithTLFileName})")
        resultWithoutTLVals.append(accuracyWithoutTransferLearning)
        resultWithTLVals.append(accuracyWithTransferLearning)
    resultWithoutTLCol.append(np.average(resultWithoutTLVals))
    resultWithTLCol.append(np.average(resultWithTLVals))
    result = result + [resultWithoutTLCol] + [resultWithTLCol]

print("result",result)




resultT = np.array(result).T
print("resultT",resultT)
# save a simple test report to the experiment folder
result_md = ""
for index, row in enumerate(resultT):
    for item in row:
        result_md += "|" + str(item)
    result_md += "|\n"
    if index == 0:
        for col in range(len(row)):
            result_md += "| -----------"
        result_md += "|\n"


with open(os.path.join(experiment_folder_path, "results.md"), "w+") as f:
    f.writelines(result_md)




