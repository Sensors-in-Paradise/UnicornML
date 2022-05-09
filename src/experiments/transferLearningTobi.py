"""
test with new config

"""

from fileinput import filename
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
import matplotlib.pyplot as plt
# Init
# TODO: refactor, find out why confusion matrix sometimes shows different results than accuracy
""" 
Number of recordings per person
{'connie.csv': 6, 'alex.csv': 38, 'trapp.csv': 9, 'anja.csv': 13, 'aileen.csv': 52, 'florian.csv': 16, 'brueggemann.csv': 36, 'oli.csv': 20, 'rauche.csv': 9, 'b2.csv': 6, 'yvan.csv': 8, 'christine.csv': 7, 'mathias.csv': 2, 'kathi.csv': 17}
"""
data_config = Sonar22CategoriesConfig(dataset_path='/dhc/groups/bp2021ba1/data/filtered_dataset_without_null')
settings.init(data_config)
random.seed(1678978086101)

# Load dataset
recordings = settings.DATA_CONFIG.load_dataset()#limit=75

# Preprocess
recordings = Preprocessor().our_preprocess(recordings)

def indexFromCategorical(categoricalY):
    index = -1
    for index, value in enumerate(categoricalY):
        if value == 1.0:
            return index
    return index
# TODO: split recordings by person , e.g. leave one person out 
# (repeat e.g. 5 times with different people, then split recordings of remaining person into transfer-learning-train and transfer-learning-test data
# --> test accuracy on test data of remaining person and compare with accuracy after doing transfer learning on transfer-learning-train data)
def split_list_by_people(recordings: "list[Recording]", peopleForListA: "list[str]"):
    """ Splits the recordings into a tuple of a sublist of recordings of people in peopleForListA and the recordings of other people"""
    return np.array(list(filter(lambda recording: recording.subject in peopleForListA,recordings))), np.array(list(filter(lambda recording: recording.subject not in peopleForListA, recordings)))
window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
n_outputs = settings.DATA_CONFIG.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "transferLearningTobi"
)

def save_pie_chart_from_dict(labelsAndFrequencyDict: dict, fileName: str):
    plt.cla()
    plt.clf()
    plt.pie([labelsAndFrequencyDict[label] for label in labelsAndFrequencyDict.keys()], labels=[label+ " "+ str(int(labelsAndFrequencyDict[label]/60))+"secs" for label in labelsAndFrequencyDict.keys()])
    plt.savefig(os.path.join(experiment_folder_path, fileName))

def getActivityCountsFromRecordings(recordings: "list[Recording]"):
    resultDict = {}
    for recording in recordings:
        counts = recording.activities.value_counts()
        for activity_id, count in counts.items():
            if activity_id in resultDict:
                resultDict[activity_id] += count
            else:
                resultDict[activity_id] = count 
    return resultDict

def save_activity_distribution_pie_chart(yTrue, fileName: str):
    unique, counts = np.unique([indexFromCategorical(yOneHot) for yOneHot in yTrue], return_counts=True)
    countsDict = dict(zip([settings.DATA_CONFIG.activity_idx_to_activity_name_map[item] for item in unique], counts))
    print(f"-------------------------------------------------------------\ncounts for file {fileName}",countsDict,unique,"\n---------------------------------------------------------")
    save_pie_chart_from_dict(countsDict, fileName)

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


def getAccuracyOnRecordings(model: "JensModel",_recordings: "list[Recording]", confusionMatrixFileName=None):
    X_test, y_test_true = model.windowize_convert(_recordings)
    y_test_pred = model.predict(X_test)
    if confusionMatrixFileName:
        create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name = confusionMatrixFileName) 
    return accuracy(y_test_pred, y_test_true), y_test_true

def instanciateModel():
    return JensModel(
        n_epochs=2,
        window_size=100,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
    )


people = get_people_in_recordings(recordings)
peopleToLeaveOutPerExpirement = ["anja.csv", "florian.csv", "oli.csv", "rauche.csv"]#, "oli.csv", "rauche.csv"

k_fold_splits = 3
result = [["fold id \ Left out person"]+[["FOLD "+str(round(i/3)), "without TL", "with TL"][i%3] for i in range(k_fold_splits*3)]+["Average without TL"]+["Average with TL"]]

for personToLeaveOut in peopleToLeaveOutPerExpirement:
    print(f"Leaving person {personToLeaveOut} out")
    model = instanciateModel()
    recordingsOfLeftOutPerson, recordingsTrain = split_list_by_people(recordings, [personToLeaveOut])
    _, yTrainTrue = model.windowize_convert_fit(recordingsTrain)
    print(f"ACTIVITY COUNTS OF TRAINING DATA WITH {personToLeaveOut} LEFT OUT: ",getActivityCountsFromRecordings(recordingsTrain))
    print(f"ACTIVITY COUNTS OF TRAINING DATA WITH {personToLeaveOut} LEFT OUT according to yTrue: ",np.unique(yTrainTrue))
    activityDistributionFileName = f"{personToLeaveOut}_trainActivityDistribution.png"
    save_activity_distribution_pie_chart(yTrainTrue,  activityDistributionFileName)

    resultCol = [personToLeaveOut  +f"<br />Train activity distribution <br />![Base model train activity distribution]({activityDistributionFileName})"]
    resultWithoutTLVals = []
    
    resultWithTLVals = []

    model.model.save_weights("ckpt")
    # Evaluate on left out person
    k_fold = KFold(n_splits=k_fold_splits, random_state=None)
    for (index, (train_index, test_index)) in enumerate(k_fold.split(recordingsOfLeftOutPerson)):
        model.model.load_weights("ckpt")
        recordingsOfLeftOutPerson_train =  recordingsOfLeftOutPerson[train_index]
        recordingsOfLeftOutPerson_test = recordingsOfLeftOutPerson[test_index]
        print(f"ACTIVITY COUNTS OF TRANSFER LEARNING TRAINING DATA WITH {personToLeaveOut} LEFT OUT IN FOLD {index}: ",getActivityCountsFromRecordings(recordingsOfLeftOutPerson_train))
        confMatrixWithoutTLFileName = f"{personToLeaveOut}_kfold{index}_withoutTL_conf_matrix"
        # Evaluate without transfer learning
        accuracyWithoutTransferLearning, yTestTrue = getAccuracyOnRecordings(model, recordingsOfLeftOutPerson_test, confMatrixWithoutTLFileName)
        activityDistributionTestFileName = f"{personToLeaveOut}_kfold{index}_testActivityDistribution.png"
        save_activity_distribution_pie_chart(yTestTrue,  activityDistributionTestFileName)
        
        print(f"Accuracy on test data of left out person {accuracyWithoutTransferLearning}")

        # Set non dense layers to not trainable (freezing them)
        for index, layer in enumerate(model.model.layers):
            layer.trainable = type(layer) == Dense
            print(f"Layer {index} trainable: {layer.trainable}")

        # Do transfer learning
        _, yTrainTrue = model.windowize_convert_fit(recordingsOfLeftOutPerson_train)
        tlActivityDistributionFileName =  f"{personToLeaveOut}_kfold{index}_TL_trainActivityDistribution.png"
        save_activity_distribution_pie_chart(yTrainTrue,  tlActivityDistributionFileName)

        confMatrixWithTLFileName = f"{personToLeaveOut}_kfold{index}_withTL_conf_matrix"
        accuracyWithTransferLearning,_ = getAccuracyOnRecordings(model, recordingsOfLeftOutPerson_test, confMatrixWithTLFileName)
        print(f"Accuracy on test data of left out person {accuracyWithTransferLearning}")
        resultCol.append("Test Activity Distribution "+f"<br />![Test activity distribution]({activityDistributionTestFileName})")
        resultCol.append("Accuracy: "+str(accuracyWithoutTransferLearning)+f"<br />![confusion matrix]({confMatrixWithoutTLFileName}.png)")
        resultCol.append("Accuracy: "+str(accuracyWithTransferLearning)+f"<br />TL-Train activity distribution <br />![TL-Train activity distribution]({tlActivityDistributionFileName})"+f"<br />![confusion matrix]({confMatrixWithTLFileName}.png)")
        resultWithoutTLVals.append(accuracyWithoutTransferLearning)
        resultWithTLVals.append(accuracyWithTransferLearning)
    resultCol.append(np.average(resultWithoutTLVals))
    resultCol.append(np.average(resultWithTLVals))
    
    result = result + [resultCol]

print("result",result)




resultT = np.array(result).T
print("resultT",resultT)
# save a simple test report to the experiment folder
wholeDataSetActivityDistributionFileName = "wholeDatasetActivityDistribution.png"
_, yAll = instanciateModel().windowize_convert(recordings)
save_activity_distribution_pie_chart(yAll,  wholeDataSetActivityDistributionFileName)
result_md = f"# Whole dataset distribution\n![activityDistribution]({wholeDataSetActivityDistributionFileName})"
result_md += "\n# Experiments\n"
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




