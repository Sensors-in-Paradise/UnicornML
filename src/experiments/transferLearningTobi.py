"""
test with new config

"""

from fileinput import filename
import os
import random

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from models.RainbowModel import RainbowModel

import utils.settings as settings
from evaluation.metrics import accuracy
from evaluation.conf_matrix import create_conf_matrix
from loader.Preprocessor import Preprocessor
from models.JensModel import JensModel
from utils.filter_activities import filter_activities
from utils.folder_operations import new_saved_experiment_folder
from utils.DataConfig import Sonar22CategoriesConfig
from tensorflow.keras.layers import (Dense)
from utils.Recording import Recording
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

def split_list_by_people(recordings: "list[Recording]", peopleForListA: "list[str]") -> tuple[np.ndarray, np.ndarray]:
    """ Splits the recordings into a tuple of a sublist of recordings of people in peopleForListA and the recordings of other people"""
    return np.array(list(filter(lambda recording: recording.subject in peopleForListA,recordings))), np.array(list(filter(lambda recording: recording.subject not in peopleForListA, recordings)))
window_size = 100
n_features = recordings[0].sensor_frame.shape[1]
n_outputs = settings.DATA_CONFIG.n_activities()

# Create Folder, save model export and evaluations there
experiment_folder_path = new_saved_experiment_folder(
    "transferLearningTobi"
)

def save_pie_chart_from_dict(labelsAndFrequencyDict: dict, fileName: str, title: str = None, subtitle: str = None)-> None:
    plt.cla()
    plt.clf()
    plt.pie([labelsAndFrequencyDict[label] for label in labelsAndFrequencyDict.keys()], labels=[f"{settings.DATA_CONFIG.raw_label_to_activity_idx_map[label]} {label} {int(labelsAndFrequencyDict[label]/60)} secs" for label in labelsAndFrequencyDict.keys()])
    if title:
        plt.suptitle(title, y=1.05, fontsize=18)
    if subtitle:
        plt.title(subtitle, fontsize=10)

    plt.savefig(os.path.join(experiment_folder_path, fileName))

def getActivityCountsFromRecordings(recordings: "list[Recording]")-> dict:
    resultDict = {}
    for recording in recordings:
        counts = recording.activities.value_counts()
        for activity_id, count in counts.items():
            if activity_id in resultDict:
                resultDict[activity_id] += count
            else:
                resultDict[activity_id] = count 
    return resultDict

def save_activity_distribution_pie_chart(yTrue, fileName: str, title: str=None, subtitle: str = None) -> None:
    unique, counts = np.unique(np.argmax(yTrue, axis=1), return_counts=True)
    countsDict = dict(zip([settings.DATA_CONFIG.activity_idx_to_activity_name_map[item] for item in unique], counts))
    save_pie_chart_from_dict(countsDict, fileName, title, subtitle)

def count_recordings_of_people(recordings: "list[Recording]")-> dict:
    peopleCount = {}
    for recording in recordings:
        if recording.subject in peopleCount:
            peopleCount[recording.subject] += 1
        else:
            peopleCount[recording.subject] = 1
    return peopleCount

def get_people_in_recordings(recordings: "list[Recording]") -> list[str]:
    people = set()
    for recording in recordings:
        people.add(recording.subject)
    return list(people)


def evaluateOnRecordings(model: "JensModel",_recordings: "list[Recording]", confusionMatrixFileName=None, confusionMatrixTitle="") -> tuple[float, float,float, np.ndarray]:
    X_test, y_test_true = model.windowize_convert(_recordings)
    y_test_pred = model.predict(X_test)
    acc = accuracy(y_test_pred, y_test_true)
    if confusionMatrixFileName:
        create_conf_matrix(experiment_folder_path, y_test_pred, y_test_true, file_name = confusionMatrixFileName, title=confusionMatrixTitle+", acc:"+str(int(acc*10000)/100)+"%") 
    f1_macro = f1_score(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1), average="macro")   
    f1_weighted = f1_score(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1), average="weighted")    
    return acc, f1_macro, f1_weighted, y_test_true

def instanciateModel():
    return JensModel(
        n_epochs=2,
        window_size=100,
        n_features=n_features,
        n_outputs=n_outputs,
        batch_size=64,
    )
def freezeDenseLayers(model: RainbowModel):
    # Set non dense layers to not trainable (freezing them)
    for layer in model.model.layers:
        layer.trainable = type(layer) == Dense

people = get_people_in_recordings(recordings)
numRecordingsOfPeopleDict = count_recordings_of_people(recordings)
peopleToLeaveOutPerExpirement = list(filter(lambda person: numRecordingsOfPeopleDict[person]>5,people)) #["anja.csv", "florian.csv", "oli.csv", "rauche.csv"]#, "oli.csv", "rauche.csv"

k_fold_splits = 3
result = [["fold id \ Left out person"]+[["**FOLD "+str(round(i/3))+"**", "without TL", "with TL"][i%3] for i in range(k_fold_splits*3)]+["Average without TL"]+["Average with TL"]]

for personIndex, personToLeaveOut in enumerate(peopleToLeaveOutPerExpirement):
    print(f"==============================================================================\nLeaving person {personToLeaveOut} out {personIndex}/{len(peopleToLeaveOutPerExpirement)}\n==============================================================================\n")
    personId = people.index(personToLeaveOut)
    model = instanciateModel()
    recordingsOfLeftOutPerson, recordingsTrain = split_list_by_people(recordings, [personToLeaveOut])
    _, yTrainTrue = model.windowize_convert_fit(recordingsTrain)
    activityDistributionFileName = f"subject{personId}_trainActivityDistribution.png"
    save_activity_distribution_pie_chart(yTrainTrue,  activityDistributionFileName)

    resultCol = [f"Subject {personId}<br />Train activity distribution <br />![Base model train activity distribution]({activityDistributionFileName})"]
    resultWithoutTLVals = []
    resultWithTLVals = []

    model.model.save_weights("ckpt")
    # Evaluate on left out person
    k_fold = KFold(n_splits=k_fold_splits, random_state=None)
    for (index, (train_index, test_index)) in enumerate(k_fold.split(recordingsOfLeftOutPerson)):
        # Restore start model state for all folds of this left out person 
        model.model.load_weights("ckpt")

        # Grab data for this fold
        recordingsOfLeftOutPerson_train =  recordingsOfLeftOutPerson[train_index]
        recordingsOfLeftOutPerson_test = recordingsOfLeftOutPerson[test_index]

        # Evaluate without transfer learning
        confMatrixWithoutTLFileName = f"subject{personId}_kfold{index}_withoutTL_conf_matrix"
        accuracyWithoutTransferLearning, f1ScoreMacroWithoutTransferLearning,f1ScoreWeightedWithoutTransferLearning, yTestTrue = evaluateOnRecordings(model, recordingsOfLeftOutPerson_test, confMatrixWithoutTLFileName, f"w/o TL, validated on subject {personId}, fold {index}")
        print(f"Accuracy on test data of left out person {accuracyWithoutTransferLearning}")

        # Store test distribution for this fold
        activityDistributionTestFileName = f"subject{personId}_kfold{index}_testActivityDistribution.png"
        save_activity_distribution_pie_chart(yTestTrue,  activityDistributionTestFileName, title="Test distribution", subtitle=f"Activities of subject {personId} used to test model w/o TL in fold {index}")
        
        # Do transfer learning
        freezeDenseLayers(model)
        _, yTrainTrue = model.windowize_convert_fit(recordingsOfLeftOutPerson_train)

        # Store TL train distribution
        tlActivityDistributionFileName =  f"subject{personId}_kfold{index}_TL_trainActivityDistribution.png"
        save_activity_distribution_pie_chart(yTrainTrue,  tlActivityDistributionFileName, title="TL Train distribution", subtitle=f"Activities of subject {personId} used for transfer learning in fold {index}")

        # Store TL train evaluation
        confMatrixWithTLFileName = f"subject{personId}_kfold{index}_withTL_conf_matrix"
        accuracyWithTransferLearning, f1ScoreMacroWithTransferLearning, f1ScoreWeightedWithTransferLearning, _ = evaluateOnRecordings(model, recordingsOfLeftOutPerson_test, confMatrixWithTLFileName, f"With TL, validated on subject {personId}, fold {index}")
        print(f"Accuracy on test data of left out person {accuracyWithTransferLearning}")

        # Append report
        resultCol.append("Test Activity Distribution "+f"<br />![Test activity distribution]({activityDistributionTestFileName})")
        resultCol.append("Accuracy: "+str(accuracyWithoutTransferLearning)+f"<br />F1-Score Macro: {f1ScoreMacroWithoutTransferLearning}<br />F1-Score Weighted: {f1ScoreWeightedWithoutTransferLearning}<br />![confusion matrix]({confMatrixWithoutTLFileName}.png)")
        resultCol.append("Accuracy: "+str(accuracyWithTransferLearning)+f"<br />F1-Score Macro: {f1ScoreMacroWithTransferLearning}<br />F1-Score Weighted: {f1ScoreWeightedWithTransferLearning}<br />TL-Train activity distribution <br />![TL-Train activity distribution]({tlActivityDistributionFileName})"+f"<br />![confusion matrix]({confMatrixWithTLFileName}.png)")
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




