# HAR Opportunity dataset CNN
- this is a annotated subset of the jens original repo - see orginal Readme below
- 10 Epochs CNN with sensor subset gets accuracy of 0.81
- NO LEAVE SUBJECT OUT: Refactoring for this test is not worth it

How To:
- download the opportunity dataset (put the path in .gitignore if its in this repo)
- edit paths in dataProcessing.py, models.py
- always execute from root (e.g. with the vscode debugger or create new main file)
- execute dataProcessing.py (this will generate two .h5 files in this folder - the preprocessed, windowized data in numpy arrays)
- execute models.py (reads the .h5 files, trains the model, shows conf matrix, training history, and saves the model)
- you will find the exported model and the evaluation (accuracy, conf_matrix) in saved_models


README from the original repo:

## Introduction

This repository is to apply deep learning models on Human Activity Recognition(HAR)/Activities of Daily Living(ADL) datasets. Three deep learning models, including Convolutional Neural Networks(CNN), Deep Feed Forward Neural Networks(DNN) and Recurrent Neural Networks(RNN) were applied to the datasets. Six HAR/ADL benchmark datasets were tested. The goal is to gain some experiences on handling the sensor data as well as classifying human activities using deep learning.

## Benchmark datasets

- [PAMAP2 dataset](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) contains data of 18 different physical activities (such as walking, cycling, playing soccer, etc.), performed by 9 subjects wearing 3 inertial measurement units and a heat rate monitor.
- [OPPORTUNITY dataset](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition) contains data of 35 ADL activities (13 low-level, 17 mid-level and 5 high-level) which were collected through 23 body worn sensors, 12 object sensors, 21 ambient sensors.
- [Daphnet Gait dataset](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait) contains the annotated readings of 3 acceleration sensors at the hip and leg of Parkinson's disease patients that experience freezing of gait (FoG) during walking tasks.
- [UCI HAR dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) contains data of 6 different physical activities walking, walking upstairs, walking downstairs, sitting, standing and laying), performed by 30 subjects wearing a smartphone (Samsung Galaxy S II) on the waist.
- [Sphere dataset](https://www.irc-sphere.ac.uk/sphere-challenge/home) contains the data collected from three sensing modalities (wrist-worn accelerometer, RGB-D cameras, passiva enviormental sensors). 20 ADL activities including static and transition activites were labeled.
- [SHL dataset](http://www.shl-dataset.org/) contains multi-modal data from a body-worn camera and from 4 smartphones, carried simultaneously at typical body locations. The SHL dataset contains 750 hours of labelled locomotion data: Car (88 h), Bus (107 h), Train (115 h), Subway (89 h), Walk (127 h), Run (21 h), Bike (79 h), and Still (127 h).

## Approach

- For each dataset, a slicing window appoarch was applied to segment the dataset. Each segment includes a series of data (usually 25 sequential data points) and two continuous windows have 50% overlapping.
- After data preprocessing which includes reading files, data cleaning, data visualization, relabling and data segmentation, the data was saved into hdf5 files.
- Deep learning models including CNN, DNN and RNN were applied. For each model in each dataset, hyperparameters were optimized to get the best performance.
- To combine the data from multimodalities, different data fusion methods were applied on Sphere and SHL dataset.

## Dependencies

- Python 3.7
- tensorflow 1.13.1

## Usage

First download the dataset and put dataProcessing.py and models.py under the same directory. Then run dataprocessing to generate h5 file. Last switch model types in models.py script and run different deep learning models using the generated h5 data file.

## Note

I am still working on tuning hyperparameters of models in certain datasets. There will be more updates.
