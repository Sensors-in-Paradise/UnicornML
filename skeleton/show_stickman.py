import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from bisect import bisect_left
# 
#matplotlib.use('TkAgg')

BODY_PARTS = ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']

BODY_LINES = [('NOSE', 'LEFT_EYE'),
        ('NOSE', 'RIGHT_EYE'),
        ('LEFT_EYE', 'LEFT_EAR'),
        ('RIGHT_EYE', 'RIGHT_EAR'),
        ('NOSE', 'LEFT_SHOULDER'),
        ('NOSE', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_HIP'),
        ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'),
        ('LEFT_HIP', 'LEFT_KNEE'),
        ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'),
        ('RIGHT_KNEE', 'RIGHT_ANKLE')]


def binary_search(a, x, lo=0, hi=None):
    if hi is None: hi = len(a)
    pos = bisect_left(a, x, lo, hi)                  # find insertion position
    return pos if pos != hi and a[pos] == x else -1  # don't walk off the end

# def findRowsFromTimestamp(csvReader: csv.DictReader, timestamps: 'list[str]'):
#      = csvReader.restval
#     #pos = binary_search(lambda x: x["TimeStamp"])
#     return rows[pos], rows[pos+1]

def extractCsvReader(file): 
    line = file.readline()[:-1].split(',')
    while len(line) > 0 and line[0] != 'TimeStamp':
        line = file.readline().split(',')

    csvReader = []
    row = file.readline().split(',')
    while row[0] != '':
        csvDict = dict([(fieldname, row[i]) for i, fieldname in enumerate(line)])
        csvReader.append(csvDict)
        print(row)

        row = file.readline().split(',')

    return csvReader

fig = plt.figure()
count = 0
with open('skeleton/poseSequence/poseSequence.csv') as file:
    
    csvReader = extractCsvReader(file)

    print(csvReader[0].keys())
    
    for i, row in enumerate(csvReader):
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        if i == 0:
            lastTimestamp = int(row["TimeStamp"]) - 1000
        
        for bodyPart in BODY_PARTS:
            x = float(row[f"{bodyPart}_X"])
            y = -float(row[f"{bodyPart}_Y"]) + 1

            plt.scatter(x, y, c='orange')

        for bodyLine in BODY_LINES:
            x_1 = float(row[f"{bodyLine[0]}_X"])
            y_1 = -float(row[f"{bodyLine[0]}_Y"]) + 1
            x_2 = float(row[f"{bodyLine[1]}_X"])
            y_2 = -float(row[f"{bodyLine[1]}_Y"]) + 1

            plt.plot([x_1, x_2], [y_1, y_2], c='black')

        # newTimestamp = int(row["TimeStamp"])
        # pauseTime = (newTimestamp - lastTimestamp) / 1000
        # plt.pause(pauseTime)
        # lastTimestamp = newTimestamp    
        
        
        fig.clf()

plt.show()
