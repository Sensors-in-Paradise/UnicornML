
from utils.Recording import Recording
import os
import pandas as pd
import json

def append_poseframe(recording: Recording):
    # try:
    metadata_file_path = os.path.join(recording.recording_folder, 'metadata.json')
    with open(metadata_file_path) as metadata_file:
        metadata = json.load(metadata_file)
        pose_startTime = metadata["startTimestamp"]
        for activity in metadata["activities"]:
            #skipping the first activities if they have the same label
            if activity["label"] != metadata["activities"][0]["label"]:
                pose_time_second_activity = activity["timeStarted"]
                break
    # except:
    #     raise Exception(f"Recording Metadata Corrupt: {recording.recording_folder}")

    pose_file_path = os.path.join(recording.recording_folder, 'poseSequence.csv')
    pose_frame = pd.read_csv(
        pose_file_path, skiprows=_get_pose_sequence_headersize(pose_file_path)
    )
    pose_frame = _adjust_columns_poseframe(
                pose_frame
    )
    # TODO maybe use starttimestamp instead of second activity time and ignore potential shift. 
    absolute_frame_start_time = _get_absolute_start_time(recording, pose_startTime, pose_time_second_activity) 
    pose_frame = _sampleUp_poseframe(
                pose_frame, recording.time_frame, absolute_frame_start_time 
    )
    pose_frame = _delete_timestamps_poseframe(pose_frame)
    recording.pose_frame = pose_frame


def _get_pose_sequence_headersize(pose_file_path: str) -> int:
    headersize = 0
    with open(pose_file_path) as csvFile:
        line = csvFile.readline()[:-1].split(',')
        while len(line) > 0 and line[0] != 'TimeStamp':
            headersize += 1
            line = csvFile.readline().split(',')
    return headersize

 
def _delete_timestamps_poseframe(frame): 
    if "TimeStamp" in frame.columns:
        del frame["TimeStamp"]

    return frame


def _get_absolute_start_time(recording: Recording, pose_start_time, pose_time_second_activity):
    activities = recording.activities
    first_activity = activities[0]
    for i in range(len(activities)):
        if first_activity != activities[i]:
            break

    frame_first_time = recording.time_frame[0]
    frame_time_second_activity = recording.time_frame[i]
    frame_diff = round((frame_time_second_activity - frame_first_time) / 1000)

    pose_diff = pose_time_second_activity - pose_start_time
    assert abs(frame_diff - pose_diff) < 1000, f"Second activity of recording {recording.recording_folder} significantly shifted"

    absolute_frame_start_time = pose_time_second_activity - frame_diff
    return absolute_frame_start_time



def _get_interpolated_pose_row(first_row, second_row, factor):
    return dict(map(
         lambda f_1, f_2: (f_1[0], f_1[1] + ((f_2[1] - f_1[1]) * factor)), 
        list(first_row.items()), 
        list(second_row.items())))

def _get_absolute_frame_time(relative_time, relative_start_time, absolute_start_time):
    relative_time = round((relative_time - relative_start_time) / 1000)

    return absolute_start_time + relative_time

def _sampleUp_poseframe(pose_frame, time_frame, absolute_frame_start_time):
    new_pose_frame = pd.DataFrame()
    feature_columns = list(pose_frame.columns)
    feature_columns.remove("TimeStamp")
    #new_pose_frame[columns] = time_frame.apply(lambda timeRow: _get_interpolated_pose_row(timeRow, pose_frame))

    CRITICAL_TIME_MARGIN = 1000
    NULL_ROW = lambda: dict([(feature, -1) for feature in feature_columns])
    relative_frame_start_time = time_frame[0]
    col_index_timestamp = pose_frame.columns.get_loc("TimeStamp")


    pose_iter = 0
    for _, sample_time_fine in time_frame.iteritems():
        timestamp = _get_absolute_frame_time(sample_time_fine, relative_frame_start_time, absolute_frame_start_time)
        
        pose_timestamp = pose_frame.iloc[pose_iter, col_index_timestamp]
        try: 
            next_pose_timestamp = pose_frame.iloc[pose_iter+1, col_index_timestamp]
        except: 
            new_pose_frame.append(NULL_ROW(), ignore_index=True)
            print(len(new_pose_frame), end="\r")
            break

        if timestamp < pose_timestamp:
            if pose_iter == 0: 
                new_pose_frame.append(NULL_ROW(), ignore_index=True)
                print(len(new_pose_frame), end="\r")
                continue

            else:
                raise Exception("Timestamp to low")
        
        else:
            while next_pose_timestamp <= timestamp:
                pose_iter += 1
                next_pose_timestamp = pose_frame.iloc[pose_iter+1, col_index_timestamp]

            if next_pose_timestamp - pose_timestamp > CRITICAL_TIME_MARGIN:
                new_pose_frame.append(NULL_ROW(), ignore_index=True)
                print(len(new_pose_frame), end="\r")
                continue
            else:
                interpolate_factor = (timestamp - pose_timestamp) / (next_pose_timestamp - pose_timestamp)
                feature_row = _get_interpolated_pose_row(
                    pose_frame.iloc[pose_iter].loc[feature_columns], 
                    pose_frame.iloc[pose_iter+1].loc[feature_columns],
                    interpolate_factor
                )
                new_pose_frame.append(feature_row, ignore_index=True)
                print(len(new_pose_frame), end="\r")
    return pd.DataFrame.from_dict(new_pose_frame)

def _adjust_columns_poseframe(frame):
    if "Confidence" in frame.columns:
        del frame["Confidence"]

    # TODO kick out low confidences 

    # TODO map head points to one
    # TODO map weist points to one

    # Convert all frame values to numbers (otherwise nans might not be read correctly!)
    frame = frame.apply(pd.to_numeric, errors='coerce').astype({"TimeStamp": "int64"})
    
    return frame 
