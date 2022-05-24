

def deleteConsecutives(series):
    filtered_items = []
    last_item = None
    for i, item in series.iteritems():
        if item != last_item:
            filtered_items.append(item)
        last_item = item
    return filtered_items

def getPoseFilesWithMetadata(dataset_path: str, activity_mapping = None):
    from utils.file_functions import get_subfolder_names
    import json
    import os
    import pandas as pd

    recording_folder_names = get_subfolder_names(dataset_path)
    recording_folder_names = [
        os.path.join(dataset_path, recording_folder_name)
        for recording_folder_name in recording_folder_names
    ]
    pose_file_metadata_tuples = []
    for recording_folder_path in recording_folder_names:
        for file_name in os.listdir(recording_folder_path):
            is_file = os.path.isfile(os.path.join(recording_folder_path, file_name))
            if is_file and file_name.endswith("poseSequence.csv"):
                pose_file_name = os.path.join(recording_folder_path, file_name)
            if is_file and file_name.endswith("metadata.json"):
                metadata_file_name = os.path.join(recording_folder_path, file_name)
            if is_file and file_name.endswith(".csv") and not file_name.endswith("poseSequence.csv")
                sensor_file_name = os.path.join(recording_folder_path, file_name)

        try:
            sensor_frame = pd.read_csv(
                sensor_file_name, skiprows=8
            )

            with open(metadata_file_name) as metadata_file:
                metadata = json.load(metadata_file)
            
            activities = list(map(
                lambda activityDict: activityDict["label"],
                metadata["activities"]
            ))

            if activity_mapping != None: 
                activities = list(map(
                    lambda activity: -1 if activity not in activity_mapping.keys() else activity_mapping[activity],
                    activities
                ))
            metadata = {
                "activities": activities,
                "startTimestamp": metadata["startTimeStamp"],
                "subject": metadata["person"],
                "numLines": sensor_frame.shape[0],

            }
            pose_file_metadata_tuples.append((pose_file_name, metadata))
        except:
            raise Exception(f"Recording with missing metadata or poseSequence found: {recording_folder_path}")
    return pose_file_metadata_tuples


    #         headersize = 0
    #     with open(pose_file_path) as csvFile:
    #         line = csvFile.readline()[:-1].split(',')
    #         while len(line) > 0 and line[0] != 'TimeStamp':
    #             headersize += 1
    #             line = csvFile.readline().split(',')

    #     pose_frame = pd.read_csv(
    #         pose_file_name, headersize
    #     )

    #     assert recording_frame != None, f"Recording without sensor data found: {recording_folder_path}"

    #     pose_frame = XSensRecordingReader.__adjust_columns_poseframe(
    #                 pose_frame
    #     )

    #     pose_frame = XSensRecordingReader.__sampleUp_poseframe(
    #                 recording_frame, time_frame
    #     )


    #     pose_frame = XSensRecordingReader.__delete_timestamps_poseframe(pose_frame)

    #     return pose_frame 

    # @staticmethod 
    # def __delete_timestamps_poseframe(frame): 
    #     if "TimeStamp" in frame.columns:
    #         del frame["TimeStamp"]

    #     return frame

    # @staticmethod
    # def _get_interpolated_pose_row(first_row, second_row, factor):
    #     return dict(map(
    #          lambda f_1, f_2: (f_1[0], f_1[1] + ((f_2[1] - f_1[1]) * factor)), 
    #         first_row.itertuples(), 
    #         second_row.itertuples()))

    # @staticmethod
    # def __sampleUp_poseframe(pose_frame, time_frame):
    #     new_pose_frame = pd.DataFrame()
    #     feature_columns = pose_frame.columns
    #     feature_columns = feature_columns.remove("TimeStamp")
    #     #new_pose_frame[columns] = time_frame.apply(lambda timeRow: _get_interpolated_pose_row(timeRow, pose_frame))

    #     CRITICAL_TIME_MARGIN = 1000
    #     NULL_ROW = lambda: dict([(feature, -1) for feature in feature_columns])

    #     pose_iter = 0
    #     for timeRow in time_frame.iterrows():
    #         timestamp = timeRow["SampleTimeFine"]
            
    #         pose_timestamp = pose_frame.iloc[pose_iter, ["TimeStamp"]]
    #         try: 
    #             next_pose_timestamp = pose_frame.iloc[pose_iter+1, ["TimeStamp"]]
    #         except: 
    #             new_pose_frame.append(NULL_ROW(), ignore_index=True)
    #             continue

    #         if timestamp < pose_timestamp:
    #             if pose_iter == 0: 
    #                 new_pose_frame.append(NULL_ROW(), ignore_index=True)
    #                 continue

    #             else:
    #                 raise Exception
            
    #         else:
    #             while next_pose_timestamp <= timestamp:
    #                 pose_iter += 1
    #                 next_pose_timestamp = pose_frame.iloc[pose_iter+1, ["TimeStamp"]]

    #             if next_pose_timestamp - pose_timestamp > CRITICAL_TIME_MARGIN:
    #                 new_pose_frame.append(NULL_ROW(), ignore_index=True)
    #                 continue
    #             else:
    #                 interpolate_factor = (timestamp - pose_timestamp) / (next_pose_timestamp - pose_timestamp)
    #                 feature_row = XSensRecordingReader.get_interpolated_pose_row(
    #                     pose_frame[pose_iter, feature_columns], 
    #                     pose_frame[pose_iter+1, feature_columns],
    #                     interpolate_factor
    #                 )
    #                 new_pose_frame.append(feature_row)
        
    #     return new_pose_frame

    # @staticmethod
    # def __adjust_columns_poseframe(frame):
    #     if "Confidence" in frame.columns:
    #         del frame["Status"]

    #     # map head points to one
    #     # map weist points to one

    #     # Convert all frame values to numbers (otherwise nans might not be read correctly!)
    #     frame = frame.apply(pd.to_numeric, errors='coerce').astype({"TimeStamp": "int64"})
    #     #  frame = XSensRecordingReader.__remove_SampleTimeFine_overflow(frame)
    #     # XSensRecordingReader.__add_suffix_except_SampleTimeFine(frame, suffix)
        
    #     return frame 