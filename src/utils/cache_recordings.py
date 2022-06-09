from curses.ascii import LF
import os
import pandas as pd
from utils.data_set import DataSet
from utils.Recording import Recording

def save_recordings(recordings: 'list[Recording]', path: str) -> None:
    """
    Save each recording to a csv file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    for (index, recording) in enumerate(recordings):
        print(f'Saving recording {index} / {len(recordings)}')

        recording.activities.index = recording.sensor_frame.index

        recording_dataframe = recording.sensor_frame.copy()
        recording_dataframe['SampleTimeFine'] = recording.time_frame
        recording_dataframe['activity'] = recording.activities

        filename = str(index) + '_' + recording.subject + '.csv'
        recording_dataframe.to_csv(os.path.join(path, filename), index=False)

    print('Saved recordings to ' + path)


def load_recordings(path: str, activityLabelToIndexMap: dict, limit: int = None) -> "list[Recording]":
    """
    Load the recordings from a folder containing csv files.
    """
    recordings = []

    recording_files = os.listdir(path)
    recording_files = list(filter(lambda file: file.endswith('.csv'), recording_files))
    
    if limit is not None:
        recording_files = recording_files[:limit]

    recording_files = sorted(recording_files, key=lambda file: int(file.split('_')[0]))

    for (index, file) in enumerate(recording_files):
        print(f'Loading recording {file}, {index+1} / {len(recording_files)}')

        recording_dataframe = pd.read_csv(os.path.join(path, file))
        time_frame = recording_dataframe.loc[:, 'SampleTimeFine']
        activities = recording_dataframe.loc[:, 'activity'].map(lambda label: activityLabelToIndexMap[label])
        sensor_frame = recording_dataframe.loc[:, 
            recording_dataframe.columns.difference(['SampleTimeFine', 'activity'])]
        subject = file.split('_')[1]

        recordings.append(Recording(sensor_frame, time_frame, activities, subject, index))

    print(f'Loaded {len(recordings)} recordings from {path}')
    
    return recordings

def load_gait_recordings(path: str, subs: list):
    recordings = []

    recording_folders = os.listdir(path)
    control_folders = list(filter(lambda folders: folders.endswith('_control'), recording_folders))
    fatigue_folders = list(filter(lambda folders: folders.endswith('_fatigue'), recording_folders))

    for (index, sub) in enumerate(subs):
        for folder in control_folders:
            print(f'Loading control recording for {folder}_{sub}')
            sub_folder = os.listdir(os.path.join(path, folder))
            sub_folder = list(filter(lambda folders: folders.startswith(sub), sub_folder))[0]
            
            LF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "LF.csv"))
            RF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "RF.csv"))
            SA_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "SA.csv"))

            GYR_LF_recording = LF_recording.loc[:,'GyrX':'GyrZ']
            GYR_LF_recording.columns = ['GYR_X_LF', 'GYR_Y_LF', 'GYR_Z_LF']
            GYR_RF_recording = RF_recording.loc[:,'GyrX':'GyrZ']
            GYR_RF_recording.columns = ['GYR_X_RF', 'GYR_Y_RF', 'GYR_Z_RF']
            GYR_SA_recording = SA_recording.loc[:,'GyrX':'GyrZ']
            GYR_SA_recording.columns = ['GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']
            ACC_LF_recording = LF_recording.loc[:,'AccX':'AccZ']
            ACC_LF_recording.columns = ['ACC_X_LF', 'ACC_Y_LF', 'ACC_Z_LF']
            ACC_RF_recording = RF_recording.loc[:,'AccX':'AccZ']
            ACC_RF_recording.columns = ['ACC_X_RF', 'ACC_Y_RF', 'ACC_Z_RF']
            ACC_SA_recording = SA_recording.loc[:,'AccX':'AccZ']
            ACC_SA_recording.columns = ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA']

            sensor_frame = pd.concat([GYR_LF_recording, ACC_LF_recording, GYR_RF_recording, ACC_RF_recording, GYR_SA_recording, ACC_SA_recording], axis=1)
            time_frame = LF_recording.loc[:, 'timestamp']
            activities = pd.Series([1] * len(GYR_LF_recording), copy=False) # non-fatigue
            subjext = sub
            recordings.append(Recording(sensor_frame, time_frame, activities, subjext, 2 * index -1))
        
        for folder in fatigue_folders:
            print(f'Loading fatigue recording for {folder}_{sub}')
            sub_folder = os.listdir(os.path.join(path, folder))
            sub_folder = list(filter(lambda folders: folders.startswith(sub), sub_folder))[0]
            
            LF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "LF.csv"))
            RF_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "RF.csv"))
            SA_recording = pd.read_csv(os.path.join(path, folder, sub_folder, "cut_by_stride", "SA.csv"))

            GYR_LF_recording = LF_recording.loc[:,'GyrX':'GyrZ']
            GYR_LF_recording.columns = ['GYR_X_LF', 'GYR_Y_LF', 'GYR_Z_LF']
            GYR_RF_recording = RF_recording.loc[:,'GyrX':'GyrZ']
            GYR_RF_recording.columns = ['GYR_X_RF', 'GYR_Y_RF', 'GYR_Z_RF']
            GYR_SA_recording = SA_recording.loc[:,'GyrX':'GyrZ']
            GYR_SA_recording.columns = ['GYR_X_SA', 'GYR_Y_SA', 'GYR_Z_SA']
            ACC_LF_recording = LF_recording.loc[:,'AccX':'AccZ']
            ACC_LF_recording.columns = ['ACC_X_LF', 'ACC_Y_LF', 'ACC_Z_LF']
            ACC_RF_recording = RF_recording.loc[:,'AccX':'AccZ']
            ACC_RF_recording.columns = ['ACC_X_RF', 'ACC_Y_RF', 'ACC_Z_RF']
            ACC_SA_recording = SA_recording.loc[:,'AccX':'AccZ']
            ACC_SA_recording.columns = ['ACC_X_SA', 'ACC_Y_SA', 'ACC_Z_SA']

            sensor_frame = pd.concat([GYR_LF_recording, ACC_LF_recording, GYR_RF_recording, ACC_RF_recording, GYR_SA_recording, ACC_SA_recording], axis=1)
            time_frame = LF_recording.loc[:, 'timestamp']
            activities = pd.Series([0] * len(GYR_LF_recording), copy=False) # fatigue
            subjext = sub
            recordings.append(Recording(sensor_frame, time_frame, activities, subjext, 2 * index))
    
    print(f'Loaded {len(recordings)} recordings from {path}')
    return recordings
