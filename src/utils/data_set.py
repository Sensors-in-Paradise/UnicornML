from utils.array_operations import split_list_by_percentage
from utils.typing import assert_type
from utils.Recording import Recording
from utils.Window import Window
import numpy as np
from utils.typing import assert_type
import itertools
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
import matplotlib.pyplot as plt
from typing import Union

class DataSet(list):
    def __init__(self, data: "Union[list[Recording], DataSet]" = None, data_config = None):
        if not data is None:
            self.extend(data)
            if isinstance(data, DataSet):
                self.data_config = data.data_config
            else: 
                assert data_config != None, "You have passed data as a list of recordings. In this case you must also pass a data_config which is not None"
                self.data_config = data_config
        else: 
            assert data_config != None, "You have not passed any data to this data set. In this case you must pass a data_config which is not None"
            self.data_config = data_config

    def windowize(self, window_size: int) -> "list[Window]":
        """
        Jens version of windowize
        - no stride size default overlapping 50 percent
        - there is is a test for that method
        - window needs be small, otherwise there will be much data loss
        """
        assert_type([(self[0], Recording)])
        assert (
            window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        if window_size > 25:
            print(
                "\n===> WARNING: the window_size is big with the used windowize algorithm (Jens) you have much data loss!!! (each activity can only be a multiple of the half the window_size, with overlapping a half of a window is cutted)\n"
            )

        self._print_jens_windowize_monitoring(window_size)
        # Refactoring idea (speed): Mulitprocessing https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop/20192251#20192251
        print("windowizing in progress ....")
        recording_windows = list(
            map(lambda recording: recording.windowize(window_size), self)
        )
        print("windowizing done")
        return list(
            itertools.chain.from_iterable(recording_windows)
        )  # flatten (reduce dimension)    

    def split_leave_subject_out(self, test_subject) -> "tuple[DataSet, DataSet]":
        recordings_train = list(
            filter(lambda recording: recording.subject != test_subject, self)
        )
        recordings_test = list(
            filter(lambda recording: recording.subject == test_subject, self)
        )
        return DataSet(recordings_train, self.data_config), DataSet(recordings_test, self.data_config)

    def split_by_subjects(self, subjectsForListA: "list[str]") -> "tuple[DataSet, DataSet]":
        """ 
        Splits the recordings into a tuple of 
            - a sublist of recordings of subjects in subjectsForListA 
            - the recordings of the subjects not in subjectsForListA
        """
        a = list(filter(lambda recording: recording.subject in subjectsForListA, self))
        b = list(filter(lambda recording: recording.subject not in subjectsForListA, self))
        return DataSet(a, self.data_config), DataSet(b, self.data_config)

    def count_activities_per_subject(self)-> "pd.DataFrame":
        values = pd.DataFrame(
            {self[0].subject: self[0].activities.value_counts()})
        for rec in self[1:]:
            values = values.add(pd.DataFrame(
                {rec.subject: rec.activities.value_counts()}), fill_value=0)
        return values

    def count_activities_per_subject_as_dict(self) -> "dict[str, int]":
        resultDict = {}
        for recording in self:
            counts = recording.activities.value_counts()
            for activity_id, count in counts.items():
                if activity_id in resultDict:
                    resultDict[activity_id] += count
                else:
                    resultDict[activity_id] = count
        for activity in self.data_config.raw_label_to_activity_idx_map:
            if not activity in resultDict:
                resultDict[activity] = 0
        return resultDict

    def count_recordings_of_subjects(self) -> "dict[str, int]":
        subjectCount = {}
        for recording in self:
            if recording.subject in subjectCount:
                subjectCount[recording.subject] += 1
            else:
                subjectCount[recording.subject] = 1
        return subjectCount

    def get_people_in_recordings(self) -> "list[str]":
        people = set()
        for recording in self:
            people.add(recording.subject)
        return list(people)

    def plot_activities_per_subject(self, dirPath, fileName: str, title: str = ""):
        values = self.count_activities_per_subject()
        values.plot.bar(figsize=(22, 16))
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.path.join(dirPath, fileName))

    def split_by_percentage(self, test_percentage: float) -> "tuple[DataSet, DataSet]":
        if len(self) == 2: #TODO: check for the number of classes and split for each of the class recordings individually
            recordings_train0, recordings_test0 = self[0].split_by_percentage(test_percentage)
            recordings_train1, recordings_test1 = self[1].split_by_percentage(test_percentage)
            recordings_train = [recordings_train0, recordings_train1]
            recordings_test = [recordings_test0, recordings_test1]
        else:  
            recordings_train, recordings_test = split_list_by_percentage(
                list_to_split=self, percentage_to_split=test_percentage
            )
        print(f"amount of recordings_train: {len(recordings_train)}\n amount of recordings_test: {len(recordings_test)}")
        return DataSet(recordings_train, self.data_config), DataSet(recordings_test, self.data_config)

    def convert_windows_sonar(
        windows: "list[Window]", num_classes: int
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        converts the windows to two numpy arrays as needed for the concrete model
        sensor_array (data) and activity_array (labels)
        """
        assert_type([(windows[0], Window)])

        sensor_arrays = list(map(lambda window: window.sensor_array, windows))
        activities = list(map(lambda window: window.activity, windows))

        # to_categorical converts the activity_array to the dimensions needed
        activity_vectors = to_categorical(
            np.array(activities),
            num_classes=num_classes,
        )

        return np.array(sensor_arrays), np.array(activity_vectors)

    def convert_windows_jens(
        windows: "list[Window]",
        num_classes: int
    ) -> "tuple[np.ndarray, np.ndarray]":
        X_train, y_train = DataSet.convert_windows_sonar(windows, num_classes)
        return np.expand_dims(X_train, -1), y_train

    def _print_jens_windowize_monitoring(self, window_size):
        def n_wasted_timesteps_jens_windowize(recording: "Recording"):
            activities = recording.activities.to_numpy()
            change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1
            # (overlapping amount self.window_size // 2 from the algorithm!)
            def get_n_wasted_timesteps(label_len):
                return (
                    (label_len - window_size) % (window_size // 2)
                    if label_len >= window_size
                    else label_len
                )

            # Refactoring to map? Would need an array lookup per change_idx (not faster?!)
            start_idx = 0
            n_wasted_timesteps = 0
            for change_idx in change_idxs:
                label_len = change_idx - start_idx
                n_wasted_timesteps += get_n_wasted_timesteps(label_len)
                start_idx = change_idx
            last_label_len = (
                len(activities) - change_idxs[-1]
                if len(change_idxs) > 0
                else len(activities)
            )
            n_wasted_timesteps += get_n_wasted_timesteps(last_label_len)
            return n_wasted_timesteps

        def to_hours_str(n_timesteps) -> int:
            hz = 30
            minutes = (n_timesteps / hz) / 60
            hours = int(minutes / 60)
            minutes_remaining = int(minutes % 60)
            return f"{hours}h {minutes_remaining}m"

        n_total_timesteps = sum(map(lambda recording: len(recording.activities), self))
        n_wasted_timesteps = sum(map(n_wasted_timesteps_jens_windowize, self))
        print(
            f"=> jens_windowize_monitoring (total recording time)\n\tbefore: {to_hours_str(n_total_timesteps)}\n\tafter: {to_hours_str(n_total_timesteps - n_wasted_timesteps)}"
        )
        print(f"n_total_timesteps: {n_total_timesteps}")
        print(f"n_wasted_timesteps: {n_wasted_timesteps}")

    def replaceNaN_ffill(self):
        """
        the recordings have None values, this function replaces them with the last non-NaN value of the feature
        """
        assert_type([(self[0], Recording)])
        fill_method = "ffill"
        for recording in self:
            recording.sensor_frame = recording.sensor_frame.fillna(
                method=fill_method)
            recording.sensor_frame = recording.sensor_frame.fillna(
                0)