from psutil import sensors_fans
from utils.typing import assert_type

import pandas as pd
from dataclasses import dataclass
import numpy as np
from typing import Union
from utils.Window import Window

@dataclass
class Recording:
    """
    our base data object
    Multilabel, so expects activity pd.Series

    Future: 
        - a dataclass creates the intializer automatically
            - consider only giving the attributes as class vars -> dataclass handles this
        - add self.recorder

    Refactoring idea:
    - subject should be a int!, add it to assert_type
    """

    def __init__(
        self,
        sensor_frame: pd.DataFrame,
        time_frame: pd.Series,
        activities: pd.Series,
        subject: Union[str, int],
        recording_index: int,
    ) -> None:
        assert_type(
            [
                (sensor_frame, pd.DataFrame),
                (time_frame, pd.Series),
                (activities, pd.Series),
                (recording_index, int),
            ]
        )
        assert isinstance(activities[0], np.float64) or isinstance(
            activities[0], np.int64
        )
        assert (
            sensor_frame.shape[0] == time_frame.shape[0]
        ), "sensor_frame and time_frame have to have the same length"
        assert (
            sensor_frame.shape[0] == activities.shape[0]
        ), "sensor_frame and activities have to have the same length"
        
        self.sensor_frame = sensor_frame
        self.time_frame = time_frame
        self.activities = activities
        self.subject = subject
        self.recording_index = recording_index

    def windowize(self, window_size: int, features : "Union[list[str], None]" = None) -> "list[Window]":
        windows = []

        sensor_frame = self.sensor_frame if features==None else self.sensor_frame[features]
       
        recording_sensor_array = (
            sensor_frame.to_numpy()
        )  # recording_sensor_array[timeaxis/row, sensoraxis/column]
        activities = self.activities.to_numpy()

        start = 0
        end = 0

        def last_start_stamp_not_reached(start):
            return start + window_size - 1 < len(recording_sensor_array)

        while last_start_stamp_not_reached(start):
            end = start + window_size - 1

            # has planned window the same activity in the beginning and the end?
            if (
                len(set(activities[start : (end + 1)])) == 1
            ):  # its important that the window is small (otherwise can change back and forth) # activities[start] == activities[end] a lot faster probably
                window_sensor_array = recording_sensor_array[
                    start : (end + 1), :
                ]  # data[timeaxis/row, featureaxis/column] data[1, 2] gives specific value, a:b gives you an interval

                activity = activities[start]  # the first data point is enough
                start += (
                    window_size // 2
                )  # 50% overlap!!!!!!!!! - important for the waste calculation
                windows.append(
                    Window(
                        window_sensor_array,
                        int(activity),
                        self.subject,
                        self.recording_index,
                    )
                )

            # if the frame contains different activities or from different objects, find the next start point
            # if there is a rest smaller than the window size -> skip (window small enough?)
            else:
                # find the switch point -> start + 1 will than be the new start point
                # Refactoring idea (speed): Have switching point array to find point immediately
                # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy/19125898#19125898
                while last_start_stamp_not_reached(start):
                    if activities[start] != activities[start + 1]:
                        start += 1
                        break
                    start += 1
        return windows

