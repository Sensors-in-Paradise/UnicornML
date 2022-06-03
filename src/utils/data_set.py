from apps.Tobi_UnicornML.src.utils.typing import assert_type
from utils.Recording import Recording
from utils.Window import Window
import numpy as np
from typing import assert_type
import itertools


class DataSet(list[Recording]):
    def __init__(self, data: list[Recording] = None):
        if not data is None:
            self.extend(data)

    def windowize(self, window_size: int) -> "list[Window]":
        """
        Jens version of windowize
        - no stride size default overlapping 50 percent
        - there is is a test for that method
        - window needs be small, otherwise there will be much data loss
        """
        assert_type([(self[0], Recording)])
        assert (
            self.window_size is not None
        ), "window_size has to be set in the constructor of your concrete model class please, you stupid ass"
        if self.window_size > 25:
            print(
                "\n===> WARNING: the window_size is big with the used windowize algorithm (Jens) you have much data loss!!! (each activity can only be a multiple of the half the window_size, with overlapping a half of a window is cutted)\n"
            )

        self._print_jens_windowize_monitoring()
        # Refactoring idea (speed): Mulitprocessing https://stackoverflow.com/questions/20190668/multiprocessing-a-for-loop/20192251#20192251
        print("windowizing in progress ....")
        recording_windows = list(map(lambda recording: recording.windowize(window_size), self))
        print("windowizing done")
        return list(
            itertools.chain.from_iterable(recording_windows)
        )  # flatten (reduce dimension)

    # Helpers ---------------------------------------------------------------------------------------------------------

    def _print_jens_windowize_monitoring(self):
        def n_wasted_timesteps_jens_windowize(recording: "Recording"):
            activities = recording.activities.to_numpy()
            change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1
            # (overlapping amount self.window_size // 2 from the algorithm!)
            def get_n_wasted_timesteps(label_len):
                return (
                    (label_len - self.window_size) % (self.window_size // 2)
                    if label_len >= self.window_size
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

        n_total_timesteps = sum(
            map(lambda recording: len(recording.activities), self)
        )
        n_wasted_timesteps = sum(map(n_wasted_timesteps_jens_windowize, self))
        print(
            f"=> jens_windowize_monitoring (total recording time)\n\tbefore: {to_hours_str(n_total_timesteps)}\n\tafter: {to_hours_str(n_total_timesteps - n_wasted_timesteps)}"
        )
        print(f"n_total_timesteps: {n_total_timesteps}")
        print(f"n_wasted_timesteps: {n_wasted_timesteps}")
    
    
    def leave_subject_out_split(self, test_subject)-> "tuple[DataSet, DataSet]":
        recordings_train = list(filter(
            lambda recording: recording.subject != test_subject, self
        ))
        recordings_test = list(filter(
            lambda recording: recording.subject == test_subject, self
        ))
        return DataSet(recordings_train), DataSet(recordings_test)