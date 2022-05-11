import pandas as pd
import numpy as np


def transition_time_cutting(
    recordings: "list[Recording]", timestep_frequency
) -> "list[Recording]":
    """
    1 - max 2 seconds at the end of every activity, to make windows cleaner

    :recordings:
    :timestep_frequency:
        - the timestep_frequency of the Opportunity dataset is 30 Hz
    
    - will return the same number of recordings (no smooth transition anymore)
    - alternative: return much more recordings with only one activity each
    """

    # side effect implementation (modifies input data, no return required)
    # RAM performance decision to not deep copy and return new recordings
    seconds_from_start = 2
    seconds_from_end = 0

    n_timesteps_from_start = int(seconds_from_start * timestep_frequency)
    n_timesteps_from_end = int(seconds_from_end * timestep_frequency)

    for recording in recordings:
        activities = recording.activities.to_numpy()
        # change_idx = on this index new number
        inner_change_idxs = np.where(activities[:-1] != activities[1:])[0] + 1
        # add start and end
        change_idxs = np.concatenate(
            (np.array([0]), inner_change_idxs, np.array([len(inner_change_idx)]))
        )
        cutting_start_end_tuples = []
        for i in range(len(change_idxs) - 1):
            cutting_start_end_tuples.append((change_idxs[i], change_idxs[i + 1]))

        # TODO continue to add, substract n_timesteps, filter out if end smaller than start


"""
>>> df = pd.DataFrame({"a": [3, 4, 5], "b": [6, 7, 8], "c": [9, 10, 11]})
>>> df
   a  b   c
0  3  6   9
1  4  7  10
2  5  8  11
>>> df.iloc[0:1]
   a  b  c
0  3  6  9
>>> df.iloc[0:2]
   a  b   c
0  3  6   9
1  4  7  10
>>> df = pd.DataFrame({"a": [3, 4, 5, 12, 13, 14, 15], "b": [6, 7, 8, 16, 17, 18, 19], "c": [9, 10, 11, 20, 21, 22, 23]})
>>> df
    a   b   c
0   3   6   9
1   4   7  10
2   5   8  11
3  12  16  20
4  13  17  21
5  14  18  22
6  15  19  23
>>> df.iloc[1:3]
   a  b   c
1  4  7  10
2  5  8  11
>>> pd.concat([df.iloc[1:3], df.iloc[4:5]])
    a   b   c
1   4   7  10
2   5   8  11
4  13  17  21
>>> pd.concat([df.iloc[1:3], df.iloc[4:6]])
    a   b   c
1   4   7  10
2   5   8  11
4  13  17  21
5  14  18  22
>>> pd.concat([df.iloc[1:3], df.iloc[4:6], df.iloc[0:2]])
    a   b   c
1   4   7  10
2   5   8  11
4  13  17  21
5  14  18  22
0   3   6   9
1   4   7  10
>>> pd.concat([df.iloc[1:3], df.iloc[4:6], df.iloc[0:2]]).reset_index()
   index   a   b   c
0      1   4   7  10
1      2   5   8  11
2      4  13  17  21
3      5  14  18  22
4      0   3   6   9
5      1   4   7  10
"""

"""
>>> se = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3])
>>> se
0     0
1     0
2     0
3     0
4     1
5     1
6     1
7     1
8     1
9     1
10    0
11    0
12    0
13    1
14    1
15    1
16    3
17    3
18    3
19    3
20    3
21    3
22    3
23    3
dtype: int64
>>> se.iloc[1:5]
1    0
2    0
3    0
4    1
dtype: int64
>>> pd.concat([df.iloc[1:5], df.iloc[7:11], df.iloc[20:23]])
    a   b   c
1   4   7  10
2   5   8  11
3  12  16  20
4  13  17  21
>>> pd.concat([se.iloc[1:5], se.iloc[7:11], se.iloc[20:23]])
1     0
2     0
3     0
4     1
7     1
8     1
9     1
10    0
20    3
21    3
22    3
dtype: int64
>>> pd.concat([se.iloc[1:5], se.iloc[7:11], se.iloc[20:23]]).reset_index()
    index  0
0       1  0
1       2  0
2       3  0
3       4  1
4       7  1
5       8  1
6       9  1
7      10  0
8      20  3
9      21  3
10     22  3
>>> pd.concat([se.iloc[1:5], se.iloc[7:11], se.iloc[20:23]]).reset_index(drop=True)
0     0
1     0
2     0
3     1
4     1
5     1
6     1
7     0
8     3
9     3
10    3
dtype: int64
>>> se_n = se.to_numpy()
>>> import numpy as np
>>> np.where(se_n[:-1] != se_n[1:])[0] + 1
array([ 4, 10, 13, 16])
>>> se
0     0
1     0
2     0
3     0
4     1
5     1
6     1
7     1
8     1
9     1
10    0
11    0
12    0
13    1
14    1
15    1
16    3
17    3
18    3
19    3
20    3
21    3
22    3
23    3
dtype: int64
>>> 
"""
