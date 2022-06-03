from curses import window
from utils.typing import assert_type
from utils.Recording import Recording
from utils.Window import Window
from utils.array_operations import transform_to_subarrays
from typing import Union
import itertools
import numpy as np
import os
import pandas as pd
import utils.settings as settings


class Windowizer:
    def __init__(self, window_size):
        self.window_size = window_size
    
    

