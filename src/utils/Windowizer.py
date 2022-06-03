import itertools

import numpy as np

from utils.Recording import Recording
from utils.Window import Window
from utils.typing import assert_type


class Windowizer:
    def __init__(self, window_size):
        self.window_size = window_size
    
    

