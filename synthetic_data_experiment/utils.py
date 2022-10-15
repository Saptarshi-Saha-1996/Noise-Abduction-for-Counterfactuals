import os
import shutil
import numpy as np
import pandas as pd


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

