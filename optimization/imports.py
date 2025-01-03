# Add paths to the sys environment 
# sys.path.append('../iot_grabber')

from toolkit.iot_grabber import IotGrabber
from toolkit.rmse import calc_rmse
from toolkit.sine_fitting import SineFitting
from toolkit.prediction import Prediction
from toolkit.convert_utc import utc_to_local


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error

from collections import defaultdict
import datetime
import sys
import math
import json

from datetime import datetime
from dateutil import tz

from datetime import datetime

from astral.sun import sun
from astral import LocationInfo



#include ../../iot-grabber.iot-grabber

# %matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

Colour_Palette = ['#01BEFE', '#FF7D00', '#FFDD00', '#FF006D', '#ADFF02', '#8F00FF']
sns.set_palette(sns.color_palette(Colour_Palette))

tqdm.pandas()