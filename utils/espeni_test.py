# augment weather with zenith, month, hour, doy etc.
# train ML model on 2018 weather to predict demand
# compare with actual 2017 and 2009 and synthetic addition of heat.

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
import pytz
from sklearn.ensemble import RandomForestRegressor
import pvlib

# custom code
import stats
import readers
import stats

# Read espini electricity demand time series.
year = '2018'
demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
espini = readers.read_espeni(demand_filename, year)
# Convert to TWh
print(espini)
print('Annual Demand for {} was {} TWh'.format(year, espini.sum() * 1e-6))
