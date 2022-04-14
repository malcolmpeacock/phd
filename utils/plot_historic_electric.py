# Investigate the espini electricity demand time series.

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

# Read historic electricity demand time series.
demand_filename = '/home/malcolm/uclan/data/kf/UKDailyELD19832014.csv'
demand = pd.read_csv(demand_filename, header=None, squeeze=True)
d = pd.date_range(start = '1984-01-01', end = '2013-12-31', freq='D' )
electric = pd.Series(demand.values[365:11323], d, dtype='float64', name='demand')

# Convert to TWh
print(electric)

electric = electric * 1e-6
yearly = electric.resample('Y').sum()

# plot
yearly.plot(color='blue', label='Annual Historic Electricity Demand')
plt.title('Yearly Electricity demand')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Electricity Demand ', fontsize=15)
plt.show()
