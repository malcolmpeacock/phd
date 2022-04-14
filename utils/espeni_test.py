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

# Read espini electricity demand time series.
year = '2018'
demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
espini = readers.read_espeni(demand_filename, year)
# Convert to TWh
print(espini)
print('Annual Demand for {} was {} TWh'.format(year, espini.sum() * 1e-6))

espini = readers.read_espeni(demand_filename, None)
espini = espini * 1e-6
daily = espini.resample('D').sum()

# plot
daily.plot(color='blue', label='Historic Electricity Demand')
plt.title('Daily Electricity demand and generation')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Electricity Demand ', fontsize=15)
# plt.legend(loc='upper right')
plt.show()

output_filename = '/home/malcolm/uclan/output/timeseries/historic_demand.csv'
espini.to_csv(output_filename)
