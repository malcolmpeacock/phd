# Investigate the espini electricity demand time series.

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
import pytz
from datetime import datetime
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

electric = electric * 1e-6
yearly_grid = electric.resample('Y').sum()

# allow for Scotland
yearly_grid = yearly_grid * 1.1
print(yearly_grid)

demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
espini = readers.read_espeni(demand_filename, None)
espini = espini * 1e-6
yearly_espini = espini.resample('Y').sum()
yearly_espini = yearly_espini[1:-1]
print(yearly_espini)

# dukes
demand_filename = '/home/malcolm/uclan/data/electricity/annualDemandDUKES.csv'
dukes = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y'),index_col=0, squeeze=True)
print(dukes)

# plot
yearly_grid.plot(color='blue', label='Annual Historic Electricity Demand (KF)')
yearly_espini.plot(color='red', label='Annual Historic Electricity Demand (Espini)')
dukes.plot(color='green', label='Annual Historic Electricity Demand (DUKES)')
plt.title('Yearly Electricity demand')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Electricity Demand ', fontsize=15)
plt.legend(loc='upper left')
plt.show()

