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
import argparse
#import pvlib

# custom code
import stats
import readers
import stats

# process command line
parser = argparse.ArgumentParser(description='Plot historic demand')
parser.add_argument('--dukes', action="store_true", dest="dukes", help='Plot dukes only', default=False)
args = parser.parse_args()

# Read historic electricity demand time series.
demand_filename = '/home/malcolm/uclan/data/kf/UKDailyELD19832014.csv'
demand = pd.read_csv(demand_filename, header=None).squeeze()
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
dukes = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y'),index_col=0 ).squeeze()
print(dukes)

# plot
if not args.dukes:
    yearly_grid.plot(color='blue', label='National Grid (Fragaki et. al paper)')
    yearly_espini.plot(color='red', label='ESPINI (Combination of Elexon and National Grid)')
dukes.plot(color='green', label='DUKES (Digest of UK Energy Statistics')
plt.title('Annual Historic Electricity demand')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Electricity Demand (TWh)', fontsize=15)
if not args.dukes:
    plt.legend(loc='upper left')
plt.show()

