# python script to validate the heat demand methods against gas meter data

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
# custom code
import stats

# main program

# read in the data
output_dir = "/home/malcolm/uclan/tools/python/scripts/heat/output/adv/"
filename = output_dir + 'winter_wind_drought_uk_return_period_1_in_5_years_severity_gwl12-3degC_event1.csv'
df2 = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
filename = output_dir + 'winter_wind_drought_uk_return_period_1_in_5_years_severity_gwl4degC_event1.csv'
df4 = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

heat2 = df2['heat'].resample('Y').sum() * 1e-6
temp2 = df2['temperature'].resample('Y').mean()
print(heat2)
heat4 = df4['heat'].resample('Y').sum() * 1e-6
temp4 = df4['temperature'].resample('Y').mean()
print(heat4)

print('Warming  Heat Demand   Mean Temp')
print(' 2-3     {}            {}       '.format(heat2.values[0], temp2.values[0]) )
print(' 2-3     {}            {}       '.format(heat2.values[1], temp2.values[1]) )
print(' 4       {}            {}       '.format(heat4.values[0], temp4.values[0]) )
print(' 4       {}            {}       '.format(heat4.values[1], temp4.values[1]) )
