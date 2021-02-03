# python script to create a pv forecast from the previous pv and
# the weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Create pv forecast.')
parser.add_argument('pv', help='PV file')
parser.add_argument('weather', help='Weather file')

args = parser.parse_args()
weather_file = args.weather
pv_file = args.pv

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"

# pv data
pv_filename = input_dir + pv_file
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(pv)

# weather data
weather_filename = input_dir + weather_file
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(weather)

# error checks
n_missing = utils.missing_times(pv, '30min')
print('Number of missing pv rows: {}'.format(n_missing) )
print('NaNs pv_power_mw {} irradiance {} temp {}'.format(pv['pv_power_mw'].isna().sum(), pv['irradiance_Wm-2'].isna().sum(), pv['panel_temp_C'].isna().sum()) )

large_pv = pv['pv_power_mw'].max() * 0.8
small_pv = pv['pv_power_mw'].max() * 0.2
large_irrad = pv['irradiance_Wm-2'].max() * 0.8
small_irrad = pv['irradiance_Wm-2'].max() * 0.2
large_temp = pv['panel_temp_C'].max() * 0.8
small_temp = pv['panel_temp_C'].max() * 0.2

# pv large but irradiance small or temp small
pv_large = pv[pv['pv_power_mw']>large_pv]
suspect = pv_large[pv_large['irradiance_Wm-2']<small_irrad]
print('PV large but irradiance small {}'.format(len(suspect)) )
print(suspect)
suspect = pv_large[pv_large['panel_temp_C']<small_temp]
print('PV large but temp small {}'.format(len(suspect)) )
print(suspect)

# pv small but irradiance large or temp large
pv_small = pv[pv['pv_power_mw']<small_pv]
suspect = pv_small[pv_small['irradiance_Wm-2']>large_irrad]
print('PV small but irradiance large {}'.format(len(suspect)) )
print(suspect)
suspect = pv_small[pv_small['panel_temp_C']>large_temp]
print('PV small but temp large {}'.format(len(suspect)) )
print(suspect)

# fix errors
pv['pv_power_mw']['2018-05-08 14:00:00'] = pv['pv_power_mw']['2018-05-08 13:00:00']
pv['pv_power_mw']['2018-06-15 11:30:00'] = pv['pv_power_mw']['2018-06-15 11:00:00']
pv['pv_power_mw']['2018-06-15 12:00:00'] = pv['pv_power_mw']['2018-06-15 12:30:00']

#low = pv[pv < 0.1]
#print(low)
#last_week = pv[low.index - pd.Timedelta(days=7)]
#last_week.index = low.index
#print(last_week)
#pv.update(last_week)

#bad_day = pv['2018-05-10']
#print(bad_day)
#previous_day = pv['2018-05-09']
#previous_day.index = bad_day.index
#print(previous_day)
#pv.update(previous_day)

# replace a suspect days with different ones.
#utils.replace_day(pv, '2018-05-10', '2018-05-09')
#utils.replace_day(pv, '2018-05-11', '2018-05-12')

# plot pv
pv['pv_power_mw'].plot(label='pv power', color='blue')
plt.title('pv')
plt.xlabel('Hour of the year', fontsize=15)
plt.ylabel('PV Generation (MW)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

pv['irradiance_Wm-2'].plot(label='iradiance', color='blue')
plt.title('PV System Measured Irradiance')
plt.xlabel('Hour of the year', fontsize=15)
plt.ylabel('Irradiance (MW)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()
pv['panel_temp_C'].plot(label='panel temp', color='blue')
plt.title('Panel Temperature')
plt.xlabel('Hour of the year', fontsize=15)
plt.ylabel('temperature (degrees C)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# Niaive PV forecast based on same as last week
pv['probability'] = 0.6
pv_forecast = pv[['pv_power_mw','probability']].tail(7*48)
print(pv_forecast)
next_day = pv_forecast.last_valid_index() + pd.Timedelta(minutes=30)
print(next_day)
last_day = pv_forecast.last_valid_index() + pd.Timedelta(days=7)
print(last_day)
#pv_forecast.index = pd.DatetimeIndex(
new_index = pd.date_range(start = next_day, end= last_day, freq='30min')
print(new_index)
pv_forecast.index = new_index
print(pv_forecast)

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = output_dir + 'pv_forecast.csv'

pv_forecast.to_csv(output_filename)
