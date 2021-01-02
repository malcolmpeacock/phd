# python script to compare daily temperatures for specific locations
# from several sources

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm

# custom code
import stats
import readers

# main program
era_filename = '/home/malcolm/uclan/tools/python/output/loctemp/BlackpoolSquiresGate2018.csv'
midas_temp_filename = '/home/malcolm/uclan/data/midas/midas-open_uk-daily-temperature-obs_dv-201908_lancashire_01090_blackpool-squires-gate_qcv-1_2018.csv'
midas_hour_filename = '/home/malcolm/uclan/data/midas/midas-open_uk-hourly-weather-obs_dv-201908_lancashire_01090_blackpool-squires-gate_qcv-1_2018.csv'

midas_temps = readers.read_midas(midas_temp_filename)
print('MIDAS')
# there are 2 values per day of min and max so take the mean 
midas_temps = midas_temps.resample('D').mean()
midas_temp = (midas_temps['max_air_temp'] + midas_temps['min_air_temp']) / 2.0
print(midas_temp)

# Read midas hourly temperature for blackpool and get daily mean.
print('MIDAS_HOUR')
midas_hourly = readers.read_midas_hourly(midas_hour_filename)
midas_hourly = midas_hourly.resample('D').mean()
print(midas_hourly)

era5_temp = readers.read_loctemp(era_filename)
# resample from hourly to daily
print('ERA5')
# print(era5_temp.index)
print(era5_temp)

# output plots

era5_temp['t_interp'].plot(label='t_interp')
era5_temp['t_weighted'].plot(label='t_weighted')
era5_temp['t_min_min'].plot(label='t_(x1,y1)')
era5_temp['t_min_max'].plot(label='t_(x2,y1)')
era5_temp['t_max_min'].plot(label='t_(x1,y2)')
era5_temp['t_max_max'].plot(label='t_(x2,y2)')
midas_temp.plot(label='MIDAS')
midas_hourly['air_temperature'].plot(label='MIDAS Hourly')
plt.title('2018 Blackpool mean daily temperature from different sources')
plt.xlabel('Day of the year')
plt.ylabel('Temperature (Degrees C)')
plt.legend(loc='upper right')
plt.show()

stats.print_stats_header()
stats.print_stats(era5_temp['t_weighted'], midas_temp, 't_weighted Midas')
stats.print_stats(era5_temp['t_interp'], midas_temp, 't_interp Midas')
stats.print_stats(era5_temp['t_min_min'], midas_temp, 't_min_min Midas')
stats.print_stats(era5_temp['t_min_max'], midas_temp, 't_min_max Midas')
stats.print_stats(era5_temp['t_max_min'], midas_temp, 't_max_min Midas')
stats.print_stats(era5_temp['t_max_max'], midas_temp, 't_max_max Midas')

