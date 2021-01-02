# python script to compare daily temperatures from several sources.

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
from misc import upsample_df

# main program
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/heatCopRef2018weather2018HDD15.5.csv'
gas_temp_filename = '/home/malcolm/uclan/data/DailyTempGasExplorer2018.csv'
met_temp_filename = '/home/malcolm/uclan/data/hadcet_mean_2018.csv'

era5_temp = readers.read_demand(demand_filename, 'temperature')
# resample from hourly to daily
era5_temp = era5_temp.resample('D').mean()
# change index so only have date not time
era5_temp.index = pd.DatetimeIndex(pd.to_datetime(era5_temp.index).date)
print('ERA5')
# print(era5_temp.index)
print(era5_temp)

gas_temp = readers.read_gas(gas_temp_filename)
print('GAS')
print(gas_temp.index)
print(gas_temp)

met_temp = readers.read_hadcet(met_temp_filename)
print('MET')
print(met_temp.index)
print(met_temp)

# output plots

era5_temp.plot(label='ERA5')
gas_temp.plot(label='Gas')
met_temp.plot(label='Met')
plt.title('2018 UK mean daily temperature from different sources')
plt.xlabel('Day of the year')
plt.ylabel('Temperature (Degrees C)')
plt.legend(loc='upper right')
plt.show()

stats.print_stats_header()
stats.print_stats(era5_temp, met_temp, 'ERA5 Met')
stats.print_stats(era5_temp, gas_temp, 'Gas  Met')

