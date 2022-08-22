# model a hybrid heat pump.

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

# percentage gas
percent_gas = 0.25
efficiency = 0.85

# read 2017 electric heat for ref year 2018 temperature, heat, electric

year = '2018'
#demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2017/GBRef2018Weather2017I-Sbdew.csv'
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef2018Weather{}I-Brhpp.csv'.format(year, year)
demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
print(demand)

total_heat = demand['heat'].sum()
print('Total Heat {}'.format(total_heat))
original_electric = demand['electricity']

# sort by tempearture

demand = demand.sort_values('temperature')
gas = demand['electricity'] * 0.0
heat_so_far = 0
threshold_temperature = 0

# starting with coldest hour, allocate to gas and zero electric
# until the % energy is reached is reached.
# (this is based on FES 2019 assumption of % energy used so we find
#  out what threshold temperature this would need )
#
irow=0
for index, row in demand.iterrows():
    heat_so_far += row['heat']
    # divide be efficiency because we use more primary gas energy than heat.
    gas.iloc[irow] = row['heat'] / efficiency
    demand.iloc[irow,demand.columns.get_loc('electricity')] = 0.0
    threshold_temperature = row['temperature']
    print(heat_so_far,threshold_temperature)
    if heat_so_far > total_heat * percent_gas:
        break
    irow+=1

# gas = heat * 0.85
# electric = 0
demand = demand.sort_index()
gas = gas.sort_index()
print(demand)
print(gas)

# output the temperature - this is the threshold temperature.
print('threshold temperature {}'.format(threshold_temperature))

# plot of daily

daily_electric_hybrid = demand['electricity'].resample('D').sum()
daily_electric_original = original_electric.resample('D').sum()
daily_gas = gas.resample('D').sum()
daily_temperature = demand['temperature'].resample('D').mean()

# plot:
# temperature
ax1 = daily_temperature.plot(color='red',label='Air temperature')
plt.ylabel('Temperature (degrees C)', fontsize=15, color='red')
# threshold temperature
daily_threshold = daily_temperature * 0.0 + threshold_temperature
daily_threshold.plot(color='red',label='threshold temperature')
# 2nd axis
ax2 = ax1.twinx()
ax2.set_ylabel('Demand (MWh)',color='black', fontsize=15)
#  -the hybrid heat pump electric series
daily_electric_hybrid.plot(ax=ax2,color='blue',label='Hybrid heat pump electricity')
#  -the original electric series
daily_electric_original.plot(ax=ax2,color='green',label='Ordinary heat pump electricity')
#  -the hybrid heat pump gas series
daily_gas.plot(ax=ax2,color='yellow',label='Hybrid heat pump gas')
plt.title('{} weather hybrid heat pumps'.format(year))
plt.xlabel('Hour of the year')
plt.legend(loc='upper right')
plt.show()

# plot of 4 days hourly

days4_electric_hybrid = demand['electricity']
start_time = '{}-03-14 00:00:00'.format(year)
end_time = '{}-03-17 23:00:00'.format(year)
days4_electric_hybrid = days4_electric_hybrid[start_time : end_time ]
days4_electric_original = original_electric[start_time : end_time ]
days4_gas = gas[start_time : end_time ]
days4_temperature = demand['temperature']
days4_temperature = days4_temperature[start_time : end_time ]
# plot:
# temperature
ax1 = days4_temperature.plot(color='red',label='Air temperature')
plt.ylabel('Temperature (degrees C)', fontsize=15, color='red')
# threshold temperature
days4_threshold = days4_temperature * 0.0 + threshold_temperature
days4_threshold.plot(color='red',label='threshold temperature')
# 2nd axis
ax2 = ax1.twinx()
ax2.set_ylabel('Demand (MWh)',color='black', fontsize=15)
#  -the hybrid heat pump electric series
days4_electric_hybrid.plot(ax=ax2,color='blue',label='Hybrid heat pump electricity')
#  -the original electric series
days4_electric_original.plot(ax=ax2,color='green',label='Ordinary heat pump electricity')
#  -the hybrid heat pump gas series
days4_gas.plot(ax=ax2,color='yellow',label='Hybrid heat pump gas')
plt.title('{} weather hybrid heat pumps'.format(year))
plt.xlabel('Hour of the year')
plt.legend(loc='upper center')
plt.show()
