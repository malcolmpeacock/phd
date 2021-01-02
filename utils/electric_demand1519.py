# plot 5 years of electric demand
# show stats to see how slimilar they are.

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
# supply_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_'

# read in the demand
electric15 = readers.read_electric_hourly(demand_filename + '2015.csv')
demand15 = electric15['ENGLAND_WALES_DEMAND']
electric16 = readers.read_electric_hourly(demand_filename + '2016.csv')
demand16 = electric16['ENGLAND_WALES_DEMAND']
electric17 = readers.read_electric_hourly(demand_filename + '2017.csv')
demand17 = electric17['ENGLAND_WALES_DEMAND']
electric18 = readers.read_electric_hourly(demand_filename + '2018.csv')
demand18 = electric18['ENGLAND_WALES_DEMAND']
electric19 = readers.read_electric_hourly(demand_filename + '2019.csv')
demand19 = electric19['ENGLAND_WALES_DEMAND']

#print(demand19.nsmallest())
# replace zeros with NaN
#demand19 = demand19.replace(0, float("NaN"))
# replace missing values (NaN) by interpolation
#demand19 = demand19.interpolate()


#print(demand19)

# plot one after the other

demand19.plot(label='2019')
demand18.plot(label='2018')
demand17.plot(label='2017')
demand16.plot(label='2016')
demand15.plot(label='2015')
plt.title('England and Wales Electricity Demand')
plt.xlabel('Day of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# plot the values so we see how similar they are

plt.plot(demand19.values, label='2019')
plt.plot(demand18.values, label='2018')
plt.plot(demand17.values, label='2017')
plt.plot(demand16.values, label='2016')
plt.plot(demand15.values, label='2015')
plt.title('England and Wales Electricity Demand')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# drop the last 24 hours from 2016, (leap year) so they are the same length
demand16.drop(demand16.tail(24).index,inplace=True)
print('2019 {} 2018 {} 2017 {} 2016 {} 2015 {}'.format(len(demand19), len(demand18), len(demand17), len(demand16), len(demand15) ) )

stats.print_stats_header()
stats.print_stats(demand18, demand19, '2018')
stats.print_stats(demand17, demand19, '2017')
stats.print_stats(demand16, demand19, '2016')
stats.print_stats(demand15, demand19, '2015')
