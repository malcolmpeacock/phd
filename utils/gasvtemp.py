# python script to compare gas demand vs daily UK temperature.

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
weather_year = '2018'
met_temp_filename = '/home/malcolm/uclan/data/hadcet_mean_2018.csv'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'

met_temp = readers.read_hadcet(met_temp_filename)
print('MET')
print(met_temp.index)
print(met_temp)

gas = readers.read_gas(gas_filename)

inverse_temp = met_temp * -1.0

# normalize

gas = stats.normalize(gas)
inverse_temp = stats.normalize(inverse_temp)

# output plots

gas.plot(label='Gas')
inverse_temp.plot(label='Inverse Temperature')
plt.title('2018 UK mean daily temperature inverse vs gas')
plt.xlabel('Day of the year')
plt.ylabel('Temperature (Degrees C)')
plt.legend(loc='upper right')
plt.show()

stats.print_stats_header()
stats.print_stats(gas, inverse_temp, 'Gas vs Temp')

