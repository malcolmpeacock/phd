# python script to compare offshore wind and wind generation from:
#  midas, era5, renewables ninja  ??? TODO

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

# Time series of wind power for midas weather stations from ERA5 weather.
# era_filename = '/home/malcolm/uclan/output/wind/era520181.csv'
# Midas conventional 
midas_filename = '/home/malcolm/uclan/output/wind/midas20181.csv'
midas = pd.read_csv(midas_filename, header=0, parse_dates=[0], index_col=0 )
# Midas marine
midas_filename = '/home/malcolm/uclan/output/wind/midas20181_m.csv'
marine = pd.read_csv(midas_filename, header=0, parse_dates=[0], index_col=0 )

# convert to daily by averaging the hours, so this is mean wind speed
offshore = marine['wind_m'].resample('D').mean()
blackpool = midas['wind_b'].resample('D').mean()

# output wind plots

offshore.plot(label='offshore')
blackpool.plot(label='blackpool')
plt.title('2018 Daily mean Wind speed Blackpool Squires Gate and Offshore')
plt.xlabel('Day of the year')
plt.ylabel('Wind Speed (M/S)')
plt.legend(loc='upper right')
plt.show()

# convert to daily by summing the hours, so this is energy/day in kWh
midas = midas.resample('D').sum()
marine = marine.resample('D').sum()

# output power plots
power_m = marine['power_m'] * 0.001
power_b = midas['power_b'] * 0.001

power_m.plot(label='Offshore')
power_b.plot(label='Blackpool')

plt.title('2018 wind power from Blackpool Squires gate and Offshore')
plt.xlabel('Day of the year')
plt.ylabel('Energy generated (MWh)')
plt.legend(loc='upper right')
plt.show()

stats.print_stats_header()
stats.print_stats(marine['wind_m'], midas['wind_b'], 'blackpool and offshore wind')
stats.print_stats(power_m, power_b, 'offshore and blackpool power')

print('Annual generation blackpool : {} offshore {}'.format(power_b.sum(), power_m.sum() ) )
