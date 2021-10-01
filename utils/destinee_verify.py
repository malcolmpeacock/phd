# Compare my electric heat with DESTINee
#   DESTINEE assumptions:
#    50% heat pumps
#    Electric Vehicles
#    Improvements in applicances and efficiency etc.

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

# assumptions
percent_heat_pumps = 0.5

# read 2010 historical electricity demand
demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
demand = readers.read_espeni(demand_filename, '2010')
# convert from MWh to GWh
historic = demand / 1000.0
print(historic)

# read 2010 electric heat series based on purely resistive heating so that
# it can be removed from the 2010 series. (ref year method)

#demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2010/GBRef2010Weather2010I-Bbdew_resistive.csv'
#resistive_heat = readers.read_demand(demand_filename)

# read 2010 electric heat for ref year 2010

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2010/GBRef2010Weather2010I-Bbdew.csv'
heat = readers.read_demand(demand_filename)
# convert from MWh to GWh
heat = heat / 1000.0
print(heat)

# TODO subtract the 2010 resistive heat

# add in 50% heat pumps for 2010
mine = historic + (percent_heat_pumps * heat)

# read destinee
destinee_filename = '/home/malcolm/uclan/data/destinee/destinee2010.csv'
destinee2010 = readers.read_destinee(destinee_filename)
print(destinee2010)
destinee_filename = '/home/malcolm/uclan/data/destinee/destinee2050.csv'
destinee2050 = readers.read_destinee(destinee_filename)
print(destinee2050)
destinee2050.index = destinee2010.index
# compare: actual 2010, DESTINee simulated 2010, DESTINEE 2050, my 2010

#ref_method = mod_electric17 + heat17.values

#  Factor to modify the other heat (industry etc) from the reference year
#  to the weather year. DESTinEE assumes its constant, so nothing to do.

#  plot of historic and electric
historic.plot(label='Historic Electricity Demand 2010')
mine.plot(label='Historic with my 50% heat pumps 2010')
destinee2010.plot(label='DESTINEE 2010')
destinee2050.plot(label='DESTINEE 2050')
plt.title('Hourly UK Historic, Electric Heat and DESTINEE')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# convert to daily
daily_historic = historic.resample('D').sum()
daily_mine = mine.resample('D').sum()
daily_destinee2010 = destinee2010.resample('D').sum()
daily_destinee2050 = destinee2050.resample('D').sum()

# daily  plot of historic and electric
daily_historic.plot(label='Daily Electricity Demand 2010')
daily_mine.plot(label='Historic with my 50% heat pumps 2010')
daily_destinee2010.plot(label='DESTINEE 2010')
daily_destinee2050.plot(label='DESTINEE 2050')
plt.title('Daily UK Historic, Electric Heat and DESTINEE')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# read destinee electric heat only
destinee_filename = '/home/malcolm/uclan/data/destinee/heat.csv'
destinee_heat = readers.read_destinee(destinee_filename)
print(destinee_heat)
# convert from TJ to Kwh then to GWh
# destinee_electric_heat = destinee_heat['electric'] * 277777.77777778 * 1e-6
# convert from TWh to GWh
destinee_electric_heat = destinee_heat['electric'] * 1e+3

print('Electric heat annual mine: {} destinee: {}'.format(heat.sum(), destinee_electric_heat.sum() ) )

daily_heat = heat.resample('D').sum()
daily_destinee_electric_heat = destinee_electric_heat.resample('D').sum()

# daily  plot of historic and electric
daily_heat.plot(label='BDEW')
daily_destinee_electric_heat.plot(label='DESTINEE')
plt.title('Daily UK Electric Heat 2010 BDEW and DESTINEE')
plt.xlabel('Day of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()
