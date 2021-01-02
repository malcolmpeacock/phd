# Using the assumptions of Hooker Stroud:
#  for EVs, insulation and change in annual electricity demand
#  and electric heat based on the weather of 2017 from my program
# create a 2050 electricity series

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
#  Reference Year = 2018
#  Weather Year   = 2017

scotland_factor = 1.1    # ( Fragaki et. al )

# read 2018 historical electricity demand

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand18 = readers.read_electric_hourly(demand_filename)
electric18 = demand18['ENGLAND_WALES_DEMAND'] * scotland_factor

# read 2018 electric heat series based on purely resistive heating so that
# it can be removed from the 2018 series. (ref year method)

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather2018I-Sbdew_resistive.csv'
resistive_heat = readers.read_demand(demand_filename)

# read 2017 electric heat for ref year 2018
#demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2017/GBRef2018Weather2017I-Sbdew.csv'
# read 2017 electric heat for ref year 2050
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2017/GBRef2050Weather2017I-Sbdew.csv'
heat17 = readers.read_demand(demand_filename)

heat_that_is_electric = 0.06     # my spreadsheet from DUKES

# REFERENCE YEAR METHOD

#  Factor to remove existing space and water heating
#  from the electricity demand time 
#  series for the reference year - subtract the 2018 resistive heat series

#  heat_that_is_electric = 6%
#  heat_that_is_heat_pumps = 1% ( greenmatch.co.uk, renewableenergyhub.co.uk)
#  so 1/6th of electric is heat pumps, rest = resistive.
#  
#  electic_that_is_heat = ?
#  electric18_current = output of heat prog with above values.

# reference year series - ref electric heat + weather year heat.
# mod_electric17 = electric18.values - (resistive_heat.values * heat_that_is_electric)
# ref_method = mod_electric17 + heat17.values
#  heat demand and data for daily and monthly plots

heat_in17 = resistive_heat * heat_that_is_electric
mod_electric17 = electric18 - heat_in17
mod_electric17.index = heat17.index
ref_method = mod_electric17 + heat17

#  Factor to modify the other heat (industry etc) from the reference year
#  to the weather year. DESTinEE assumes its constant, so nothing to do.

#  annual 2017
annual2017 = ref_method.sum()
print('Annual 2017 {}'.format(annual2017))
#  from Hooker Stroud industry + appliances.
annual2050 = 271000000
# scale to 2050
ref_method = ref_method * ( annual2050 / annual2017 )
# electric transport strange profile at night and peak
ev = ref_method * 0.0
for i in range(0,len(ev)):
    if i%24 == 1:
        ev[i] = 0.2
    if i%24 == 2:
        ev[i] = 0.2
    if i%24 == 3:
        ev[i] = 0.2
    if i%24 == 14:
        ev[i] = 0.1
    if i%24 == 15:
        ev[i] = 0.1
    if i%24 == 16:
        ev[i] = 0.1
    if i%24 == 17:
        ev[i] = 0.1
#  42TWh daily because the above is an hourly profile for the day
ev = ev * (42000000.0 / 365)
print(ev)
ref_method = ref_method + ev


#  plots of daily

day_elec_heat17 = heat17.resample('D').sum()
day_elec_hist17 = mod_electric17.resample('D').sum()
day_combined17 = ref_method.resample('D').sum()
day_ev = ev.resample('D').sum()

# heat, electric heat and historic

day_elec_hist17.plot(color='blue', label='Historic Electric 2018')
day_elec_heat17.plot(color='red', label='2050 Electric Heat with 2017 weather')
day_combined17.plot(color='green', label='2050 Electricity demand')
day_ev.plot(color='yellow', label='2050 Electric Transport')
plt.title('Daily Electricity for 2050 with 2017 weather')
plt.xlabel('Month', fontsize=15)
plt.ylabel('Energy (Mwh) per day', fontsize=15)
plt.legend(loc='upper right')
plt.show()
