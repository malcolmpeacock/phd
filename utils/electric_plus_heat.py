# add electric heat demand to historical demand

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

# read 2017 historical electricity demand
demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2017.csv'
demand17 = readers.read_electric_hourly(demand_filename)
electric17 = demand17['ENGLAND_WALES_DEMAND'] * scotland_factor

# read 2018 historical electricity demand

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand18 = readers.read_electric_hourly(demand_filename)
electric18 = demand18['ENGLAND_WALES_DEMAND'] * scotland_factor

# read 2018 electric heat series based on purely resistive heating so that
# it can be removed from the 2018 series. (ref year method)

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather2018I-Sbdew_resistive.csv'
resistive_heat = readers.read_demand(demand_filename)

# read 2017 electric heat for ref year 2018

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2017/GBRef2018Weather2017I-Sbdew.csv'
heat17 = readers.read_demand(demand_filename)

heat_that_is_electric = 0.06     # my spreadsheet from DUKES

# USE HISTORIC ELECTRICITY GENERATION FOR EACH YEAR

# add in a constant value to the 2017 electricity time series based on 
# the different between the totals of 2017 and 2018.

hist_factor = (electric18.sum() - electric17.sum()) / (365.0 * 24.0)
print('hist_factor {}'.format(hist_factor))
hist_method = electric17.values + hist_factor

# add in the 2017 electric heat
hist_method = hist_method + (heat17.values * ( 1 - heat_that_is_electric) )

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
mod_electric17 = electric18.values - (resistive_heat.values * heat_that_is_electric)
ref_method = mod_electric17 + heat17.values

#  Factor to modify the other heat (industry etc) from the reference year
#  to the weather year. DESTinEE assumes its constant, so nothing to do.

#  plot of historic and electric
electric17.plot(label='Electricity Demand 2017')
electric18.plot(label='Electricity Demand 2018')
heat17.plot(label='Electric Heat for 2017 weather')
resistive_heat.plot(label='Heat part of 2018 demand')
plt.title('GB Historic and Electric Heat')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

#  plot of combined
plt.plot(ref_method,label='Reference year method')
plt.plot(hist_method,label='Historic year method')
plt.title('GB Addition of Electric Heat')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

#  heat demand and data for daily and monthly plots

cop_heat17 = readers.read_copheat(demand_filename,parms=['heat','temperature','ASHP_radiator','ASHP_water'])
# mod_electric17 = electric18.values - (resistive_heat.values * heat_that_is_electric)
# mod_electric17 = electric18.values - (resistive_heat.values * heat_that_is_electric)
print('heat_that_is_electric {}'.format(heat_that_is_electric))
print('resistive_heat')
print(resistive_heat)
heat_in17 = resistive_heat * heat_that_is_electric
print('heat_in17')
print(heat_in17)
mod_electric17 = electric18 - heat_in17
print('mod_electric17')
mod_electric17.index = heat17.index
print(mod_electric17)
print('ref_method')
ref_method = mod_electric17 + heat17
print(ref_method)


# sink temperature for radiator
sink_radiator = 40.0
deltat = cop_heat17['temperature'] * -1.0
deltat = deltat + sink_radiator

#  plots of n days

days_17 = cop_heat17['2017-03-01 00:00:00' : '2017-03-04 23:00:00' ]
print('days_17')
print(days_17)
days_elec_heat17 = heat17['2017-03-01 00:00:00' : '2017-03-04 23:00:00' ]
print('days_elec_heat17')
print(days_elec_heat17)
days_elec_hist17 = mod_electric17['2017-03-01 00:00:00' : '2017-03-04 23:00:00' ]
print('days_elec_hist17')
print(days_elec_hist17)
days_combined17 = ref_method['2017-03-01 00:00:00' : '2017-03-04 23:00:00' ]
print('days_combined17')
print(days_combined17)
days_deltat = deltat['2017-03-01 00:00:00' : '2017-03-04 23:00:00' ]

# temp and cop

ax = days_17['temperature'].plot(color='blue',label='Air temperature')
days_deltat.plot(ax=ax,color='green',label='Delta T')
plt.ylabel('Temperature (degrees C)', fontsize=15, color='blue')
plt.legend(loc='upper right')
ax2 = ax.twinx()
ax2.set_ylabel('COP',color='red', fontsize=15)
days_17['ASHP_radiator'].plot(ax=ax2,color='red')
plt.title('Temperature and COP March 2017')
plt.xlabel('Hour', fontsize=15)
plt.show()

# heat, electric heat and historic

days_elec_hist17.plot(color='blue', label='Historic Electric')
days_elec_heat17.plot(color='red', label='Elecric Heat')
days_17['heat'].plot(color='orange', label='Heat demand')
days_combined17.plot(color='green', label='Combined historic and electric heat')
plt.title('Electricity and heat March 2017')
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Energy (Mwh)', fontsize=15)
plt.legend(loc='upper right')
plt.show()


#  plots of daily

day_temp = cop_heat17['temperature'].resample('D').mean()
day_heat = cop_heat17['heat'].resample('D').sum()
day_ashp = cop_heat17['ASHP_radiator'].resample('D').mean()
day_deltat = deltat.resample('D').mean()
day_elec_heat17 = heat17.resample('D').sum()
day_elec_hist17 = mod_electric17.resample('D').sum()
day_combined17 = ref_method.resample('D').sum()

# temp and cop - daily

ax = day_temp.plot(color='blue',label='Air temperature')
day_deltat.plot(ax=ax,color='green',label='Delta T')
plt.ylabel('Temperature (degrees C)', fontsize=15, color='blue')
plt.legend(loc='upper right')
ax2 = ax.twinx()
ax2.set_ylabel('COP',color='red', fontsize=15)
day_ashp.plot(ax=ax2,color='red')
plt.title('Daily Temperature and COP')
plt.xlabel('Hour', fontsize=15)
plt.show()

# heat, electric heat and historic

day_elec_hist17.plot(color='blue', label='Historic Electric')
day_elec_heat17.plot(color='red', label='Electric Heat')
day_heat.plot(color='orange', label='Heat demand')
day_combined17.plot(color='green', label='Combined historic and electric heat')
plt.title('Daily Electricity and heat 2017')
plt.xlabel('Month', fontsize=15)
plt.ylabel('Energy (Mwh) per day', fontsize=15)
plt.legend(loc='upper right')
plt.show()

#  daily plot of historic electric vs temperature
plt.scatter(day_elec_hist17,day_temp,s=12)
plt.title('Historic Electric vs temperature')
plt.xlabel('Historic Electric (Mwh) per day', fontsize=15)
plt.ylabel('Temperature (degrees C)', fontsize=15)
plt.show()

#  daily plot of electric heat vs temperature
plt.scatter(day_elec_heat17,day_temp,s=12)
plt.title('Electric heat vs temperature')
plt.xlabel('Electric Heat (Mwh) per day', fontsize=15)
plt.ylabel('Temperature (degrees C)', fontsize=15)
plt.show()

#  correlation
negative_temp = day_temp * -1.0
stats.print_stats_header()
stats.print_stats(day_elec_heat17, negative_temp, 'Electric heat with temperature')
stats.print_stats(day_elec_hist17, negative_temp, 'Historic Electric with temperature')
stats.print_stats(day_combined17, negative_temp, 'Combined Historic and heat with temperature')

#  plots of monthy

month_temp = cop_heat17['temperature'].resample('M').mean()
month_heat = cop_heat17['heat'].resample('M').sum()
month_ashp = cop_heat17['ASHP_radiator'].resample('M').mean()
month_deltat = deltat.resample('M').mean()
month_elec_heat17 = heat17.resample('M').sum()
month_elec_hist17 = mod_electric17.resample('M').sum()
month_combined17 = ref_method.resample('M').sum()

# temp and cop

ax = month_temp.plot(color='blue',label='Air temperature')
month_deltat.plot(ax=ax,color='green',label='Delta T')
plt.ylabel('Temperature (degrees C)', fontsize=15, color='blue')
plt.legend(loc='upper right')
ax2 = ax.twinx()
ax2.set_ylabel('COP',color='red', fontsize=15)
month_ashp.plot(ax=ax2,color='red')
plt.title('Mohtly Temperature and COP')
plt.xlabel('Hour', fontsize=15)
plt.show()

# heat, electric heat and historic

month_elec_hist17.plot(color='blue', label='Historic Electric')
month_elec_heat17.plot(color='red', label='Electric Heat')
month_heat.plot(color='orange', label='Heat demand')
month_combined17.plot(color='green', label='Combined historic and electric heat')
plt.title('Mohtly Electricity and heat 2017')
plt.xlabel('Month', fontsize=15)
plt.ylabel('Energy (Mwh) per month', fontsize=15)
plt.legend(loc='upper right')
plt.show()
