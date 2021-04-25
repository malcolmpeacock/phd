# Create a 2050 electric time series from:
#  reference year heat demand
#  weather year heat demand
#  reference year electricity time series
# And assumption from various sources, such as FES 2019

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

# electric vehicle time series

def ev_series(temperature, annual_energy):
    daily_energy = annual_energy * 1000000 / 365.0
    # at 14.4 kWh/100 km, gives
    daily_range_km = daily_energy * 100.0 / 14.4
    #  42TWh annual ecause the above is an hourly profile for the day
    daytime_temp = 19.0
    new_daily_temp = 0.0
    ev = temperature * 0.0
    for i in range(0,len(ev)):
        hour = i%24
        if hour==1 or hour==2 or hour==3:
            ev[i] = 0.2
        if hour==14 or hour==15 or hour==16 or hour==17:
            ev[i] = 0.1
        if hour==23:
            daytime_temp = new_daily_temp / 12.0
            new_daily_temp = 0.0
        # calculate temperature during the day
        if hour > 7 or hour < 21:
            new_daily_temp += temperature.values[i]
        # calculate energy increase if below 10 or above 28
        increase = 0.0
        if daytime_temp < 10.0:
            increase = ( (2.4 * daily_range_km)/100.0 ) * (10.0 - daytime_temp) / 5
        if daytime_temp > 28.0:
            increase = ( (2.3 * daily_range_km)/100.0 ) * (daytime_temp - 28.0) / 5
#       print('Temp {} energy {} increase {}'.format(daytime_temp, daily_energy, increase) )
        energy = daily_energy + increase
        ev[i] = ev[i] * energy
    print(ev)
    return ev

# hybrid heat pump series

def hybrid_heat_pump(heat, efficiency, threshold):
    demand = heat.copy()
    total_heat = demand['heat'].sum()
    # sort by temperature
    demand = demand.sort_values('temperature')
    hydrogen = demand['electricity'] * 0.0
    # starting with coldest hour, allocate to gas and zero electric
    # until the % energy is reached is reached.
    # (this is based on FES 2019 assumption of % energy used so we find
    #  out what threshold temperature this would need )
    #
    irow=0
    for index, row in demand.iterrows():
        # divide be efficiency because we use more primary gas energy than heat.
        hydrogen.iloc[irow] = row['heat'] / efficiency
        demand.iloc[irow,demand.columns.get_loc('electricity')] = 0.0
        threshold_temperature = row['temperature']
        if row['temperature'] > threshold:
            break
        irow+=1

    # gas = heat * 0.85
    # electric = 0
    demand = demand.sort_index()
    hydrogen = hydrogen.sort_index()
    print(demand)
    print(hydrogen)
    # output the temperature - this is the threshold temperature.

    return demand['electricity'], hydrogen

def hydrogen_boiler(heat):

    return hydrogen

# main program
#  Reference Year = 2018
#  Weather Year   = 2017

# program options
plot = True
reference_year = '2018'
weather_year = '2017'
# input assumptions for reference year
heat_that_is_electric = 0.06     # my spreadsheet from DUKES
heat_that_is_heat_pumps = 0.01   # greenmatch.co.uk, renewableenergyhub.co.uk

# input assumptions 2050
percent_hydrogen_boiler = 0.45   # fes 2019 Net Zero 2050
percent_heat_pump = 0.28         # fes 2019 Net Zero 2050
percent_hybrid_pump = 0.13       # fes 2019 Net Zero 2050
percent_other = 0.14             # fes 2019 Net Zero 2050
boiler_efficiency = 0.85         # hydrogen boiler efficiency
ev_annual_energy = 90            # annual ev energy TWh fes Net Zero 2050
electricity_residential = 109    # fes 2019 Net Zero 2050
electricity_industry = 175       # fes 2019 Net Zero 2050
hybrid_threshold = 5.0           # temperature below which hybrid heat pumps
                                 # switch completely to hydrogen to balance
                                 # the grid

# weather year

scotland_factor = 1.1    # ( Fragaki et. al )

# read historical electricity demand for reference year

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + reference_year + '.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_ref = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor

# read reference year electric heat series based on purely resistive heating
# so that it can be removed from the reference year series. 

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef2018Weather{0:}I-Sbdew_resistive.csv'.format(reference_year) 
resistive_heat = readers.read_demand(demand_filename)

# read 2017 electric heat for ref year 2050
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef2050Weather{0:}I-Sbdew.csv'.format(weather_year)
demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])

heat_weather = demand['heat']

#  To remove existing space and water heating from the electricity demand time 
#  series for the reference year - subtract the resistive heat series

heat_in_weather_year = resistive_heat * heat_that_is_electric
mod_electric17 = electric_ref - heat_in_weather_year
mod_electric17.index = heat_weather.index
# TODO - this should be electric ?
ref_method = mod_electric17 + heat_weather

#  Factor to modify the other heat (industry etc) from the reference year
#  to the weather year. DESTinEE assumes its constant, so nothing to do.

#  annual 2017
annual2017 = ref_method.sum()
print('Annual 2017 {}'.format(annual2017))
#  from fes 2019
annual2050 = ( electricity_residential + electricity_industry ) * 1000000
# scale to 2050
ref_method = ref_method * ( annual2050 / annual2017 )
# electric transport strange profile at night and peak
ev = ev_series(demand['temperature'], ev_annual_energy)
print(ev)
if plot:
    ev.plot(color='blue', label='EV Electric 2018')
    plt.title('Hourly EV energy use')
    plt.xlabel('day', fontsize=15)
    plt.ylabel('Energy (Mwh) per day', fontsize=15)
    plt.show()

ref_method = ref_method + ev

#  plots of daily

day_elec_heat17 = heat_weather.resample('D').sum()
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
