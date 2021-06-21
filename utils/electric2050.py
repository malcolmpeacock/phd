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
import argparse

# custom code
import stats
import readers
from misc import upsample_df

# electric vehicle time series

def ev_series(temperature, annual_energy):
    daily_energy = annual_energy * 1000000 / 365.0
    # at 14.4 kWh/100 km, gives
    daily_range_km = daily_energy * 100.0 / 14.4
    # calculate ev profile
    daytime_temp = 19.0
    new_daily_temp = 0.0
    ev = temperature * 0.0
    for i in range(0,len(ev)):
        hour = i%24
        # night charging at 1am, 2am, 3am
        if hour==1 or hour==2 or hour==3:
            ev[i] = 0.2
        # peak daily charging
        if hour==14 or hour==15 or hour==16 or hour==17:
            ev[i] = 0.1
        # factor increased energy for exterme low or high temperatures
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

def hydrogen_boiler(heat, efficiency):
    hydrogen = heat / efficiency
    return hydrogen

# main program

# process command line

parser = argparse.ArgumentParser(description='Create future electric time series.')
parser.add_argument('--weather', action="store", dest="weather", help='Weather Year', default='2017' )
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
parser.add_argument('--model', action="store", dest="model", help='Model Year', default='2050' )
args = parser.parse_args()

# program options
plot = True
reference_year = args.reference
weather_year = args.weather
model_year = args.model

# input assumptions for reference year
heat_that_is_electric = 0.06     # my spreadsheet from DUKES
heat_that_is_heat_pumps = 0.01   # greenmatch.co.uk, renewableenergyhub.co.uk

model = {}
# input assumptions model year = 2050
d2050 = {}
# percentage of hydrogen boilers - fes 2019 Net Zero 2050
d2050['percent_hydrogen_boiler'] = 0.45 
d2050['percent_heat_pump']       = 0.28   # fes 2019 Net Zero 2050
d2050['percent_hybrid_pump']     = 0.13   # fes 2019 Net Zero 2050
d2050['percent_other']           = 0.14   # fes 2019 Net Zero 2050
d2050['boiler_efficiency']       = 0.85   # hydrogen boiler efficiency
d2050['ev_annual_energy']        = 95.83  # annual ev energy TWh fes Net Zero 2050
d2050['electricity_residential'] = 109    # fes 2019 Net Zero 2050
d2050['electricity_industry']    = 175       # fes 2019 Net Zero 2050
# temperature below which hybrid heat pumps switch completely to
# hydrogen to balance the grid
d2050['hybrid_threshold'] = 5.0       
model ['2050'] = d2050

# input assumptions model year = 2018 and same ev% as 2050
d2018 = {}
d2018['percent_hydrogen_boiler'] = 0.0    # 
d2018['percent_heat_pump'] = 0.28         # fes 2019 Net Zero 2050
d2018['percent_hybrid_pump'] = 0.13       # fes 2019 Net Zero 2050
d2018['percent_other'] = 0.14             # fes 2019 Net Zero 2050
d2018['boiler_efficiency'] = 0.85         # hydrogen boiler efficiency
d2018['ev_annual_energy'] = 93.38         # annual ev energy TWh fes Net Zero 2050
d2018['electricity_residential'] = 109    # fes 2019 Net Zero 2050
d2018['electricity_industry'] = 175       # fes 2019 Net Zero 2050
d2018['hybrid_threshold'] = 5.0           # temperature below which hybrid heat pumps
                                 # switch completely to hydrogen to balance
                                 # the grid
model ['2018'] = d2018

scotland_factor = 1.1    # ( Fragaki et. al )

# read historical electricity demand for reference year

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + reference_year + '.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_ref = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor

# read reference year electric heat series based on purely resistive heating
# so that it can be removed from the reference year series. 

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Sbdew_resistive.csv'.format(reference_year) 
ref_resistive_heat = readers.read_demand(demand_filename)

# read weather year electric heat for ref year 2050
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef2050Weather{0:}I-Sbdew.csv'.format(weather_year)
demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])

heat_weather = demand['heat']

#  To remove existing space and water heating from the electricity demand time 
#  series for the reference year - subtract the resistive heat series

mod_electric_ref = electric_ref - (ref_resistive_heat * heat_that_is_electric)
mod_electric_ref.index = heat_weather.index

#  annual demand of reference year with resistive heating removed.
annual_ref = mod_electric_ref.sum()
#  from fes 2019
annual_model = ( model[model_year]['electricity_residential'] + model[model_year]['electricity_industry'] ) * 1000000
# scale to 2050
print('Annual Ref {} 2050 {}'.format(annual_ref, annual_model))
ref_scaled = mod_electric_ref * ( annual_model / annual_ref )

# hybrid heat pumps

hybrid_electric, hybrid_hydrogen = hybrid_heat_pump(demand, model[model_year]['boiler_efficiency'], model[model_year]['hybrid_threshold'])

# hydrogen boilers
boiler_hydrogen = hydrogen_boiler(demand['heat'], model[model_year]['boiler_efficiency'])

# electric transport charging
ev = ev_series(demand['temperature'], model[model_year]['ev_annual_energy'])
print(ev)
if plot:
    ev.plot(color='blue', label='EV Electric {}'.format(weather_year))
    plt.title('Hourly EV energy use')
    plt.xlabel('day', fontsize=15)
    plt.ylabel('Energy (Mwh) per day', fontsize=15)
    plt.show()

# Form final electric series by adding together:
#   - the electric heat from the weather year (heat pumps)
#   - hybrid heat pumps
#   - the ev series
heat_electric = demand['electricity']*model[model_year]['percent_heat_pump'] + hybrid_electric*model[model_year]['percent_hybrid_pump']
final_electric = ref_scaled + heat_electric + ev

# Form the final hydrogen series by adding together:
#   - the hydrogen from boilers
#   - the hydrogen from hybrid heat pumps

final_hydrogen = boiler_hydrogen * model[model_year]['percent_hydrogen_boiler'] + hybrid_hydrogen * model[model_year]['percent_hybrid_pump']

#  plots of daily

daily_electric_ref = electric_ref.resample('D').sum()
daily_ref_scaled = ref_scaled.resample('D').sum()
daily_heat_pump = heat_electric.resample('D').sum()
daily_ev = ev.resample('D').sum()
daily_final = final_electric.resample('D').sum()

# change index to same as weather year for plotting
daily_electric_ref.index = daily_final.index

# plot daily

daily_electric_ref.plot(color='blue', label='Historic Electric {}'.format(reference_year))
daily_ref_scaled.plot(color='red', label='Historic Electric scaled to FES')
daily_heat_pump.plot(color='green', label='Electric heat pumps {} weather'.format(weather_year))
daily_ev.plot(color='yellow', label='Electric Transport {} weather'.format(weather_year))
daily_final.plot(color='purple', label='New combined electric series')
plt.title('Daily Electricity with electrification of heat and transport with {} weather'.format(weather_year))
plt.xlabel('Month', fontsize=15)
plt.ylabel('Energy (Mwh) per day', fontsize=15)
#plt.legend(loc='upper right')
plt.legend(loc='upper center')
plt.show()

# create another plot for KF
# historic 2018 electricity demand
daily_electric_ref = electric_ref.resample('D').sum()

# historic 2018 with the existing electric heat removed.
mod_electric_ref = electric_ref - (ref_resistive_heat * heat_that_is_electric)
daily_mod_electric_ref = mod_electric_ref.resample('D').sum()

# read reference year electric heat series based on all heat pumps
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef2018Weather{0:}I-Sbdew.csv'.format(reference_year) 
ref_actual_heat = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
electric_ref_heat = ref_actual_heat['electricity']
# historic electric with 100% heat pumps
daily_electric_ref_heat = electric_ref_heat.resample('D').sum()
print(daily_electric_ref_heat)
daily_hist_plus_hp = daily_mod_electric_ref + daily_electric_ref_heat

# change index to same as weather year for plotting
# EV charging based on same % of 2018 vehicles ( wierd I know! )
ev = ev_series(ref_actual_heat['temperature'], model['2018']['ev_annual_energy'])
daily_ev = ev.resample('D').sum()
#daily_ev.index = daily_electric_ref_heat.index
print(daily_electric_ref)

daily_historic_plus_heat_ev = daily_hist_plus_hp + daily_ev

daily_electric_ref.plot(color='blue', label='Historic Electricty 2018')
daily_hist_plus_hp.plot(color='red', label='Historic Electricity 2018 if all heating were with electric heat pumps')
daily_historic_plus_heat_ev.plot(color='green', label='Historic Electricity 2018 plus 100% heat pumps plus 98% Electric Vehicles')
plt.title('Daily Electricity for 2018 plus heat pumps and EVs')
plt.xlabel('Month', fontsize=15)
plt.ylabel('Electricity (Mwh) per day', fontsize=15)
#plt.legend(loc='upper right')
plt.legend(loc='upper center')
plt.show()

# read 1984 year electric heat for ref year 2018
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef2018Weather{0:}I-Bbdew.csv'.format('1983')
demand1984 = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
heat_electric_1984 = demand1984['electricity']
heat_electric_1984 = heat_electric_1984.resample('D').sum()

daily_mod_electric_ref.index = heat_electric_1984.index
combined1984 = heat_electric_1984 + daily_mod_electric_ref

daily_mod_electric_ref.plot(color='blue', label='Historic Electricty 2018 with heat removed')
heat_electric_1984.plot(color='yellow', label='Electricity for heat pumps based on 2018 annual demand with 1983 weather')
combined1984.plot(color='orange', label='Historic Electricty 2018 with heat removed and 1983 Heating Electricity for 1983 weather added')
plt.xlabel('Month', fontsize=15)
plt.ylabel('Electricity (Mwh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()
