# python script to plot electricity time series from methods TODO
# used for plot in my IECSF20 paper presentation.

import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np

# custom scripts
import utils.readers as readers
import heat.scripts.read as read
import heat.scripts.plot as plots

# convert heat time series to hourly using profile
# TODO note that the paper mentions that bdw don't have seperate space and
#      water heating profile and therefore water heating is subtracted 
#      from the total to give space. - i have not done that here.
#      anyway it would be better to have specific water and space profiles.
def heat2hourly(daily_heat, building_type, percentage, daily_temp, hourly_parameters):
    print('Daily heat total: {} '.format(daily_heat.sum()) )
    # reset last index value so hour is 23:00 to get whole year after 
    # resample
    year = daily_heat.first_valid_index().date().year
    daily_heat.index.values[-1] = pd.Timestamp(str(year) + '-12-31 23:00:00')
    # create an hourly heat series ( mean() just replaced by values )
    hourly_heat = daily_heat.resample('60min').mean()
    parms = hourly_parameters[building_type]
#   print(parms)
    count=0
    for day, heat in daily_heat.iteritems():
        day_str = day.strftime('%Y-%m-%d')
#       print(heat, day, day_str)
#       print(type(day))
        if building_type == 'COM':
            dayofweek = day.strftime('%w')
            profile = parms.loc[int(dayofweek)]
            profile = profile[str(daily_temp[count])]
#           print(profile)
        else:
            profile = parms[str(daily_temp[count])]
        day_values = profile.values * heat
#       print(day_values)
        hourly_heat.loc[day_str] = day_values
    print('Hourly heat total: {} '.format(hourly_heat.sum()) )
    return hourly_heat * percentage

# main program
parser = argparse.ArgumentParser(description='Create electric heat series from gas.')
parser.add_argument('--year', action="store", dest="year", help='year', type=int, default=2018)
parser.add_argument('--profile', action="store", dest="profile", help='Hourly profile file name', default='bdew')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

year = str(args.year)

# read 2018 heat demand to get hourly COP time series and daily temp

hdd155_filename = "/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather2018I-Sbdew.csv"
heat = readers.read_copheat(hdd155_filename, ['ASHP_floor','ASHP_radiator','ASHP_water','GSHP_floor','GSHP_radiator','GSHP_water','WSHP_floor','WSHP_radiator','WSHP_water', 'temperature'])
daily_temp = heat['temperature'].resample('D', axis=0).mean()
daily_temp = daily_temp.clip(lower=-15, upper=30)
print(daily_temp)
search_temp = (np.floor(daily_temp.values - 5.0) * 5).astype(int)
print(search_temp)

# read historic gas demand
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + year + '.csv'
gas = readers.read_gas(gas_filename)
gas.index = gas.index.tz_localize('UTC')

# read the distribution of heating and pumps etc
# Read in combination of ASHP GSHP etc.

heating_path = "/home/malcolm/uclan/tools/python/scripts/heat/input"
parm_heat, parm_water = read.electric_parameters(heating_path, 'GB')
print(parm_heat)
print(parm_water)

# hourly profile
hourly_parameters = read.hourly_parameters(heating_path, args.profile)

# plot hourly profiles to check
if args.plot:
    plots.hourly_profile(hourly_parameters)


# Convert gas energy from kWh to TWh
gas = gas * (10 ** -9)

# 
# scale gas by 0.8 to convert to heat
gas = gas * 0.8

# scale by the annual demand as per the paper
domestic_space = 191
domestic_water = 56
commercial_space = 70
commercial_water = 9
non_heat = 166
total_heat = domestic_space + domestic_water + commercial_space + commercial_water
print('Total gas heat {} total annual {}'.format(gas.sum(), total_heat) )
# split half equally between days.
even_part = (gas.sum() - total_heat)  / ( 365.0 * 2.0 )
gas = gas - even_part
# scale the rest.
gas = gas * (total_heat + non_heat) / gas.sum()
print('Total gas heat {} total annual {}'.format(gas.sum(), total_heat) )

# create daily gas-heat series for domestic/commercial, space/water
heat_domestic_space = gas * (domestic_space / total_heat )
heat_domestic_water = gas * (domestic_water / total_heat )
heat_commercial_space = gas * (commercial_space / total_heat )
heat_commercial_water = gas * (commercial_water / total_heat )
# convert to hourly via either the flat or BDEW profile.
hourly_domestic_space = heat2hourly(heat_domestic_space, 'SFH', 0.7, search_temp, hourly_parameters) + heat2hourly(heat_domestic_space, 'MFH', 0.3, search_temp, hourly_parameters)
hourly_commercial_space = heat2hourly(heat_commercial_space, 'COM', 1.0, search_temp, hourly_parameters)
hourly_domestic_water = heat2hourly(heat_domestic_water, 'SFH', 0.7, search_temp, hourly_parameters) + heat2hourly(heat_domestic_water, 'MFH', 0.3, search_temp, hourly_parameters)
hourly_commercial_water = heat2hourly(heat_commercial_water, 'COM', 1.0, search_temp, hourly_parameters)

hourly_space = hourly_domestic_space + hourly_commercial_space
hourly_water = hourly_domestic_water + hourly_commercial_water

hp_keys = { 'ground' : 'GSHP', 'air' : 'ASHP', 'water' : 'WSHP' }

# create space heating electric series
hourly_cop = hourly_space * 0.0
for key,value in parm_heat.items():
    print(key)
    print(value)
    for sink,percent in value.items():
        if key == 'resistive':
            print(sink,percent,'resistive')
            hourly_cop = hourly_cop + percent
        else:
            cop_column = hp_keys[key] + '_' + sink
            print(sink,percent,cop_column)
            hourly_cop = hourly_cop + (heat[cop_column] * percent)
print('Space hourly cop mean {}'.format(hourly_cop.mean() ) )
power_space = hourly_space / hourly_cop
print(power_space)

# create water heating electric series
hourly_cop = hourly_water * 0.0
for key,percent in parm_water.items():
    print(key)
    print(percent)
    if key == 'resistive':
        print(sink,percent,'resistive')
        hourly_cop = hourly_cop + percent
    else:
        cop_column = hp_keys[key] + '_' + sink
        print(sink,percent,cop_column)
        hourly_cop = hourly_cop + (heat[cop_column] * percent)
print('Water hourly cop mean {}'.format(hourly_cop.mean() ) )
power_water = hourly_water / hourly_cop
print(power_water)

# combine to create total gas electric heat
# ( 0.9 is real world efficiency adjustment )
power = (power_space + power_water ) / 0.9

print('Total electric heat {}'.format(power.sum() ) )

# output the gas heat hourly electric series.
heat['space'] = hourly_space
heat['water'] = hourly_water
heat['heat'] = hourly_water + hourly_space
heat['electricity'] = power * 1000.0 * 1000.0

print(heat)

output_dir = "/home/malcolm/uclan/tools/python/scripts/heat/output/" + year
output_filename = '{}/GBRef{}Weather{}I-G{}.csv'.format(output_dir, year, year, args.profile)
print(output_filename)
heat.to_csv(output_filename, sep=',', decimal='.', float_format='%g')
