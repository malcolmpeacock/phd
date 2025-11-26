# Program to generate electricity demand files with varying amounts of heat 
# pumps or EVs
# Given the folling input:
#  - a range of years,
#  - % of heating by heat pumps.
#  - % of transport electrified.
#  - baseline file
# Creates an electricity demand time series.

# library stuff
import sys
import os
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy.stats import wasserstein_distance

# custom code
import stats
import readers
import storage
import math

# electric vehicle time series
ev_annual_energy = 95.83  # annual ev energy TWh fes Net Zero 2050
ev_annual_energy = 93.38  # percentage adjusted for 2018 with all ev

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
#   print(ev)
#   print('EV Series. Temp len {} max {} min {} annual_energy {}'.format(len(temperature), temperature.max(), temperature.min(), annual_energy) )
    return ev

# create 40year demand
#
#  baseline        - reference year electricity with heat removed
#  heat_pump_share - percentage of heating from heat pumps
#  years           - list of years to do the analysis for
def demand(baseline, heat_pump_share, ev_share, years, existing):

    # create the synthetic years
    
    # ordinary year
    ordinary_year = baseline.values
    # leap year
    # create a feb 29th by interpolating between feb 28th and Mar 1st
    # find doy for feb 28th ( 31 days in jan )
    feb28 = 31 + 28
    feb28 = baseline['2018-02-28'].values
    mar1 = baseline['2018-03-01'].values
    feb29 = np.add(feb28, mar1) * 0.5
    leap_year = np.concatenate([baseline['2018-01-01 00:00' : '2018-02-28 23:00'].values, feb29,  baseline['2018-03-01 00:00' : '2018-12-31 23:00'].values])

    demand_years=[]
    baseline_years=[]

    # for each weather year ...
    for year in years:

        # B    - the BDEW method
        # rhpp - the rhpp hourly (heat pump) profile
        file_base = 'Brhpp'
        demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(year, args.reference, file_base)
        demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
        #  account for leap years.
        #  Need to create a new df with the index same as heat_weather
        #  then create a 29th of Feb by interpolation between 28th and
        #  1st March and then set the data values into empty DF
        hourly_weather = demand['temperature']
        heat_weather = demand['heat'] * 1e-6 * args.heat_that_is_electric
        heat_electric = demand['electricity'] * 1e-6
#       print(heat_weather)

        # use leap year data or ordinary year data
        if calendar.isleap(year):
            year_values = leap_year
            print('Leap Year')
        else:
            year_values = ordinary_year
            print('Not Leap Year')

        # Convert to a series
        electric_ref = pd.Series(year_values, index=heat_weather.index)

        print('Year {} total heat demand {:.2f} baseline total {:.2f} total electric heat {:.2f}'.format(year, heat_weather.sum(), electric_ref.sum(), heat_electric.sum() ) )

        # add to the list
        baseline_years.append( electric_ref )

        # Add in existing 
        if existing:
            electric_ref = electric_ref + heat_weather

        # Add in heat pumps
        heat_pumps = heat_electric * heat_pump_share
        electric_ref = electric_ref + heat_pumps

        # electric transport charging
        hourly_ev = ev_series(hourly_weather, ev_annual_energy)
        hourly_ev = hourly_ev * ev_share * 1e-6
        electric_ref = electric_ref + hourly_ev

        print('total demand {:.2f} heat_pumps total {:.2f} ev total {:.2f}'.format(electric_ref.sum(), heat_pumps.sum(), hourly_ev.sum() ) )

        # add to the list
        demand_years.append( electric_ref )

    # concantonate the demand series
    all_demand = pd.concat(demand_years[year] for year in range(len(years)) )
    all_baseline = pd.concat(baseline_years[year] for year in range(len(years)) )
    #
    print('Average demand over all years {}'.format(all_demand.sum() / len(years) ) )

    return all_demand, all_baseline
   
# main program

# process command line
parser = argparse.ArgumentParser(description='Show the impact of heat pumps or hydrogen on different shares of wind and solar')
# Journal heat paper was 0.06 but thesis was 0.09
parser.add_argument('--heat_electric', action="store", dest="heat_that_is_electric", help='Proportion of heat in electricity demand, default=0.06', type=float, default=0.06)
parser.add_argument('--start', action="store", dest="start", help='Start Year', type=int, default=2017 )
parser.add_argument('--end', action="store", dest="end", help='End Year', type=int, default=2019 )
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
parser.add_argument('--baseline', action="store", dest="baseline", help='Baseline filename', default='base2018' )
parser.add_argument('--dir', action="store", dest="dir", help='Demand directory', default='demand' )
parser.add_argument('--filename', action="store", dest="filename", help='Demand filename', default='3years_hp41' )
parser.add_argument('--existing', action="store_true", dest="existing", help='Add in heating assuming 2018 technology with 2018 portion of heat', default=False)
parser.add_argument('--hp_share', action="store", dest="hp_share", help='Share of heating provided by heat pumps.', type=float, default=0.41)
parser.add_argument('--ev_share', action="store", dest="ev_share", help='Share of transport electrified.', type=float, default=0.0)
parser.add_argument('--ev_profile', action="store", dest="ev_profile", help='EV profile', default='old' )

args = parser.parse_args()

print('Inputs: existing {} heat pumps {} ev {} baseline {} filename {} directory {}'.format(args.existing, args.hp_share, args.ev_share, args.baseline, args.filename, args.dir))

output_dir = "/home/malcolm/uclan/output/new/" + args.dir + '/'
if not os.path.isdir(output_dir):
    print('Error output dir {} does not exist'.format(output_dir))
    quit()
    

# read baseline demand for reference year
baseline_filename = '/home/malcolm/uclan/output/new/baseline/{}.csv'.format(args.baseline)
baseline = readers.read_demand(baseline_filename, parm='demand_twh')

print('Baseline demand: total {} number of values {}'.format(baseline.sum(), len(baseline) ) )

last_weather_year = args.end
years = range(args.start, last_weather_year+1)

electricity_demand, all_baseline = demand(baseline, args.hp_share, args.ev_share, years, args.existing)

# output the demand file
output_file = output_dir + args.filename + '.csv'
electricity_demand.to_csv(output_file, index_label='time', header=['demand_twh'])

# output the baseline file
output_file = output_dir + args.filename + '.baseline.csv'
all_baseline.to_csv(output_file, index_label='time', header=['demand_twh'])

# output settings file
settings = {
    'start'          : args.start,
    'end'            : args.end,
    'baseline'       : args.baseline,
    'existing'       : args.existing,
    'heat_electric'  : args.heat_that_is_electric,
    'reference'      : args.reference,
    'hp_share'       : args.hp_share,
    'ev_share'       : args.ev_share,
    'ev_profile'     : args.ev_profile
}
settings_df = pd.DataFrame.from_dict(data=settings, orient='index')
output_file = output_dir + args.filename + '.settings.csv'
settings_df.to_csv(output_file, header=False)
