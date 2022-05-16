# compare the baselines electricity demands using different reference year
# heat demands

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pytz
import pvlib
import calendar
import argparse

# custom code
import stats
import readers
import storage

def get_demand(year, espini):
    if espini:
        demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
        demand = readers.read_espeni(demand_filename, year)
        electric = demand / 1000000.0
    else:
        # TODO - could we include the actual Scottish demand here?
        scotland_factor = 1.1
        # read historical electricity demand for reference year
        if year=='2009':
            demand_filename = '/home/malcolm/uclan/data/electricity/demanddata_2009.csv'
        else:
            demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'
        demand_ref = readers.read_electric_hourly(demand_filename)
        # Convert to TWh
        electric = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor / 1000000.0
    return electric

def get_heat_bdew(year):
    # input assumptions TODO 2016
    electrical_heat = {'2018' : 40.9, '2017' : 40.2, '2016' : 42.7 }
    # read heat demand.
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew.csv'.format(year)
    heat_demand = readers.read_copheat(demand_filename, ['heat', 'temperature'])
    # convert to GWh ?
    ref_resistive_heat = heat_demand['heat'] * 1e-6
    existing_heat = ref_resistive_heat * electrical_heat[year] / ref_resistive_heat.sum()
    return existing_heat

def get_baseline(year, espini, ref_index=pd.Series(dtype='float64')):

#   print('Year {}'.format(year))
    # electricity demand
    electric_ref = get_demand(str(year), espini)
#   print('DEMAND')
#   print(electric_ref)

    # heat demand
    heat_electric = get_heat_bdew(str(year))
#   print('HEAT')
#   print(heat_electric)

    # remove heating electricity to get baseline
    baseline = electric_ref - heat_electric

    # shift days to match 2018 (if its not 2018! )
    if len(ref_index) == 0:
        ref_index = electric_ref.index
        year_baseline = baseline
    else:
        year_baseline = storage.year_baseline(baseline, year, ref_index, args.shift, True)
        electric_ref_values = storage.remove_feb29(electric_ref, year, True)
        electric_ref = pd.Series(data=electric_ref_values, index=ref_index)

    return year_baseline, electric_ref

# process command line
parser = argparse.ArgumentParser(description='Compare baselines generated for difference years.')
parser.add_argument('--espini', action="store_true", dest="espini", help='Use the espini data', default=True)
parser.add_argument('--plot', action="store_true", dest="plot", help='Includes plots', default=False)
parser.add_argument('--shift', action="store_true", dest="shift", help='Shift day pattern', default=False)
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
args = parser.parse_args()

years = [2016, 2017]
baselines = {}
historics = {}

# get 2018 baseline
ref_baseline, ref_electric = get_baseline(2018, args.espini)
#print('REF_BASELINE')
#print(ref_baseline)

stats.print_stats_header()
for year in years:

    year_baseline, year_electric = get_baseline(year, args.espini, ref_baseline.index)
    # compare each year to 2018
#   print('REF_BASELINEa')
#   print(ref_baseline)
#   print('YEAR_BASE')
#   print(year_baseline)
    stats.print_stats(ref_baseline, year_baseline, 'baseline ' + str(year))
    # compare each historic to 2018
    stats.print_stats(ref_electric, year_electric, 'historic ' + str(year))
    baselines[year] = year_baseline
    historics[year] = year_electric

if args.plot:
    ref_baseline.plot(label='2018 baseline')
    for year in years:
        baselines[year].plot(label='{} baseline'.format(year))
    plt.title('Hourly Baselines')
    plt.xlabel('hour of the year', fontsize=15)
    plt.ylabel('Demand (Twh)', fontsize=15)
    plt.legend(loc='upper center')
    plt.show()

    daily_ref = ref_baseline.resample('D').sum()
#   print(daily_ref.values[0:13])
    if args.rolling != 0:
        daily_ref = daily_ref.rolling(args.rolling, min_periods=1).mean()
    daily_ref.plot(label='2018 baseline')
    for year in years:
        daily_ref = baselines[year].resample('D').sum()
#       print(daily_ref.values[0:13])
        if args.rolling != 0:
            daily_ref = daily_ref.rolling(args.rolling, min_periods=1).mean()
        daily_ref.plot(label='{} baseline'.format(year))
    plt.title('Daily Baselines')
    plt.xlabel('day of the year', fontsize=15)
    plt.ylabel('Demand (Twh)', fontsize=15)
    plt.legend(loc='upper center')
    plt.show()

    # plot of all years 
    daily_baselines = {}
    daily_historics = {}
    for year in years:
        print('year {}'.format(year))
        year_baseline, year_electric = get_baseline(year, args.espini)
        daily_baselines[year] = year_baseline.resample('D').sum()
        daily_historics[year] = year_electric.resample('D').sum()
    daily_baselines[2018] = ref_baseline.resample('D').sum()
    daily_historics[2018] = ref_electric.resample('D').sum()
    all_years = [2016, 2017, 2018]
    baseline = pd.concat([daily_baselines[year] for year in all_years])
    print(baseline)
    historic = pd.concat([daily_historics[year] for year in all_years])
    baseline.plot(label='baseline', color='blue')
    historic.plot(label='historic', color='green')
    plt.title('Daily Baselines')
    plt.xlabel('day of the year', fontsize=15)
    plt.ylabel('Demand (Twh)', fontsize=15)
    plt.legend(loc='upper center')
    plt.show()
