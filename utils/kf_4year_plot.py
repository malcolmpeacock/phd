# plot electricity demand time series using the baselines for the actual years
# 2016 to 2018 rather than the 2018 baseline.

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

def get_demand(year):
    demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
    demand = readers.read_espeni(demand_filename, year)
    electric = demand / 1000000.0
    return electric

def get_heat_bdew(year):
    ref_year = 2018
    # read heat demand.
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-Brhpp.csv'.format(year,ref_year)
    heat_demand = readers.read_copheat(demand_filename, ['heat', 'electricity'])
    # 41% heat pumps
    electric_41hp = heat_demand['electricity'] * 1e-6 * 0.41
    
    return electric_41hp

def get_ref_baseline(ref_baseline, year, ref_index):
#   ref_index = pd.Series(dtype='float64')
    
    year_baseline = storage.ref_baseline(ref_baseline, year, ref_index, False, True)
    return year_baseline

def get_baseline(year):
    # input assumptions for amount of electricity time series that is heat
    heat_thats_electric = {'2018' : 0.06, '2017' : 0.06, '2016' : 0.06 }
    #electrical_heat = {'2018' : 40.9, '2017' : 40.2, '2016' : 42.7 }

    # read heat demand.
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew.csv'.format(year)
    heat_demand = readers.read_copheat(demand_filename, ['heat', 'electricity'])
    # 41% heat pumps
    electric_41hp = heat_demand['electricity'] * 1e-6 * 0.41

    # read resistive heat.
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew_resistive.csv'.format(year)
    heat_resistive = readers.read_copheat(demand_filename, ['heat', 'electricity'])
    electric_existing = heat_resistive['electricity'] * 1e-6 * heat_thats_electric[year]

    return electric_existing, electric_41hp


# process command line
parser = argparse.ArgumentParser(description='Plot 3 years with historic and 41% hp using own baselines.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Includes plots', default=False)
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
args = parser.parse_args()

year = 2018
# get heating electricity - existing and 41% heat pumps
heat_electric, electric_hp41 = get_baseline(str(year))

# get historic electricity demand
electric_ref = get_demand(str(year) )

# calculate baseline
ref_baseline = electric_ref - heat_electric

years = [2016, 2017, 2018, 2019]
baselines = {}
historics = {}
news = {}
existings = {}
hp_hourly = {}

for year in years:
    print('year {}'.format(year))
    # get heating electricity - existing and 41% heat pumps
    electric_hp41 = get_heat_bdew(str(year))

    # get historic electricity demand
    electric_ref = get_demand(str(year) )

    # calculate baseline
#   baseline = electric_ref - heat_electric

    # add heat for heat pumps
#   new_electric = baseline + electric_hp41

    # store
#   baselines[year] = baseline.resample('D').sum()
#   news[year] = new_electric.resample('D').sum()
    historics[year] = electric_ref.resample('D').sum()
#   existings[year] = heat_electric.resample('D').sum()

    hp_hourly[year] = electric_hp41

#baseline = pd.concat([baselines[year] for year in years])
#with41hp = pd.concat([news[year] for year in years])
historic = pd.concat([historics[year] for year in years])
#heating = pd.concat([existings[year] for year in years])


ref_news = {}
for year in years:
    print('year {}'.format(year))

    # add heat for heat pumps
    if year==2018:
        new_electric = ref_baseline + hp_hourly[year]
    else:
        year_baseline = get_ref_baseline(ref_baseline, year, hp_hourly[year].index)
        new_electric = year_baseline + hp_hourly[year]

    ref_news[year] = new_electric.resample('D').sum()

ref_41hp = pd.concat([ref_news[year] for year in years])

# figure (6) but 3 years with the 2018 base line ( only include 41% )
ref_41hp.rolling(5, min_periods=1).mean().plot(label='Electricity demand including 41% heating provided by heat pumps', color='green')
historic.rolling(5, min_periods=1).mean().plot(label='Historic electricity demand time series', color='blue')
#plt.title('Impact of heat pumps on 2018 daily electricity demand (3 years 2018 baseline)')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper left')
plt.show()
