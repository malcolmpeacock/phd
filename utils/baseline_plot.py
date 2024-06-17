# Plot the baseline electricity demand
# Print out stats about the baseline.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

def read_espeni(filename, year=None, cols=['ELEXM_utc', 'POWER_ESPENI_MW']):
    espini = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, usecols=cols).squeeze()
    # convert from half hourly to hourly
    hourly = espini.resample('H').sum() * 0.5
    hourly.index = pd.DatetimeIndex(pd.to_datetime(hourly.index.strftime("%Y-%m-%d %H") )).tz_localize('UTC')
    if year==None:
        return hourly
    else:
        return hourly.loc[year+'-01-01 00:00:00' : year + '-12-31 23:00:00']

def get_demand(year):
    demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
    demand = read_espeni(demand_filename, year)
    electric = demand / 1000000.0
    return electric

def read_copheat(filename, parms=['electricity']):
    demand = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, usecols=['time']+parms ).squeeze()
    return demand

# process command line
parser = argparse.ArgumentParser(description='Plot baseline for given year.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Includes plots', default=False)
parser.add_argument('--hourly', action="store_true", dest="hourly", help='Use hourly series', default=False)
parser.add_argument('--output', action="store_true", dest="output", help='Write out the baseline', default=False)
parser.add_argument('--year', action="store", dest="year", help='Year ', default='2018')
parser.add_argument('--heat', action="store", dest="heat", help='Proportion of electricity demand that is heating', type=float, default=0)
args = parser.parse_args()


year = args.year
electric_2018 = get_demand(year)

# input assumptions for reference year 2018 heating
# this includes:
#   domestic space 18
#   services space  9 (16 including industry)
#   domestic water  5
#   services water  2
#   industry space  7
#   total          40.94923
# ( industry is in the electricity time series, unlike the gas time series)
if args.heat == 0:
    heat_in_the_electricity_time_series = 40.94923
    heat_that_is_electric = heat_in_the_electricity_time_series / electric_2018.sum()
else:
    heat_that_is_electric = args.heat
print('heat_that_is_electric {}'.format(heat_that_is_electric) )

# electric_2018 = demand_ref['ENGLAND_WALES_DEMAND']
if args.hourly:
    daily_electric_2018 = electric_2018
else:
    daily_electric_2018 = electric_2018.resample('D').sum()

print('Historic demand {}: max {} min {} total {} '.format(year, electric_2018.max(), electric_2018.min(), electric_2018.sum() ) )

# read resistive heat for 2018

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef{}Weather{}I-Bbdew_resistive.csv'.format(year, year, year)
heat_demand2018 = read_copheat(demand_filename, ['electricity', 'temperature'])
# electricity demand if all heating were electric
resistive_heat_2018 = heat_demand2018['electricity'] * 1e-6

if args.heat == 0:
    heat_in_the_electricity_time_series = 40.94923
    heat_that_is_electric = heat_in_the_electricity_time_series / electric_2018.sum()
    heat_that_is_electric_2018 = electric_2018.sum() * 0.14 / resistive_heat_2018.sum()
else:
    heat_that_is_electric_2018 = args.heat
print('heat_that_is_electric {}'.format(heat_that_is_electric_2018) )

# get the portion of heat the is currently electric
electric2018_heat = resistive_heat_2018 * heat_that_is_electric_2018
print('resistive_heat_2018 min {} max {} sum {} heat_that_is_electric_2018 {}'.format(resistive_heat_2018.min(), resistive_heat_2018.max(), resistive_heat_2018.sum(), heat_that_is_electric_2018 ) )
electric2018_no_heat = electric_2018 - electric2018_heat
print('Hourly Baseline {} Original {} Heating {}'.format(electric2018_no_heat.values[0], electric_2018.values[0], electric2018_heat.values[0]) )

# plot 2018 electric, heat and difference. daily
if args.hourly:
    daily_electric2018_heat = electric2018_heat
    daily_electric2018_no_heat = electric2018_no_heat
else:
    daily_electric2018_heat = electric2018_heat.resample('D').sum()
    daily_electric2018_no_heat = electric2018_no_heat.resample('D').sum()
print('Baseline {} Original {} Heating {}'.format(daily_electric2018_no_heat.values[0], daily_electric_2018.values[0], daily_electric2018_heat.values[0]) )

daily_electric_2018.plot(color='blue', label='Historic electricity demand time series {}'.format(year) )
daily_electric2018_heat.plot(color='red', label='Electricty used for heating {}'.format(year))
daily_electric2018_no_heat.plot(color='purple', label='Electricity {} with heating electricity removed'.format(year))
plt.title('Removing existing heating electricity from the daily electricty demand series')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.show()

if args.output:
    output_name = "/home/malcolm/uclan/output/timeseries/baseline_{}.csv".format(year)
    daily_electric2018_no_heat.to_csv(output_name)

# stats
dayspm  = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
monthly = daily_electric2018_no_heat.resample('M').sum() 
mppd = np.divide(monthly.values,dayspm)
print(mppd)
# exclude Xmas
no_xmas = daily_electric2018_no_heat['{}-01-01'.format(year) : '{}-12-16'.format(year)]
print(no_xmas)
weekly_no_xmas = no_xmas.resample('W').sum() / 7.0
print('Daily variation: min {} max {} diff {}'.format(no_xmas.min(), no_xmas.max(), no_xmas.max() - no_xmas.min() ) )
print('Weekly variation: min {} max {} diff {}'.format(weekly_no_xmas.min(), weekly_no_xmas.max(), weekly_no_xmas.max() - weekly_no_xmas.min() ) )

# plot baseline 2018 daily and weekly
no_xmas.plot(color='blue', label='Daily baseline without Christmas holiday period {}'.format(year))
no_xmas.rolling(7, min_periods=1).mean().plot(color='green', label='7 day rolling average Daily baseline without Christmas holiday period {}'.format(year))
#weekly_no_xmas.plot(color='red', label='Weekly aseline without Christmas holiday period 2018')
plt.title('Baseline seasonaily {}'.format(year))
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.ylim(0,1.0)
plt.show()
