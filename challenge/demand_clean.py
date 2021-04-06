# python script to clean the demand data
# the weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math

# custom code
import utils

# main program
parser = argparse.ArgumentParser(description='Clean demand data.')
parser.add_argument('set', help='demand file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
demand_filename = "{}demand_train_{}.csv".format(input_dir,dataset)
print('Cleaning {} {}'.format(dataset, demand_filename) )

# demand data
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

#print(demand)

# error checks
n_missing = utils.missing_times(demand, '30min')
print('Number of missing demand rows: {}'.format(n_missing) )

# fix zeros

#last_week = demand[low.index - pd.Timedelta(days=7)]
#last_week.index = low.index
#print(last_week)
#demand.update(last_week)

#bad_day = demand['2018-05-10']
#print(bad_day)
#previous_day = demand['2018-05-09']
#previous_day.index = bad_day.index
#print(previous_day)
#demand.update(previous_day)

mean_demand = demand.mean()
low_threshold = mean_demand * 0.01
high_threshold = mean_demand * 2.0
print('Mean demand : {} Low threshold {} High threshold {}'.format(mean_demand, low_threshold, high_threshold) )

# fix low values
low = demand[demand < low_threshold]
low_days = pd.Series(low.index.date).unique()
for day in low_days:
    day_str = day.strftime('%Y-%m-%d')
    low_day = low.loc[day_str]
    nlow = len(low_day)
    print('Day {} has {} low values'.format(day_str, nlow) )
    if nlow>2:
        print('Dropping Day {}'.format(day_str) )
        demand.drop(demand[day_str].index, inplace=True)
    else:
        print('Fixing Day {} by interpolation'.format(day_str) )
        day_to_fix = demand[day_str]
        day_to_fix.loc[low_day.index] = float("NaN")
        day_to_fix = day_to_fix.interpolate()
        demand.loc[day_str] = day_to_fix.values

# fix high values
high = demand[demand > high_threshold]
high_days = pd.Series(high.index.date).unique()
for day in high_days:
    day_str = day.strftime('%Y-%m-%d')
    high_day = high.loc[day_str]
    nhigh = len(high_day)
    print('Day {} has {} high values'.format(day_str, nhigh) )
    if nhigh>2:
        print('Dropping Day {}'.format(day_str) )
        demand.drop(demand[day_str].index, inplace=True)
    else:
        print('Fixing Day {} by interpolation'.format(day_str) )
        day_to_fix = demand[day_str]
        day_to_fix.loc[high_day.index] = float("NaN")
        day_to_fix = day_to_fix.interpolate()
        demand.loc[day_str] = day_to_fix.values

# replace a suspect days with different ones.
if dataset[0:3]=='set':
#   utils.replace_day(demand, '2018-05-10', '2018-05-09')
#   utils.replace_day(demand, '2018-05-11', '2018-05-12')
    print('Dropping days 2018-05-08 2018-05-09')
    # drop because of small values
#   demand.drop(demand['2018-05-08'].index, inplace=True)
#   demand.drop(demand['2018-05-09'].index, inplace=True)
    # drop because of large values
#   demand.drop(demand['2018-05-10'].index, inplace=True)
#   demand.drop(demand['2018-05-11'].index, inplace=True)
#   demand.drop(demand['2018-11-04'].index, inplace=True)
    # replace low values
#   demand['2020-02-28 07:30:00'] = demand['2020-02-08 07:00:00']
#   demand['2020-03-17 12:00:00'] = demand['2020-03-17 11:30:00']
#   demand['2020-03-17 12:30:00'] = demand['2020-03-17 13:00:00']
print('Low demand values')
low = demand[demand < low_threshold]
print(low)
print('SMALLEST')
print(demand.nsmallest())
print('LARGEST')
print(demand.nlargest())

# plot demand
if args.plot:
    demand.plot(label='demand', color='blue')
    plt.title('demand')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    fewdays = demand['2020-03-20 00:00:00' : '2020-07-03 23:30:00']
#   fewdays = demand['2020-06-26 00:00:00' : '2020-07-03 23:30:00']
    fewdays.plot(label='demand', color='blue')
    plt.title('demand for 4 days')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

demand.rename('demand', inplace=True)
output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}demand_fixed_{}.csv'.format(output_dir, dataset)

demand.to_csv(output_filename)
