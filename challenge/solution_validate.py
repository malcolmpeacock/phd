# python script to validate solution file for the pod challenge.

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

# custom
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Create charging strategy.')
parser.add_argument('solution', help='solution file name')
parser.add_argument('--demand', action="store", dest="demand", help='Demand file for plotting' , default=None )
parser.add_argument('--pv', action="store", dest="pv", help='PV file for plotting' , default=None )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
solution = args.solution

# read in the data
input_dir = "/home/malcolm/uclan/challenge/output/"
filename = input_dir + solution
print('***Validating File {}'.format(filename) )
s = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

#print(s)

# no NaN's
nans = s.isna().sum()
if nans >0:
    print('FAIL {} nans'.format(nans) )

# 1 weeks worth of values
if len(s) != 48 * 7:
    print('FAIL wrong number of values')

count=0
store=0.0
errors=0

daily_charge = {}
daily_discharge = {}

# For each day ...
days = s.resample('D', axis=0).mean().index
for day in days:
    daily_discharge[day] = 0.0
#   print(day)
    s_day = s[day : day + pd.Timedelta(hours=23,minutes=30)]
#   print(s_day)

    # for each period of the day ...
    for index, value in s_day.items():
        count+=1
        # -2.5 <= B <= 2.5
        if value<-2.5 or value>2.5:
            print('Value {} out of range at line {}'.format(value, count) )
            errors+=1
        # period
        k = utils.index2k(index)
        # B <= 0 if period >= 32
        if k>31 and k<43 and value >0:
            print('Charging at the wrong time: Value {} period {} at line {}'.format(value, k, count) )
            errors+=1
        # B >= 0 if period <= 31
        if k<32 and value <0:
            print('Discharging at the wrong time: Value {} period {} at line {}'.format(value, k, count) )
            errors+=1
        # battery storage
        store = store + (0.5 * value)
        if store <0 or store >6:
            print('Store out of range: store {} day {} period {} at line {}'.format(store, day, k, count) )
            errors+=1
        # daily summary
        if k<32:
            daily_charge[day] = store
        else:
            daily_discharge[day] += (0.5 * value)

    # Check for full discharge at day end
    if store>0.0:
        print('Warning: on day {} store had some left {}'.format(day, store) )
        store=0.0

if errors==0:
    print("PASSED")
else:
    print("Failed {} errors".format(errors) )

for day, value in daily_charge.items():
    print('Day {} Charge {}'.format(day, value) )
for day, value in daily_discharge.items():
    print('Day {} DisCharge {}'.format(day, value) )

if args.demand and args.pv:
    # demand data
    demand_filename = input_dir + args.demand
    demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

    print(demand)

    # pv data
    pv_filename = input_dir + args.pv
    pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

    final_score, new_demand = utils.solution_score(s, pv, demand)
    print('Final Score {}'.format(final_score) )

    if args.plot:
        pv['prediction'].plot(label='PV Generation Forecast', color='red')
        new_demand.plot(label='Modified demand', color='yellow')
        demand['prediction'].plot(label='Demand Forecast', color='blue')
        s.plot(label='Battery Charge', color='green')
        plt.title('Charging Solution')
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('MWh', fontsize=15)
        plt.legend(loc='lower right', fontsize=15)
        plt.show()

