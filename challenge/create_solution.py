# python script to create a charging strategy from the demand and pv forecast

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
from numpy import trapz

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Create charging strategy.')
parser.add_argument('demand', help='Demand file')
parser.add_argument('pv', help='Weather file')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
pv_file = args.pv
demand_file = args.demand
print('Demand forecast {} PV forecast {}'.format(demand_file, pv_file) )

# read in the data
input_dir = "/home/malcolm/uclan/challenge/output/"

# demand data
demand_filename = input_dir + demand_file
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(demand)

# pv data
pv_filename = input_dir + pv_file
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# set weight to sort on
pv['weight'] = pv['prediction'] * pv['probability']
demand['weight'] = demand['prediction'] * demand['probability']

# create solution file of zeros

solution = pv.copy()
solution['charge_MW'] = 0.0
#solution = solution.drop(['prediction', 'probability', 'weight'], axis=1)
solution = solution['charge_MW']
solution.index.rename('datetime')
solution = solution.squeeze()

# For each day ...
days = solution.resample('D', axis=0).mean().index
for day in days:
    print(day)
    pv_day = pv[day : day + pd.Timedelta(hours=23,minutes=30)].copy()
#   print(pv_day)

    # sort by power weighted by probability of correct forecast
    pv_day.sort_values(by='weight', ascending=False, inplace=True)

    # charge the battery from the pv
    # we only put weight into the battery incase the forecast is too high
    # TODO - modify this if there is not enough PV to fill the battery?
    #        the weighting could cause unncessary charging from the grid?
    #        perhaps a 2nd pass if we run out?

    battery = 0.0
    # this is 12 rather than 6 to avoid multiplying by 0.5 all the time
    # ( so its 6 half MWhs )
    capacity = 11.9999
    for index, row in pv_day.iterrows():
        k = utils.index2k(index)
        print('Charging k {} battery {} index {} weight {} pv {}'.format(k, battery, index, row['weight'], row['prediction']) )
        if k<32:
            charge = min(row['weight'],2.5)
            charge_output = min(charge, capacity-battery)
            print('Adding Charge {}'.format(charge_output) )
            # don't overfill the battery
            solution[index] = charge_output
            battery += charge
        # stop when battery is full
        if battery>capacity:
            battery = capacity
            break;
    print('Charged: k {} battery {} index {} weight {} pv {}'.format(k, battery, index, row['weight'], row['prediction']) )

    # get the demand for this day
    demand_day = demand[day : day + pd.Timedelta(hours=23,minutes=30)]
#   print(demand_day)
    # sort the demand so we discharge at points of heighest demand
    demand_day.sort_values(by='weight', ascending=False, inplace=True)
    # calculate areas under the curve
#   heights
#   for index, row in demand_day.iterrows():
#       k = utils.index2k(index)
#       if k>31 and k<43:
    # discharge the battery
    for index, row in demand_day.iterrows():
        k = utils.index2k(index)
        print('Discharging k {} battery {} index {} demand {}'.format(k, battery, index, row['prediction']) )
        if k>31 and k<43:
            # don't discharge the battery below empty
            discharge = min(battery,2.5)
            print('Dis Charge {}'.format(discharge) )
            solution[index] = -discharge
            battery -= discharge
        # stop when battery is empty
        if battery<0.001:
            break;

#print(solution)
final_score, new_demand = utils.solution_score(solution, pv, demand)
print('Final Score {}'.format(final_score) )

if args.plot:
    pv['prediction'].plot(label='PV Generation Forecast', color='red')
    new_demand.plot(label='Modified demand', color='yellow')
    demand['prediction'].plot(label='Demand Forecast', color='blue')
    solution.plot(label='Battery Charge', color='green')
    plt.title('Charging Solution')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('MWh', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()



output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = output_dir + 'solution.csv'

solution.index.rename('datetime', inplace=True)
solution.to_csv(output_filename, float_format='%g')
