# python script to see how we did. ie compare our forecasts with reality.

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
parser.add_argument('set', help='eg 0 for set 0 ', type=int)
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--perfect', action="store_true", dest="perfect", help='Use actual PV instead of the forecast', default=False)
parser.add_argument('--sf', action="store_true", dest="sf", help='Read from set file not u_cvml', default=False)

args = parser.parse_args()
set1 = args.set
set2 = set1 + 1

print('Checking set {} against set {}  to see how we did'.format(set1, set2) )

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
output_dir = "/home/malcolm/uclan/challenge/output/"

# new demand data
demand_filename = input_dir + 'demand_train_set{}.csv'.format(set2)
demand_new = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(demand_new)
demand_filename = output_dir + 'demand_forecast_set{}.csv'.format(set1)
demand_forecast = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# pv data
pv_filename = input_dir + 'pv_train_set{}.csv'.format(set2)
pv_new = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(pv_new)

pv_filename = output_dir + 'pv_forecast_set{}.csv'.format(set1)
pv_forecast = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(pv_forecast)

solution_filename = output_dir + 'u_cvml_set{}.csv'.format(set1)
if args.sf:
    solution_filename = output_dir + 'solution_set{}.csv'.format(set1)
   
solution = pd.read_csv(solution_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(solution)

#data = solution.join(pv_new, how='left')
#print(data)

pv_new = pv_new.loc[solution.index]
if args.perfect:
    pv_forecast['prediction'] = pv_new['pv_power_mw'].values

#print(pv_new)
demand_new = demand_new.loc[solution.index]
#print(demand_new)
print('PV sum {}'.format(pv_new['pv_power_mw'].sum()))

print('PV Forecast')
utils.print_metrics(pv_new['pv_power_mw'], pv_forecast['prediction'])
print('Demand Forecast')
utils.print_metrics(demand_new, demand_forecast['prediction'])

print('Demand total Forecast {} Actual {}'.format(demand_forecast['prediction'].sum(), demand_new.sum() ) )
print('PV total Forecast {} Actual {}'.format(pv_forecast['prediction'].sum(), pv_new['pv_power_mw'].sum() ) )

final_score, modified_demand = utils.solution_score(solution, pv_new['pv_power_mw'], demand_new)
print('Final Score {}'.format(final_score) )

# plots
if args.plot:
    pv_forecast['prediction'].plot(label='PV Generation Forecast', color='red')
    pv_new['pv_power_mw'].plot(label='PV Actual', color='blue')
    # theorectical max? PR=0.8, 5.0MwP of plant, 2 because of half hour periods
    pv_forecast['pv_max'] = (pv_forecast['cs_ghi'] * 0.8 * 5.0 ) * 0.002
    pv_forecast['pv_max'].plot(label='Max PV from clear sky ghi', color='purple', linestyle = 'dotted')
    demand_forecast['prediction'].plot(label='Demand Forecast', color='orange')
    demand_new.plot(label='Actual demand', color='yellow')
    solution.plot(label='Battery Charge', color='green')
    modified_demand.plot(label='Modified Demand', color='black')
    plt.title('How we did set {}'.format(set1) )
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('MWh', fontsize=15)
#   plt.legend(loc='upper right', fontsize=15, bbox_to_anchor=(0.9,1.5) )
    plt.legend(loc='upper left', fontsize=13 )
    plt.show()
