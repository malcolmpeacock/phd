# python script to compare demand forecats.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
import pvlib

# custom code
import utils
import glob

# main program

# process command line

parser = argparse.ArgumentParser(description='Clean weather data.')
parser.add_argument('set', help='weather file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
output_dir = "/home/malcolm/uclan/challenge/output/"

# demand data

demand_names = ['regd_svr', 'regd_ann', 'regd_linear', 'regs_svr', 'regs_ann', 'regs_linear']
#demand_names = ['regd', 'new', 'regs']
#demand_names = ['w1', 'w0', 'regm', 'newall', 'new400']
demand_labels = { 'w0' : 'model per k,day - L1 loss',
                  'w1' : 'model per k,day - weighted loss',
                  'regm' : 'regression for set2, binary flags',
                  'new' : 'ANN predicts max demand, then similar day',
                  'regd' : 'model per k,day - L1 loss',
                  'regs' : 'model per k, season- L1 loss ',
                  'regs_linear' : 'model per k, season- L1 loss ',
                  'regs_svr' : 'model per k, season. svr ',
                  'regs_ann' : 'model per k, season. ann ',
                  'regd_linear' : 'model per k, day of week- L1 loss ',
                  'regd_svr' : 'model per k, day of week. svr ',
                  'regd_ann' : 'model per k, day of week. ann ',
                  'mean' : 'average of ANN and model perk,day',
                }
demand_dfs={}
for name in demand_names:
    filename = '{}demand_forecast_{}_{}.csv'.format(output_dir, name, dataset)
    print(filename, name)
    df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    df = df[(df['k']>31) & (df['k']<43)]
    df.index=range(len(df))
    demand_dfs[name] = df
    print(demand_dfs[name])

#mean_dfs = demand_dfs['regd'][['demand', 'prediction', 'dailytemp']]
#mean_dfs['prediction'] = ( demand_dfs['regd']['prediction'] + demand_dfs['new']['prediction'] ) / 2
#demand_dfs['mean'] = mean_dfs

# plot 
if args.plot:

    count=0
    for name, pdf in demand_dfs.items():
        print(name)
        print(pdf)
        if count==0:
#           ax = pdf['tempm'].plot(label='temp', color='red')
#           plt.ylabel('Mean Temperature', fontsize=15, color='red')
#           ax2 = ax.twinx()
            pdf['demand'].plot(label='actual demand')
        pdf['prediction'].plot(label=demand_labels[name])
        count+=1
            
    plt.title('Comparison of demand forecasts')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.legend(loc='lower right', fontsize=15)
    plt.show()

    for name, pdf in demand_dfs.items():
        error = pdf['demand'] - pdf['prediction']
        plt.scatter(pdf['dailytemp'].values, error.values, s=12, color='blue')
            
        plt.title('{} Demand forecast error vs temperature'.format(demand_labels[name]) )
        plt.xlabel('Mean daily temperature (degress C)', fontsize=15)
        plt.ylabel('Demand forecast error (MWh)', fontsize=15)
        plt.show()

for name, pdf in demand_dfs.items():
    print(name)
    utils.print_metrics(pdf['demand'], pdf['prediction'], args.plot)
