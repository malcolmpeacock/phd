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
demand_dfs={}
glob_filename = '{}demand_forecast_*_{}.csv'.format(output_dir, dataset)
for filename in glob.glob(glob_filename):
    method = filename[53:-4]
    print(filename, method)
    df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    df = df[(df['k']>31) & (df['k']<43)]
    demand_dfs[method] = df
    print(demand_dfs[method])

sdays_name = 'sdays_{}'.format(dataset)
nreg_name = 'nreg_{}'.format(dataset)
regl_name = 'regl_{}'.format(dataset)
regm_name = 'regm_{}'.format(dataset)
demand_dfs[sdays_name].index=range(len(demand_dfs[sdays_name]))
demand_dfs[nreg_name].index=range(len(demand_dfs[nreg_name]))
demand_dfs[regl_name].index=range(len(demand_dfs[regl_name]))
demand_dfs[regm_name].index=range(len(demand_dfs[regm_name]))

# plot 
if args.plot:

    demand_dfs[sdays_name]['demand'].plot(label='actual demand', color='red')
    demand_dfs[sdays_name]['prediction'].plot(label='10 similar days', color='blue')
    demand_dfs[nreg_name]['prediction'].plot(label='Regression #3', color='green')
    demand_dfs[regl_name]['prediction'].plot(label='Regression #1', color='orange')
    demand_dfs[regm_name]['prediction'].plot(label='Regression #2', color='purple')
    plt.title('Comparison of demand forecasts')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

print('Sdays')
utils.print_metrics(demand_dfs[sdays_name]['demand'], demand_dfs[sdays_name]['prediction'], args.plot)
print('nreg')
utils.print_metrics(demand_dfs[sdays_name]['demand'], demand_dfs[nreg_name]['prediction'], args.plot)
print('regl')
utils.print_metrics(demand_dfs[sdays_name]['demand'], demand_dfs[regl_name]['prediction'], args.plot)
print('regm')
utils.print_metrics(demand_dfs[sdays_name]['demand'], demand_dfs[regm_name]['prediction'], args.plot)
