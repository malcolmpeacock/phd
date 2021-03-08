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

demand_names = ['w1', 'w0', 'regm']
demand_dfs={}
for name in demand_names:
    filename = '{}demand_forecast_{}_{}.csv'.format(output_dir, name, dataset)
    print(filename, name)
    df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    df = df[(df['k']>31) & (df['k']<43)]
    df.index=range(len(df))
    demand_dfs[name] = df
    print(demand_dfs[name])

# plot 
if args.plot:
    count=0
    for name, pdf in demand_dfs.items():
        print(name)
        print(pdf)
        if count==0:
            pdf['demand'].plot(label='actual demand')
        pdf['prediction'].plot(label=name)
        count+=1
            
    plt.title('Comparison of demand forecasts')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

for name, pdf in demand_dfs.items():
    print(name)
    utils.print_metrics(pdf['demand'], pdf['prediction'], args.plot)
