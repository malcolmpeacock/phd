# Investigate the affect of onshore wind pattern vs offshore.
# Plot of capacity vs storage for different efficiency

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from os.path import exists

# custom code
import stats
import readers
import storage

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

output_dir = '/home/malcolm/uclan/output/experiments/zeropveta/'
cases = ['offshore', 'onshores']
etas = [30, 50, 80]

markers = ['o', 'v', '+', '<', 'x', 'D', '*', 'X','o', 'v', '+', '<', 'x', 'D', '*', 'X']
styles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed', 'dashdot' ]
colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'grey', 'pink', 'oive' ]

count_case=0
for case in cases:
    count_eta=0
    for eta in etas:
        share_file = '{}{}2/e{}/sharesENS.csv'.format(output_dir,case,str(eta))
        df = pd.read_csv(share_file, header=0, index_col=0)
        print(df)
        plt.plot(df['storage'], df['f_wind'],label='{} {}%'.format(case, eta), linestyle=styles[count_case], marker=markers[count_eta], color=colours[count_eta])
        count_eta+=1
    count_case+=1

plt.title('Wind capacity vs Storage (zero PV)')
plt.xlabel('Storage (days)')
plt.ylabel('Wind capacity (days)')
plt.legend(loc='upper right')
plt.show()

etas = [30, 40, 50, 60, 70, 80]
results = { 'eta' : etas }
for case in cases:
    results[case] = []
    results[case+str(2)] = []
    for eta in etas:
        share_file = '{}{}/e{}/sharesENS.csv'.format(output_dir,case,str(eta))
        df = pd.read_csv(share_file, header=0, index_col=0)

        storage_10 = df[df['storage']==10.0]
        wind_value = storage_10['f_wind'].values[0]
        results[case].append(wind_value)

        share_file = '{}{}2/e{}/sharesENS.csv'.format(output_dir,case,str(eta))
        df = pd.read_csv(share_file, header=0, index_col=0)
        capacity75 = df[df['f_wind']==7.5]
        storage_value = capacity75['storage'].values[0]
        results[case+str(2)].append(storage_value)

df = pd.DataFrame(data=results)
df.index = df['eta']
print(df)

if args.plot:
    # Wind capacity for 10 days storage
    df['offshore'].plot(label='Offshore wind, 10 days storage')
    df['onshores'].plot(label='Onshore wind scaled, 10 days storage')

    plt.title('Wind generation only')
    plt.xlabel('round trip efficiency')
    plt.ylabel('minimum wind capacity (days)')
    plt.legend(loc='upper right')
    plt.show()

    # Storage capacity for 7.5 days wind
    df['offshore2'].plot(label='Offshore wind, 7.5 days wind capacity')
    df['onshores2'].plot(label='Onshore wind scaled, 7.5 days wind capacity')

    plt.title('Wind generation only')
    plt.xlabel('round trip efficiency')
    plt.ylabel('minimum storage capacity (days)')
    plt.legend(loc='upper right')
    plt.show()

