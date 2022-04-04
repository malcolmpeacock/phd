# python script to plot hourly profiles in heat_series.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
# custom code
import stats

def read_series(name):
    dir = "/home/malcolm/uclan/output/timeseries/"
    filename = dir + name + '.csv'
    series = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    return series

def plot_series(series, title, xlab, ylab):
    for name, data in series.items():
        print(name)
        print(data)
        data.plot(label=name)

    plt.title(title)
    plt.xlabel(xlab, fontsize=15)
    plt.ylabel(ylab, fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

# main program

names = ['baseline_daily_2018', 'heatpumps_41_daily_2018', 'heatpumps_all_daily_2018']
series = {}

for name in names:
   series[name] = read_series(name)

plot_series(series, 'Daily time series 2018', 'day', 'Demand')

names = ['baseline_daily_all', 'heatpumps_41_all_daily_all', 'heatpumps_all_daily_all']
series = {}

for name in names:
   series[name] = read_series(name)

plot_series(series, 'Daily time series 40 years', 'day', 'Demand')
