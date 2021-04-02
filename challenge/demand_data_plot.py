# python script to do some plots to investigate demand

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

def plot_days(df, days, title, dot=False):
    k = range(1,49)
    count=0
    for day in days:
        count+=1
        dft = df[day + ' 00:00:00' : day + ' 23:30:00']
        meantemp = dft['tempm'].mean()
        demand = dft['demand'].values
        if dot and count>len(days)/2:
            plt.plot(k, demand, label='{}, {:.2f}'.format(day, meantemp), linestyle = 'dotted' )
        else:
            plt.plot(k, demand, label='{}, {:.2f}'.format(day, meantemp) )

    plt.title(title)
    plt.xlabel('Hour of the day', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

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

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)


k = range(1,49)
# plot 
if args.plot:
    lockdown1 = ['2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19', '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27']
    plot_days(df, lockdown1, 'Start of the first lockdown', True)
    lockdown2 = ['2019-06-21', '2019-06-22', '2019-06-23', '2019-06-24', '2019-06-25', '2019-06-26', '2019-06-27', '2020-06-26', '2020-06-27', '2020-06-28', '2020-06-29', '2020-06-30', '2020-07-01', '2020-07-02']
    plot_days(df, lockdown2, 'Comparison of lockdown end with year before', True)

    demand_bins = pd.cut(df['demand'], bins=10).value_counts()
    demand_bins.plot()
    plt.title('Demand distribution')
    plt.show()

    bank_hols = pd.Series(df[df['dtype'] == 7].index.date).unique()
    print(bank_hols)
    days=[]
    for day in bank_hols:
        days.append(day.strftime('%Y-%m-%d') )
    print(days)
    plot_days(df, days, 'Bank holidays')

    tuesdays = ['2017-11-07', '2017-11-14', '2017-11-21', '2017-11-28', '2017-12-05', '2017-12-12', '2017-12-19', '2017-12-26']
    plot_days(df, tuesdays, 'Comparison of Tuesday demands')

    sundays = ['2017-11-05', '2017-11-12', '2017-11-19', '2017-11-26', '2017-12-03', '2017-12-10', '2017-12-17', '2017-12-24', '2017-12-25', '2017-12-26']
    plot_days(df, sundays, 'Comparison of Sunday demands')

    thursdays = ['2017-11-09', '2017-11-16', '2017-11-23', '2017-11-30', '2017-12-07', '2017-12-14', '2017-12-21', '2018-12-08', '2018-12-15', '2018-12-22']
    plot_days(df, thursdays, 'Comparison of Thursday demands')

    fridays = ['2017-11-03', '2017-11-10', '2017-11-17', '2017-11-24', '2017-12-01', '2017-12-08', '2017-12-15', '2017-12-22', '2018-12-16', '2018-12-23']
    plot_days(df, fridays, 'Comparison of Friday demands')

    days = ['2017-12-18', '2017-12-19', '2017-12-20', '2017-12-21', '2017-12-22', '2017-12-23', '2017-12-24']
    plot_days(df, days, 'Week before Christmas 2017')
    days = ['2018-12-17', '2018-12-18', '2018-12-19', '2018-12-20', '2018-12-21', '2018-12-22', '2018-12-23']
    plot_days(df, days, 'Week before Christmas 2018')
    days = ['2019-12-11', '2019-12-12', '2019-12-13', '2019-12-14', '2019-12-15', '2019-12-16', '2019-12-17']
    plot_days(df, days, 'Last week of data we have')
    days = ['2018-05-06', '2018-05-07', '2018-05-13', '2018-05-20', '2018-05-27', '2018-05-28']
    plot_days(df, days, 'Bank holidays and Sundays')
    days = ['2018-05-07', '2018-05-14', '2018-05-21', '2018-05-28', '2018-06-04']
    plot_days(df, days, 'Bank holidays and Mondays')

