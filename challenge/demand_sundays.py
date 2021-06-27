# python script to investigate recent sundays demand

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
import statsmodels.api as sm

# custom code
import utils
import glob

def plot_days(df, days, title):
    k = range(1,49)
    for day in days:
        dft = df[day + ' 00:00:00' : day + ' 23:30:00']
        meantemp = dft['tempm'].mean()
        demand = dft['demand'].values
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

ndays = 100
tdf = df.tail(ndays * 48)
sundays = tdf[tdf['dtype'] == 6]
sundays = sundays[(sundays['k'] > 31) & (sundays['k'] < 43)]

#daily_temp = sundays['tempm'].resample('D', axis=0).mean()
#print(daily_temp)
#days = daily_temp.index
days = pd.Series(sundays.index.date).unique()

temps = []
peaks = []
labels = []
k = range(32,43)
# plot 
if args.plot:

    for day in days:
        day_str = day.strftime('%Y-%m-%d')
        dft = sundays.loc[day_str]
        meantemp = dft['tempm'].mean()
        demand = dft['demand'].values
        plt.plot(k, demand, label='{}, {:.2f}'.format(day, meantemp) )
        temps.append(meantemp)
        peaks.append(demand.max())
        labels.append(day_str)

    plt.title('Comparison of Sunday demands')
    plt.xlabel('Hour of the day', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    # Fit line through the points - the add constant bit gives us 
    # the intercept as well as the gradient of the fit line.
    rmodel = sm.OLS(peaks, sm.add_constant(temps))
    residual_results = rmodel.fit()
#   print(residual_results.summary())
    res_const = residual_results.params[0]
    res_grad = residual_results.params[1]
    print('Gradient {} intercept {}'.format(res_grad, res_const) )
    # Fit of residuals line
    x = np.array([min(temps),max(temps)])
    y = res_const + res_grad * x

    fig, ax = plt.subplots()
    ax.scatter(temps, peaks)
    plt.title('Sunday Temp vs demand')
    plt.xlabel('Temperature ', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    for i in range(len(labels)):
        ax.annotate(labels[i], (temps[i], peaks[i]))
    plt.plot(x,y,color='red')
    plt.show()
