# python script to create a demand forecast based on the most similar previous
# day.

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
#import torch
#import torch.nn as nn
# Import tensor dataset & data loader
#from torch.utils.data import TensorDataset, DataLoader
# Import nn.functional
#import torch.nn.functional as F

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method:' , default='similarday' )
parser.add_argument('--mode', action="store", dest="mode", help='Mode: forecast or test', default='forecast' )
parser.add_argument('--week', action="store", dest="week", help='Week to forecast: set=read the set forecast file, first= first week, last=last week, otherwise integer week' , default='set' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
method = args.method
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(df)

# weather data (forecast)
forecast_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(forecast)

if args.week != 'set':
    df, forecast = utils.extract_forecast_week(args.week,df,'demand',forecast)

# naive forecast based on the previous week
forecast['probability'] = 0.9
forecast['prediction'] = df['demand'].tail(7*48).values
print(forecast)

# find most similar day to a given day and then plot the profiles.

days = pd.Series(df.index.date).unique()
print(" -- days --")
print(days)
holidays = pd.Series(df[df['holiday']==1].index.date).unique()
print(" -- holidays --")
print(holidays)
weekdays = pd.Series(df[df['holiday']==0].index.date).unique()
print(" -- weekdays --")
print(weekdays)

# testing the algorithm

if args.mode == 'test':
    # try it for one day
#   given_day = days[len(days)-1]
    given_day = datetime(2017,11,3).date()
    closest_day, closeness = utils.find_closest_day(given_day, days, df, df, 'tempm', True)

    demand_score = utils.forecast_diff(given_day, closest_day, 'demand', df)
    print("given_day {} closest day {} demand score {}".format(given_day, closest_day, demand_score) )
    given_data = df.loc[given_day.strftime('%Y-%m-%d')]
    found_data = df.loc[closest_day.strftime('%Y-%m-%d')]
    if args.plot:
        plt.plot(given_data['k'], given_data['temp2'], label='Day to look for', color='red')
        plt.plot(found_data['k'], found_data['temp2'], label='Day that was found', color='blue')
        plt.title('Finding a similar day to a given day')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Temperature (Degrees C)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

        plt.plot(given_data['k'], given_data['demand'], label='Day to look for', color='red')
        plt.plot(found_data['k'], found_data['demand'], label='Day that was found', color='blue')
        plt.title('Finding a similar day to a given day')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    # assess the naive forecast
    # for each day in the last week ...
    #   compare its demand difference with the week before with its best 
    print('Naive forecast assesment')
    for id in range(len(days)-7,len(days)-1):
        dd = utils.forecast_diff(days[id], days[id-7], 'demand', df)
        print('{} {} {}'.format(days[id].strftime('%Y-%m-%d'), days[id-7].strftime('%Y-%m-%d'), dd) )

    # try it for all days!
    print('Assessing all days')
    all_scores={}
    for given_day in days:
        print("Testing {}".format(given_day))
        day_list = weekdays
        if df.loc[given_day.strftime('%Y-%m-%d'),'holiday'][0] == 1:
            print('Holiday')
            day_list = holidays
        closest_day, closeness = utils.find_closest_day(given_day, day_list, df, df, 'tempm', True)
        all_scores[given_day] = utils.forecast_diff(given_day, closest_day,'demand', df)
        rows = df.loc[closest_day.strftime('%Y-%m-%d')]
        df.loc[given_day.strftime('%Y-%m-%d'), 'prediction'] = utils.get_forecast(df.loc[given_day.strftime('%Y-%m-%d')], rows, 'demand', True)

    print("=========== Results")
    sorted_scores = sorted(all_scores, key=all_scores.get, reverse=True)
    for w in sorted_scores:
        print(w, all_scores[w])

    worst = sorted_scores[0]
    print('Worst {}'.format(worst) )
    worst_closest, closeness = utils.find_closest_day(worst, days, df, df, 'tempm', True)
    closest_data = df.loc[worst_closest.strftime('%Y-%m-%d')]
    worst_data = df.loc[worst.strftime('%Y-%m-%d')]
    utils.print_metrics(df['demand'], df['prediction'])
    if args.plot:
        plt.plot(closest_data['k'], closest_data['temp2'], label='Day that was found', color='blue')
        plt.title('Worst forecast day from the data')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Temperature (Degrees C)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

        plt.plot(worst_data['k'], worst_data['demand'], label='Day to look for', color='red')
        plt.plot(closest_data['k'], closest_data['demand'], label='Day that was found', color='blue')
        plt.title('Worst forecast day from the data')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

        df['demand'].plot(label='Actual Demand', color='blue')
        df['prediction'].plot(label='Predicted Demand', color='red')
        plt.title('Comparison of actual and forecast demand')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Demand', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

else:
    fdays = pd.Series(forecast.index.date).unique()
    temp_range = forecast['temp2'].max() - forecast['temp2'].min()
    print(fdays)
    for day in fdays:
        print("Testing {}".format(day))
        day_list = weekdays
        if forecast.loc[day.strftime('%Y-%m-%d'),'holiday'][0] == 1:
            print('Holiday')
            day_list = holidays
        closest_day, closeness = utils.find_closest_day(day, day_list, forecast, df, 'tempm', True)
        print('Closest day: {}'.format(closest_day) )
        rows = df.loc[closest_day.strftime('%Y-%m-%d')]
#       print(rows)
#       forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = rows['demand'].values
        forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = utils.get_forecast(forecast.loc[day.strftime('%Y-%m-%d')], rows, 'demand', True)
        probability = (temp_range - closeness) / temp_range
        forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability
    print(forecast)

# metrics
if 'demand' in forecast.columns:
    utils.print_metrics(forecast['demand'], forecast['prediction'])
    if args.plot:
        forecast['demand'].plot(label='Actual Demand', color='blue')
        forecast['prediction'].plot(label='Predicted Demand', color='red')
        plt.title('Comparison of actual and forecast demand')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Demand', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}demand_forecast_{}.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')

# only the demand for Bogdan
forecast = forecast['prediction']
forecast = forecast.squeeze()
forecast = forecast.rename('demand_forecast')
forecast.index.rename('datetime', inplace=True)

output_filename = '{}demand_forecast_{}_only.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')
