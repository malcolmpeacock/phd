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

# function to find the difference in weather between 2 days

def day_diff(day1, day2, df1, df2):
    day1_data = df1.loc[day1.strftime('%Y-%m-%d')]
#   print(day1_data)
    day2_data = df2.loc[day2.strftime('%Y-%m-%d')]
#   print(day2_data)
    diff = day1_data['temp2'].values - day2_data['temp2'].values
#   print(diff)
    score = np.abs(diff).sum()
#   print(day1,day2,score)
    return score

# function to assess the demand forecast accuracy between 2 days
def demand_diff(day1, day2):
    day1_data = df.loc[day1.strftime('%Y-%m-%d')]
    day2_data = df.loc[day2.strftime('%Y-%m-%d')]
    day1k = utils.krange(day1_data)
    day2k = utils.krange(day2_data)
    diff = day1k['demand'].values - day2k['demand'].values
    score = np.abs(diff).sum()
#   print(day1,day2,score)
    return score

def find_closest_day(given_day, days, df1, df2):
    closest_day = days[0]
    closest_day_score = 999999999999999.9
    for day in days:
        if day!=given_day:
            day_diff_score = day_diff(given_day, day, df1, df2)
            if day_diff_score < closest_day_score:
                closest_day = day
                closest_day_score = day_diff_score
    return closest_day


# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method:' , default='similarday' )
parser.add_argument('--mode', action="store", dest="mode", help='Mode forecast or test', default='forecast' )
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

#days = df.resample('D', axis=0).mean().index.date
days = pd.Series(df.index.date).unique()
print(days)

# testing the algorithm

if args.mode == 'test':
    given_day = days[len(days)-1]
    closest_day = days[0]
    closest_day_score = 999999999999999.9
    for day in days:
        if day!=given_day:
            print(given_day, day)
            day_diff_score = day_diff(given_day, day, df, df)
            if day_diff_score < closest_day_score:
                closest_day = day
                closest_day_score = day_diff_score

    demand_score = demand_diff(given_day, closest_day)
    print("given_day {} closest day {} weather score {} demand score {}".format(given_day, closest_day, closest_day_score, demand_score) )
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
        dd = demand_diff(days[id], days[id-7])
        print('{} {} {}'.format(days[id].strftime('%Y-%m-%d'), days[id-7].strftime('%Y-%m-%d'), dd) )

    # try it for all days!
    print('Assessing all days')
    all_scores={}
    for given_day in days:
        print("Testing {}".format(given_day))
        closest_day = find_closest_day(given_day, days, df, df)
        all_scores[given_day] = demand_diff(given_day, closest_day)

    print("=========== Results")
    sorted_scores = sorted(all_scores, key=all_scores.get, reverse=True)
    for w in sorted_scores:
        print(w, all_scores[w])

    worst = sorted_scores[0]
    print('Worst {}'.format(worst) )
    worst_closest = find_closest_day(worst, days, df, df)
    closest_data = df.loc[worst_closest.strftime('%Y-%m-%d')]
    worst_data = df.loc[worst.strftime('%Y-%m-%d')]
    if args.plot:
        plt.plot(closest_data['k'], closest_data['temp2'], label='Day that was found', color='blue')
        plt.title('Finding a similar day to a given day')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Temperature (Degrees C)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

        plt.plot(worst_data['k'], worst_data['demand'], label='Day to look for', color='red')
        plt.plot(closest_data['k'], closest_data['demand'], label='Day that was found', color='blue')
        plt.title('Finding a similar day to a given day')
        plt.xlabel('K period of the day', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

else:
    fdays = pd.Series(forecast.index.date).unique()
    print(fdays)
    for day in fdays:
        print("Testing {}".format(day))
        closest_day = find_closest_day(day, days, forecast, df)
        print(closest_day)
        rows = df.loc[closest_day.strftime('%Y-%m-%d')]
#       print(rows)
        forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = rows['demand'].values
#       times = forecast.loc[day.strftime('%Y-%m-%d')].index
#       print(times)
#       quit()
#       forecast.loc[day, 'prediction'] = rows['demand'].values
#       for index, row in rows.iterrows():
#           print(index, row)
#           forecast.loc[index, 'prediction'] = row['demand']
    print(forecast)

# metrics
if 'demand' in forecast.columns:
    utils.print_metrics(forecast['demand'], forecast['prediction'])

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}demand_forecast_{}.csv'.format(output_dir, dataset, method)

forecast.to_csv(output_filename, float_format='%.2f')
