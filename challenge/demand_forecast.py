# python script to create a demand forecast from the previous demand and
# the weather

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

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('demand', help='Demand file')
parser.add_argument('weather', help='Weather file')

args = parser.parse_args()
weather_file = args.weather
demand_file = args.demand

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"

# demand data
demand_filename = input_dir + demand_file
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

#print(demand)

# weather data
weather_filename = input_dir + weather_file
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(weather)

# error checks
n_missing = utils.missing_times(demand, '30min')
print('Number of missing demand rows: {}'.format(n_missing) )
print(demand.nsmallest())
print(demand.nlargest())

# fix zeros

low = demand[demand < 0.1]
#print(low)
last_week = demand[low.index - pd.Timedelta(days=7)]
last_week.index = low.index
#print(last_week)
demand.update(last_week)

#bad_day = demand['2018-05-10']
#print(bad_day)
#previous_day = demand['2018-05-09']
#previous_day.index = bad_day.index
#print(previous_day)
#demand.update(previous_day)

# replace a suspect days with different ones.
utils.replace_day(demand, '2018-05-10', '2018-05-09')
utils.replace_day(demand, '2018-05-11', '2018-05-12')

# plot demand
demand.plot(label='demand', color='blue')
plt.title('demand')
plt.xlabel('Hour of the year', fontsize=15)
plt.ylabel('Demand (MW)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

demand_forecast = demand.tail(7*48)
print(demand_forecast)
next_day = demand_forecast.last_valid_index() + pd.Timedelta(minutes=30)
print(next_day)
last_day = demand_forecast.last_valid_index() + pd.Timedelta(days=7)
print(last_day)
#demand_forecast.index = pd.DatetimeIndex(
new_index = pd.date_range(start = next_day, end= last_day, freq='30min')
print(new_index)
demand_forecast.index = new_index
print(demand_forecast)

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = output_dir + 'demand_forecast.csv'

demand_forecast.to_csv(output_filename)
