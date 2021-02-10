# python script to clean weather data and split into the historic and
# the forecast weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
# Import tensor dataset & data loader
# Import nn.functional

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Clean weather data.')
parser.add_argument('set', help='weather file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"

# weather data
weather_filename = "{}weather_train_{}.csv".format(input_dir,dataset)
print('Cleaning {} {}'.format(dataset, weather_filename) )
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# up sample to 30 mins (with interpolation)
weather = weather.resample('30min').interpolate()

print(weather)

# Weather
n_missing = utils.missing_times(weather, '60min')
if n_missing>0:
    print("Missing rows in weather {}".format(n_missing) )
for col in weather.columns:
    if weather[col].isna().sum() >0:
        print("ERROR nans in {}".format(col))
        print(weather[col].isnull().values)
        quit()

# TODO add ( bilinear interpolation ) for ( 50.33, -4.034 )
# TODO add mean weather of the 6 points
#weather['temp_mean'] = weather['temp_location3']
#weather['solar_mean'] = weather['solar_location3']

# plot weather
if args.plot:
    weather['temp_location1'].plot(label='temperature 1', color='red')
    weather['temp_location2'].plot(label='temperature 2', color='blue')
    plt.title('temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Temp (degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    fewdays = weather['2018-06-01 00:00:00' : '2018-06-04 23:30:00']
    fewdays['solar_location1'].plot(label='sun 1', color='red')
    fewdays['solar_location2'].plot(label='sun 2', color='green')
    plt.title('Solar Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()


print(weather.columns)
weather.columns = ['temp3', 'temp6', 'temp2', 'temp4', 'temp5', 'temp1', 'sun3', 'sun6', 'sun2', 'sun4', 'sun5', 'sun1']

output_dir = "/home/malcolm/uclan/challenge/output/"

# output historic weather
output_filename = '{}weather_fixed_{}.csv'.format(output_dir, dataset)
weather.to_csv(output_filename, float_format='%.2f')
