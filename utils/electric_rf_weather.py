# Create Random Forrest (RF) Models to predict electricity from weather.
# Using different variables previously identified by LASSO.
# For the purpose of identifying which variables should be used to create
#  weather dependent demand. 

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
import pytz
from sklearn.ensemble import RandomForestRegressor
import pvlib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler
import calendar
import argparse

# custom code
import stats
import readers
import augment

# using sklearn Random Forest to predict the demand from weather

def forecast_demand(input_df, output, input_f):
    X_train = input_df
    y_train = output
    X_test = input_f
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
#   print(y_pred)
    # put the index of the year forecast on but may need to change this.
    return pd.Series(y_pred, index=output.index)

def get_demand(year, espini):
    if espini:
        demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
        demand = readers.read_espeni(demand_filename, year)
        electric = demand / 1000000.0
    else:
        # TODO - could we include the actual Scottish demand here?
        scotland_factor = 1.1
        # read historical electricity demand for reference year
        if year=='2009':
            demand_filename = '/home/malcolm/uclan/data/electricity/demanddata_2009.csv'
        else:
            demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'
        demand_ref = readers.read_electric_hourly(demand_filename)
        # Convert to TWh
        electric = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor / 1000000.0
    return electric

# process command line
parser = argparse.ArgumentParser(description='Create Random Forrest Prediction of electricity demand from weather')
parser.add_argument('--espini', action="store_true", dest="espini", help='Use the espini data', default=True)
parser.add_argument('--features', action="store", dest="features", help='List of comma seperated features to include in the regression model', default='hdh,cdh')
parser.add_argument('--year', action="store", dest="year", help='Training year', default='2018')
parser.add_argument('--test', action="store", dest="test", help='Test year', default='2017')
parser.add_argument('--frequency', action="store", dest="frequency", help='Frequency H=hourly, D=Daily, W=weekly', default='D')
parser.add_argument('--plot', action="store_true", dest="plot", help='Includes plots', default=False)
args = parser.parse_args()

freqs = {'H' : 'Hourly', 'D' : 'Daily', 'W' : 'Weekly' }

# Get the model features
variables = args.features.split(',')
print(variables)

# 
print('Train {} Test {} Frequency {} Features {}'.format(args.year, args.test, freqs[args.frequency], args.features) )

# Get training year weather

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(args.year)
weather = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

augment.augment(weather)

# Get test year weather

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(args.test)
weather_test = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather_test.index = pd.DatetimeIndex(weather_test.index).tz_localize('UTC')

augment.augment(weather_test)

# Get electricity demand

electric = get_demand(args.year, args.espini)
electric_test = get_demand(args.test, args.espini)

# Resample to required frequency Hourly or Daily

if args.frequency != 'H':
    electric = electric.resample(args.frequency).sum()
    electric_test = electric_test.resample(args.frequency).sum()
    weather = weather.resample(args.frequency).mean()
    weather_test = weather_test.resample(args.frequency).mean()

#variables = ['dailytemp', 'hour', 'temp_dp', 'hdh', 'temp', 'cdh', 'surface_pressure']
# Make a copy of the reference electricity demand and give it the time index
# from the test one
electric_reindex = electric.copy()
electric_reindex.index = electric_test.index
# forecast the test series from the weather
predicted = forecast_demand(weather[variables], electric_reindex, weather_test[variables])


# Plots
if args.plot:
    electric_test.plot(color='blue', label='Historic {} Electricity demand {}'.format(freqs[args.frequency], args.test))
    # TODO for a leap year would need to adjust the data 
    # (depending on frequency )
    predicted.plot(color='green', label='Forecast {} Electricity demand {} from RF model trained on {}'.format(freqs[args.frequency], args.test, args.year))
    plt.title('{} electricty demand series'.format(freqs[args.frequency]))
    plt.xlabel('day', fontsize=15)
    plt.ylabel('Energy (Twh) ', fontsize=15)
    plt.legend(loc='upper center')
    plt.show()

print('Forecast {} From {} {} electricity demand'.format(args.test, args.year, freqs[args.frequency]))
stats.print_stats_header('Year        ')
stats.print_stats(electric_test, electric_reindex, 'Unmodified', nvars=1, plot=False)
stats.print_stats(electric_test, predicted, '{}'.format(args.test), nvars=1, plot=False)
