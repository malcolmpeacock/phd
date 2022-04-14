# repeat the linear regression of Bloomfield et. al. to prove that
# removing the existing heat using the heat demand time series is more
# accurate.
# 

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

# using sklearn Random Forest to predict the base from weather

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

def regression(X, y):
    estimator = LinearRegression()
#   model = make_pipeline(PolynomialFeatures(1),estimator)
#   fit = model.fit(X, y)
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
    print('Fit {} Intercept {}'.format(fit.score(pf,y), estimator.intercept_))
#   p = fit.predict(Xp)
    return coeffs

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

def get_heat_bdew(year):
    # input assumptions for reference year ( true for both 2018 and 2017 )
    heat_that_is_electric = 0.06     # my spreadsheet from DUKES
    heat_that_is_heat_pumps = 0.01   # greenmatch.co.uk, renewableenergyhub.co.uk
    # remove the existing heat using the heat demand method
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef2018Weather{0:}I-Bbdew_resistive.csv'.format(year)
    ref_resistive = readers.read_copheat(demand_filename, ['electricity', 'temperature'])
    ref_resistive_heat = ref_resistive['electricity']
    existing_heat = ref_resistive_heat * heat_that_is_electric * 1e-6
    return existing_heat

def get_wdd_regression(demand, weather, model, hdh=False):
    print('Model {} '.format(model))
    parms = ['hdh']
    if model == 'cdhd':
        parms.append('cdh')
    if model == 'bloomfield':
        parms = ['doy', 'hdh', 'cdh', 'd0','d1','d2','d3','d4','d5','d6' ]
    if model == 'light':
        parms = ['hdh', 'cdh', 'ghi_w', 'ghi_w2', 'ghi_w3' ]
    if model == 'lightcp':
        parms = ['hdh', 'ghi_w', 'ghi_w2', 'cp' ]
    if model == 'cp':
        parms = ['hdh', 'cdh', 'cp']
    if args.parm:
        parms.append(args.parm)
    if hdh:
        use = ['hdh']
    else:
        use = parms

    print(parms)
    coeffs = regression(weather[parms].values, demand.values)
    # Remove the weather dependent demand.
    wdd = weather ['hdh'] * 0.0
    print('Regression Coeffients ')
    for i in range(len(parms)):
        c = coeffs[i+1]
        print('{} = {}'.format(parms[i], c))
        if parms[i] in use:
            wdd = wdd + weather[parms[i]] * c
        if parms[i] == 'hdh' or parms[i] == 'cdh':
           print('{} = {}'.format(parms[i][0:2]+'d', c/24))

    return wdd

# process command line
parser = argparse.ArgumentParser(description='Prove heat demand method is better than regression of Bloomfield et. al.')
parser.add_argument('--year', action="store", dest="year", help='Training year', default='2018')
parser.add_argument('--test', action="store", dest="test", help='Test year', default='2017')
parser.add_argument('--espini', action="store_true", dest="espini", help='Use the espini data', default=True)
parser.add_argument('--plot', action="store_true", dest="plot", help='Includes plots', default=False)
parser.add_argument('--frequency', action="store", dest="frequency", help='Frequency H=hourly, D=Daily, W=weekly', default='D')
parser.add_argument('--model', action="store", dest="model", help='Model eg Bloomfield, cdhd, etc.', default='cdhd')
parser.add_argument('--parm', action="store", dest="parm", help='Additional parameter to add, eg ghi', default=None)
parser.add_argument('--hdh', action="store_true", dest="hdh", help='Use only hdh')
parser.add_argument('--new', action="store_true", dest="new", help='Try to improve the heat demand method with RF')
args = parser.parse_args()
freqs = {'H' : 'Hourly', 'D' : 'Daily', 'W' : 'Weekly' }

print('Train {} Test {} Frequency {} Model {}'.format(args.year, args.test, freqs[args.frequency], args.model ) )

# read in the weather for the reference year:
weather_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(args.year)
weather_ref = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather_ref.index = pd.DatetimeIndex(weather_ref.index).tz_localize('UTC')

# read in the weather for the test year:
weather_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(args.test)
weather_test = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather_test.index = pd.DatetimeIndex(weather_test.index).tz_localize('UTC')

# augment with hdh, cdh and days of the week etc.

print('Augmenting test year weather ...')
augment.augment(weather_test)
print('Augmenting ref year weather ...')
augment.augment(weather_ref)
# weather_ref.to_csv('/home/malcolm/uclan/output/temp/weather_ref', float_format='%g')

# electricity demand
electric_ref = get_demand(args.year, args.espini)
electric_test = get_demand(args.test, args.espini)

# heat demand
heat_ref = get_heat_bdew(args.year)
heat_test = get_heat_bdew(args.test)

# Resamble to required frequency Hourly or Daily
if args.frequency != 'H':
    electric_ref = electric_ref.resample(args.frequency).sum()
    electric_test = electric_test.resample(args.frequency).sum()
    weather_ref = weather_ref.resample(args.frequency).mean()
    weather_test = weather_test.resample(args.frequency).mean()
    heat_ref = heat_ref.resample(args.frequency).sum()
    heat_test = heat_test.resample(args.frequency).sum()

# get the base electricity for reference year using regression.
base_ref_bdew = electric_ref - heat_ref
heat_regression_ref = get_wdd_regression(electric_ref, weather_ref, args.model, args.hdh)
base_ref_regression = electric_ref - heat_regression_ref
print('Reference year regression wdd {:.2f} base {:.2f} total {:.2f} bdew {:.2f}'.format(heat_regression_ref.sum(), base_ref_regression.sum(), electric_ref.sum(), heat_ref.sum() ))

# plot the regression derived heat along with the heat demand one.

if args.plot:
    electric_ref.plot(color='blue', label='2018 historic electicity demand')
    heat_ref.plot(color='red', label='Heat for electricity extracted by method proposed in this paper')
    heat_regression_ref.plot(color='green', label='Heat for electricity extracted by linear regression')
    plt.title('Validating removal of eixsting heat from 2018 - Daily')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

# Get heat in the test year electricity series - linear regression
base_test_regression = base_ref_regression.copy()
base_test_regression.index = electric_test.index
heat_regression_test = get_wdd_regression(electric_test, weather_test, args.model, args.hdh)
synthetic_test_regression = base_test_regression + heat_regression_test

# Add the 2017 heat using heat demand BDEW.
base_test_bdew = base_ref_bdew.copy()
base_test_bdew.index = electric_test.index
synthetic_test_bdew = base_test_bdew + heat_test
print('Test year regression wdd {:.2f} base {:.2f} total {:.2f} bdew {:.2f}'.format(heat_regression_test.sum(), base_test_regression.sum(), electric_test.sum(), heat_test.sum() ))

# plot the test against the real one
if args.plot:
    electric_test.plot(color='blue', label='{} historic electicity demand'.format(args.test))
    synthetic_test_regression.plot(color='red', label='Heat electricity for {} added to {} baseline using our method'.format(args.test, args.year))
    synthetic_test_bdew.plot(color='green', label='Heat electricity for {} added to {} baseline using linear regression'.format(args.test, args.year))
    plt.title('Adding heating electricity by different methods - {}'.format(freqs[args.frequency]))
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

print('{} Comparison of predictions to base'.format(freqs[args.frequency]))
stats.print_stats_header()
stats.print_stats(electric_test, synthetic_test_regression, 'Regres')
stats.print_stats(electric_test, synthetic_test_bdew, 'bdew')

# attempt to improve the heat demand method
if args.new:
    base_reindex = base_ref_bdew.copy()
    base_reindex.index = electric_test.index
#   variables = ['ghi_w','temp_dp','surface_pressure']
#   variables = ['ghi_w','ghi_w2', 'ghi_w3', 'cp']
    variables = ['temp_dp']
    # train a model using the reference year weather to forecast the baseline
    # then use it to get the test year baseline
    base_rf = forecast_demand(weather_ref[variables], base_reindex, weather_test[variables])
    synthetic_test_rf = base_rf + heat_test

    stats.print_stats(electric_test, synthetic_test_rf, 'rf')
