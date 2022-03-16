# Augment weather with zenith, month, hour, doy etc.
# Find what variables are significant in terms of 
#   Correlation (R) and Lasso
# At: Daily /Hourly / Weekly
# For: - The historic electricity demand time series
#      - The historic after heat has been removed by heat demand
#      - The historic after heat has been removed by linear regression

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
import scipy.stats as statsp

# custom code
import stats
import readers
import augment

def regression(X, y):
    estimator = LinearRegression()
#   model = make_pipeline(PolynomialFeatures(1),estimator)
#   fit = model.fit(X, y)
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
    print(fit.score(pf,y))
    print(estimator.coef_)
    print(estimator.intercept_)
#   p = fit.predict(Xp)
    return coeffs[1], coeffs[2]

# split cooling energy by CDD

def cooling(temperature, cooling_energy):
#   print(cooling_energy)
#   print(temperature)
#   cdd = (temperature - 15.0).clip(0.0)
#   Bloomfield et. al. suplementary material
    cdd = (temperature - 22.0).clip(0.0)
#   print(cdd)
    cooling = cdd * ( cooling_energy / cdd.sum() )
#   print('Cooling Degree Hours {}'.format(cdd.sum() ) )
#   print(cooling)
    return cooling

# feature identification using lasso
def lasso(input_df, output, title, plot=False):
    print('LASSO #################')
    transformer = MaxAbsScaler().fit(input_df.values)
    X = transformer.transform(input_df.values)
    transformer = MaxAbsScaler().fit(output.values.reshape(-1, 1))
    y = transformer.transform(output.values.reshape(-1, 1))
#   print('normalized Y')
#   print(y)
    # defaults : tol=1e-4, max_iter=1000, 
#   reg = LassoCV(max_iter=8000, tol=1e-1)
    reg = LassoCV(max_iter=8000, tol=0.1)
    reg.fit(X, y)
#   print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
#   print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = input_df.columns)
    imp_coef = coef.sort_values()
#   print(imp_coef)
    if plot:
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind = "barh")
        plt.title("{} Feature importance using Lasso Model".format(title))
        plt.show()
    return coef

# feature correlation
def correlation(input_df, output, title, plot=False):
#   print(input_df.index)
#   print(output.index)
    coef = {}
    for column in input_df.columns:
        coef[column] = output.corr(input_df[column])
        tt = statsp.ttest_rel(input_df[column], output)
        print(tt)
    coefs = pd.Series(coef, index = input_df.columns)
    print(coefs)
    imp_coef = coefs.sort_values()
    if plot:
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind = "barh")
        plt.title("{} Feature importance using R (Pearsons Correlation Coefficient) ".format(title))
        plt.show()
    return(coef)

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
parser = argparse.ArgumentParser(description='Show the impact of weather variables on electricity demand')
parser.add_argument('--espini', action="store_true", dest="espini", help='Use the espini data', default=True)
parser.add_argument('--input', action="store", dest="input", help='What to perform the test on', default='historic')
parser.add_argument('--plot', action="store_true", dest="plot", help='Includes plots', default=False)
parser.add_argument('--year', action="store", dest="year", help='Training year', default='2018')
parser.add_argument('--frequency', action="store", dest="frequency", help='Frequency H=hourly, D=Daily, W=weekly', default='D')
parser.add_argument('--vplot', action="store", dest="vplot", help='Variable to scatter plot against bastline demand.', default=None)

args = parser.parse_args()

freqs = {'H' : 'Hourly', 'D' : 'Daily', 'W' : 'Weekly' }

print('Year {} Frequency {} Input {}'.format(args.year, freqs[args.frequency], args.input) )

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(args.year)
weather = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

augment.augment(weather, False)
#weather.to_csv('/home/malcolm/uclan/output/temp/weather_ref.csv', float_format='%g')

# Get electricity demand

electric = get_demand(args.year, args.espini)


# input assumptions for reference year
heat_that_is_electric = 0.06     # my spreadsheet from DUKES
# remove the existing heat
if args.input == 'noheat':
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew_resistive.csv'.format(args.year)
    ref_resistive = readers.read_copheat(demand_filename, ['electricity', 'temperature'])
    ref_resistive_heat = ref_resistive['electricity'] * heat_that_is_electric
#   ref_temperature = ref_resistive['temperature']

# Resamble to required frequency Hourly or Daily

if args.frequency != 'H':
    electric = electric.resample(args.frequency).sum()
    weather = weather.resample(args.frequency).mean()
    unmodified = electric.copy()
    if args.input == 'noheat':
        ref_resistive_heat = ref_resistive_heat.resample(args.frequency).sum()

#  To remove existing space and water heating from the electricity demand time 
#  series for the reference year - subtract the resistive heat series
#  ( even if using historic series we calculate it so can scale to it )

        existing_heat = ref_resistive_heat * 1e-6
        electric = electric - existing_heat

    if args.plot:
        unmodified.plot(color='blue', label='Unmodified electicity')
        if args.input == 'noheat':
            existing_heat.plot(color='red', label='Heat')
            electric.plot(color='green', label='Electiricty with heat removed')
        plt.title('Electricity with heat removed')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Energy', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()

    if args.vplot:
        label = 'historic'
        if args.input == 'noheat':
            label = 'baseline'
        plt.scatter(weather[args.vplot]/1000.0, electric)
        plt.title('Relationship between {} and {} electricity demand'.format(args.vplot, label))
        plt.xlabel('{}'.format(args.vplot), fontsize=15)
        plt.ylabel('Energy', fontsize=15)
        plt.show()

# correlation of the various forecasting parameters with the demand
coeffs = correlation(weather, electric, '{} {} '.format(freqs[args.frequency],args.year), plot=True)
print('{} Feature correlation'.format(freqs[args.frequency]))
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# lasso
print('Lasso')
coeffs = lasso(weather, electric, '{} {}'.format(freqs[args.frequency],args.year), plot=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

