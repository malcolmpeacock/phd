# python script to investigate idenfitying the non-heat gas algorithmically.
# IDEAS
# g(t) = cwv(t) * ( annual_gas_heat / 365.0 ) + n(t)
#  where cwt is the normalised composite weather variable
#
# then look at n(t) dependence on temperature and other variables.
# n(t) = a + b*T + c*T-sqaured
# calculate a, b, c by training a regression model.
#
# ISSUES: cwv only available for each LDZ
# so just using hdh

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import argparse

# custom code
import stats
import readers

def get_temperature(year):
    ref_year = '2018'
    file_base = 'Brhpp'
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(year, ref_year, file_base)
    demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
    daily_temp = demand['temperature'].resample('D').mean()
    return daily_temp

def regression(X, y):
    estimator = LinearRegression()
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
    print('Regression', fit.score(pf,y))
#   print(estimator.coef_)
#   print(estimator.intercept_)
#   p = fit.predict(Xp)
    # return intercept as coeff[0]
    coeffs[0] = estimator.intercept_
    return coeffs

# main program

# process command line
parser = argparse.ArgumentParser(description='User linear regression on HDD to derive non-heat gas')
parser.add_argument('--year', action="store", dest="year", help='Year', default='2018' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--temp', action="store_true", dest="temp", help='Use temperature, instead of HDD', default=False)
parser.add_argument('--basetemp', action="store", dest="basetemp", help='Base temp for HDD.', type=float, default=14.8)

args = parser.parse_args()

print('Year {} HDD base temp {} '.format(args.year, args.basetemp) )

# read gas

gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + args.year + '.csv'
gas = readers.read_gas(gas_filename)
# print(gas)
gas.index = pd.DatetimeIndex(gas.index).tz_localize('UTC')
# TWh
gas = gas * 1e-9

# read weather

daily_temperature = get_temperature(args.year)
print(daily_temperature)

# heating degree days
if args.temp:
    hdd = daily_temperature
    var = 'temperature'
    xlabel = 'Temperature ( degrees C)'
else:
    hdd = (args.basetemp - daily_temperature).clip(0.0)
    var = 'HDD'
    xlabel = 'Heating Degree Days ( degrees C)'

coeffs = regression(hdd.values.reshape(-1,1), gas.values)
c0 = coeffs[0]
c1 = coeffs[1]
print('Regression coeffs')
print(coeffs)

# calculate non-heat gas
y = c0 + hdd * c1
gas_hdh  = hdd * c1
base = gas - gas_hdh
print('Gas: annual {:.2f} min {:.2f} max {:.2f}'.format(gas.sum(), gas.min(), gas.max()))
print('Base: annual {:.2f} min {:.2f} max {:.2f}'.format(base.sum(), base.min(), base.max()))
print('{} Non heat gas {} heat gas {} total {}'.format(args.year, base.sum(), gas_hdh.sum(), gas.sum() ) )
print('{} Heat energy {}'.format(args.year, 0.8 * gas_hdh.sum() ) )

# plot regression with hdd

if args.plot:
    ax = plt.scatter(hdd.values, gas.values, color='blue', label='Daily gas demand {}'.format(args.year))
    plt.plot(hdd, y, color='red')
    plt.title('Relationship between daily gas consumption and {}'.format(var))
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Gas Demand (Twh) per day', fontsize=15)
    legend_elements = [Line2D([0], [0], color='red', label='Ordinary Least Squares Regression (Rd)'),
                       Line2D([0], [0], marker='o', color='blue', label='Daily gas demand {} (Gd)'.format(args.year)) ]
    plt.legend(loc='upper left', handles=legend_elements)
    plt.show()

    # plot gas with heating removed
    gas.plot(color='blue', label='Daily total gas demand {} (Gd)'.format(args.year))
    base.plot(color='green', label='Daily Gas demand with heating removed (Dd)')
    plt.title('Daily gas demand {} base using {}'.format(args.year, var))
    plt.xlabel('day of the year', fontsize=15)
    plt.ylabel('Gas Demand (Twh) per day', fontsize=15)
    plt.legend(loc='upper center')
    plt.show()




