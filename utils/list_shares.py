# List the shares files and costs from a specified directory

# library stuff
import sys
import glob
import os
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
import math
from os.path import exists
from skimage import measure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# custom code
import stats
import readers
import storage

def days2twh(days, one_day, hourly):
    one_day = one_day * 1e-6
    if hourly:
        one_day = one_day * 24
    twh = days * ( one_day )
    return twh

def get_series(name, hourly, doIndex):
    path = output_dir + name + scenario
    store = pd.read_csv(path, header=0, index_col=0).squeeze()
    if doIndex:
        store.index = pd.DatetimeIndex(pd.to_datetime(store.index).date)
    return store

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

# main program

# process command line
parser = argparse.ArgumentParser(description='List shares files by directory')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--pstore', action="store_true", dest="pstore", help='Plot the sample store history ', default=False)
parser.add_argument('--pdemand', action="store_true", dest="pdemand", help='Plot the demand ', default=False)
parser.add_argument('--fit', action="store_true", dest="fit", help='Try fiting a curve', default=False)
parser.add_argument('--nolist', action="store_true", dest="nolist", help='Do not list values', default=False)
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
parser.add_argument('--dir', action="store", dest="dir", help='Directory', default='adhoc')
parser.add_argument('--units', action="store", dest="units", help='Units', default='days')
parser.add_argument('--normalise', action="store", dest="normalise", help='Normalise factor to override the one from settings', type=float, default=0.0)
parser.add_argument('--pv', action="store", dest="pv", help='A particular value of PV', default=None, type=float)
parser.add_argument('--wind', action="store", dest="wind", help='A particular value of wind', default=None, type=float)
parser.add_argument('--inrate', action="store_true", dest="inrate", help='Base the charge rate on the energy input, not energy stored', default=False)
args = parser.parse_args()

# variables and axis labels
axis_labels = {
    'f_pv': 'Solar PV ( generation capacity in proportion to normalised demand)',
    'f_wind': 'Wind ( generation capacity in proportion to nomarlised demand)',
    'energy' : 'Energy generated ( normalised to demand )',
    'fraction' : 'Wind energy fraction',
    'wind_energy' : 'Wind energy ( normalised to demand )',
    'pv_energy' : 'PV energy ( normalised to demand )',
    'storage' : 'Amount of energy storage (days)',
    'cost' : 'cost ( £/Kwh )',
    'last' : 'store level % fill at end',
}

# units
units = {
    'f_pv': 'days',
    'f_wind': 'days',
    'energy' : 'days',
    'fraction' : '%',
    'wind_energy' : 'days',
    'pv_energy' : 'days',
    'storage' : 'days',
    'cost' : '£/Kwh',
    'last' : '%',
}

stores={}
durations={}
hydrogens={}
labels={}

# loop round the files
output_dir = '/home/malcolm/uclan/output/' + args.dir + '/'

if not args.nolist:
    print('File f_wind f_pv storage    charge discharge cost  energy                                       last ')
    print('                 days twh   rate   rate            wind pv   fraction total discharged charged')
for path in glob.glob(output_dir + 'shares*.csv'):
    df = pd.read_csv(path, header=0, index_col=0)
    for col in ['base', 'variable', 'wind_energy', 'pv_energy', 'charge_rate', 'discharge_rate', 'variable_energy', 'yearly_store_min', 'yearly_store_max']:
        if col not in df.columns:
            print('Warning {} missing, setting to zero'.format(col))
            df[col] = 0.0

    tolerence = 0.01
    if args.pv:
        df = df[(df['f_pv'] > args.pv-tolerence) & (df['f_pv'] < args.pv+tolerence)]
    if args.wind:
        df = df[(df['f_wind'] > args.wind-tolerence) & (df['f_wind'] < args.wind+tolerence)]

    filename = os.path.basename(path)
    scenario = filename[-7:]

    path = output_dir + 'settings' + scenario
    if exists(path):
        setting = readers.read_settings(path)
        if not 'normalise' in setting:
            setting['normalise'] = 818387.7082191781
        if not 'hourly' in setting:
            setting['hourly'] = 'False'
        if not 'baseload' in setting:
            setting['baseload'] = 0.0
        if not 'etad' in setting:
            setting['etad'] = 0.0
    else:
        setting = {'storage' : 'kf', 'baseload' : '0.0', 'start' : 1980, 'end': 2019, 'hourly': 'False', 'normalise' : 818387.7082191781, 'etad' : 0.0, 'eta' : 0.80 }

    # Override normalise factor if requested.
    if args.normalise > 0.0:
        setting['normalise'] = args.normalise

    # get labels file if it is present
    label = scenario[0:3]
    path = output_dir + 'labels' + label + '.txt'
    if exists(path):
        label_file = open(path, 'r')
        labels[label] = label_file.readline().rstrip()
        label_file.close()
    else:
        labels[label] = label

    # get demand
    path = output_dir + 'demand' + scenario
    demand = pd.read_csv(path, header=0, index_col=0).squeeze()
    demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
    # set total demand
    normalise_factor = float(setting['normalise'])
    demand = demand * normalise_factor
    total_demand = demand.sum() * 1e3
#   print('DEBUG {} factor {} total_demand {}'.format(scenario, normalise_factor, total_demand))

    # calculate efficiencies
    if float(setting['etad']) > 0:
        etad = float(setting['etad']) / 100.0
        eta = float(setting['eta']) / 100.0
    else:
        eta = math.sqrt(float(setting['eta']) / 100.0 )
        etad = eta

    # use the input energy for charge rate, not the stored energy
    #   charge_rate is:
    #     for daily series, the maximum energy charge in days in a day
    #     for hourly series, the maximum energy charge in hours in an hour
    if args.inrate:
        df['charge_rate'] = df['charge_rate'] / ( eta * etad )
        df['discharge_rate'] = df['discharge_rate'] / ( eta * etad )

    # calculate cost and energy
    n_years = int(setting['end']) - int(setting['start']) + 1
    # for hourly this is one hours energy
    one_day = float(setting['normalise'])
    hourly = setting['hourly']=='True'
    storage.generation_cost(df, 'caes', one_day, total_demand, n_years, hourly, 'both', 'B'  )

    # calculate energy
    df['energy'] = df['wind_energy'] + df['pv_energy']
    # calculate wind energy fraction
    df['fraction'] = df['wind_energy'] / df['energy']

    # each row of dataframe
    for index, row in df.iterrows():
        # rates in days per day or days per hour
        if args.units == 'days':
            factor = 1
            if hourly:
                factor = factor / 24
            charge_rate =  row['charge_rate'] * factor
            discharge_rate = row['discharge_rate'] * factor
            discharge = row['discharge']
            charge = row['charge']
        else:
            number_of_days = n_years * 365.25
            # rates in GW ??
            charge_rate = storage.days2capacity(row['charge_rate'], one_day * 1e-3, False)
            discharge_rate = storage.days2capacity(row['discharge_rate'], one_day * 1e-3, False)
            discharge = storage.days2energy(row['discharge'], one_day , number_of_days, True)
            charge = storage.days2energy(row['charge'], one_day , number_of_days, True)
        if not args.nolist:
            print('{}  {:.1f}    {:.1f}  {:4.1f} {:4.1f} {:5.2f}  {:5.2f}      {:.3f} {:.2f} {:.2f} {:.2f}     {:.3f} {:.3f}      {:.3f}  {:.2f}'.format(scenario[0:3], row['f_wind'], row['f_pv'], row['storage'], days2twh(row['storage'], one_day, hourly), charge_rate, discharge_rate, row['cost'], row['wind_energy'], row['pv_energy'], row['fraction'], row['energy'], discharge, charge, row['last'] ) )

    if args.fit:
        print('Doing regression {} '.format(scenario[0:3] ))
        coeffs = regression(df[['wind_energy', 'pv_energy']].values, df[['storage']].values)
        print('FIT', coeffs)

    if args.pstore:
#       path = output_dir + 'store' + scenario
#       store = pd.read_csv(path, header=0, index_col=0, squeeze=True)
#       store.index = pd.DatetimeIndex(pd.to_datetime(store.index).date)
#       if hourly:
#           store = store / 24
#       stores[scenario[0:3]] = store
        store = get_series('store', hourly, True)
        if hourly:
            store = store / 24
        stores[scenario[0:3]] = store
        duration = get_series('duration', hourly, False)
        if hourly:
            duration = duration / 24
            duration.index = duration.index / 24
        durations[scenario[0:3]] = duration

if args.pstore:
    for label,store in stores.items():
        if args.rolling >0:
            store = store.rolling(args.rolling, min_periods=1).mean()
        store.plot(label='Store size: {}'.format(labels[label]) )

    plt.xlabel('Time')
    plt.ylabel('Storage days')
    plt.title('Store history ')
    plt.legend(loc='lower center', fontsize=15)
    plt.show()

    for label,store in stores.items():
        store_sorted = store.sort_values(ascending=False)
        if args.rolling >0:
            store_sorted = store_sorted.rolling(args.rolling, min_periods=1).mean()
        store_sorted.plot(label='Store size: {}'.format(labels[label]), use_index=False )

    plt.xlabel('Time sorted by state of charge')
    plt.ylabel('State of charge (days)')
    plt.title('Store history sorted by state of charge')
    plt.legend(loc='lower center', fontsize=15)
    plt.show()

    for label,store in durations.items():
        if args.rolling >0:
            store = store.rolling(args.rolling, min_periods=1).mean()
        store.plot(label='Store duration: {}'.format(labels[label]) )

    plt.xlabel('State of Charge (stored energy) in days')
    plt.ylabel('Time in days store contained this amount of energy')
    plt.title('Store Duration ')
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

