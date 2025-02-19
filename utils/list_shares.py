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

def bounds(df, bvars):
    svars = bvars.split(',')
    b_var = svars[0]
    v1 = svars[1]
    v2 = svars[2]
    bdf = df[[b_var, v1, v2]]
    print(bdf)
    bdf.sort_values(by=[b_var], ascending=True, inplace=True)
    bdf['d1'] = bdf[v1].diff() / bdf[v1].max()
    bdf['d1'] = bdf['d1'].abs()
    bdf['d2'] = bdf[v2].diff() / bdf[v2].max()
    bdf['d2'] = bdf['d2'].abs()
    bdf['max'] = bdf[['d1', 'd2']].max(axis=1)
    bdf.fillna(0, inplace=True)
    plt.plot(bdf[b_var].values, bdf['max'] )
    plt.ylabel('max change of {} and {}'.format(v1, v2), fontsize=15)
    plt.xlabel(b_var, fontsize=15)
    plt.show()

def norm(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm

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

def filter(df, variable, value, operator):
    tolerence = 0.01
    if operator == 'equal':
        df = df[(df[variable] > value-tolerence) & (df[variable] < value+tolerence)]
    if operator == 'greater':
        df = df[(df[variable] > value)]
    if operator == 'less':
        df = df[(df[variable] < value)]
    return df
# main program

# process command line
parser = argparse.ArgumentParser(description='List shares files by directory')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--pstore', action="store_true", dest="pstore", help='Plot the sample store history ', default=False)
parser.add_argument('--pdemand', action="store_true", dest="pdemand", help='Plot the demand ', default=False)
parser.add_argument('--fit', action="store_true", dest="fit", help='Try fiting a curve', default=False)
parser.add_argument('--debug', action="store_true", dest="debug", help='Print debug stuff', default=False)
parser.add_argument('--nolist', action="store_true", dest="nolist", help='Do not list values', default=False)
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
parser.add_argument('--dir', action="store", dest="dir", help='Directory', default='adhoc')
parser.add_argument('--units', action="store", dest="units", help='Units', default='days')
parser.add_argument('--normalise', action="store", dest="normalise", help='Normalise factor to override the one from settings', type=float, default=0.0)
parser.add_argument('--pv', action="store", dest="pv", help='A particular value of PV', default=None, type=float)
parser.add_argument('--wind', action="store", dest="wind", help='A particular value of wind', default=None, type=float)
parser.add_argument('--gradient', action="store", dest="gradient", help='A particular value of gradient', default=None, type=float)
parser.add_argument('--storage', action="store", dest="storage", help='A particular value of storage', default=None, type=float)
parser.add_argument('--operator', action="store", dest="operator", help='Operator for selecting rows: equal, less, greater', default='equal' )
parser.add_argument('--bounds', action="store", dest="bounds", help='Plot bounds based on variables passed in', default=None )
parser.add_argument('--inrate', action="store_true", dest="inrate", help='Base the charge rate on the energy input, not energy stored', default=False)
args = parser.parse_args()

stores={}
durations={}
hydrogens={}
labels={}
max_charge_rate={}
max_discharge_rate={}
max_f_pv={}
min_f_pv={}
max_f_wind={}
min_f_wind={}
max_storage={}
min_storage={}

# loop round the files
output_dir = '/home/malcolm/uclan/output/' + args.dir + '/'

if not args.nolist:
    print('File f_wind f_pv storage    charge discharge cost  energy                                      last  lost slost sum load gradient')
    print('                 days twh   rate   rate            wind pv   fraction total discharged charged')
for path in glob.glob(output_dir + 'shares*.csv'):
    if args.debug:
        print('DBG: Reading {}'.format(path) )
    df = pd.read_csv(path, header=0, index_col=0)
    for col in ['base', 'variable', 'wind_energy', 'pv_energy', 'charge_rate', 'discharge_rate', 'variable_energy', 'yearly_store_min', 'yearly_store_max']:
        if col not in df.columns:
            print('Warning {} missing, setting to zero'.format(col))
            df[col] = 0.0

    tolerence = 0.01
#   gradient_vals = df[['wind_energy', 'pv_energy', 'storage']].diff()
#   gradient = np.gradient(gradient_vals, 0.1, 0.1)
#   print('gradient')
#   print(gradient)
#   print('x')
#   print(gradient[0])
#   magnitude = np.linalg.norm(gradient)
#   magnitude = np.sqrt(np.dot(gradient)
#   print('magnitude')
#   print(magnitude)
#       df = df[(df[variable] > value-tolerence) & (df[variable] < value+tolerence)]
    wind_vals = np.unique(df['f_wind'].values)
#   print(wind_vals)
    pv_vals = np.unique(df['f_pv'].values)
#   print(pv_vals)
    n_pvs=len(pv_vals)
    n_winds=len(wind_vals)
    storage_vals = np.zeros(shape=(n_winds,n_pvs))
#   print(storage_vals)
#   print('Dimensions {} w {} p'.format(n_winds,n_pvs))
    for w in range(n_winds):
        for p in range(n_pvs):
#           print('{} ,{} , {}, {}'.format(w,p, wind_vals[w], pv_vals[p]))
            storage_df=df[(df['f_pv'] == pv_vals[p]) & (df['f_wind'] == wind_vals[w])]
#           print(storage_df)
            storage_val=storage_df['storage']
#           print('storage_val')
#           print(storage_val)
            if len(storage_val)>0:
                sval=storage_val.values[0]
#               print('Settings {} for {} {}'.format(sval,w,p))
                storage_vals[w,p]=sval
#   print(storage_vals)
#   gradients=np.gradient(storage_vals, wind_vals, pv_vals)
    gradients=np.gradient(norm(storage_vals), norm(wind_vals), norm(pv_vals) )
#   print(len(wind_vals), len(pv_vals), np.shape(wind_vals), np.shape(pv_vals), np.shape(storage_vals), np.shape(gradients))
    if args.pv:
#       df = df[(df['f_pv'] > args.pv-tolerence) & (df['f_pv'] < args.pv+tolerence)]
        df = filter(df, 'f_pv', args.pv, args.operator)
    if args.wind:
#       df = df[(df['f_wind'] > args.wind-tolerence) & (df['f_wind'] < args.wind+tolerence)]
        df = filter(df, 'f_wind', args.wind, args.operator)
    if args.storage:
        df = filter(df, 'storage', args.storage, args.operator)

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
    load = demand.sum()
    # set total demand
    normalise_factor = float(setting['normalise'])
    demand = demand * normalise_factor
    total_demand = demand.sum() * 1e3

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

    # need to divide variable energy by number of values to get the mean
    # (should have been done in HvH )
    factor = 1
    if hourly:
        factor = 24
    df['charge'] = df['charge'] / ( n_years * 365.25 * factor )
    df['discharge'] = df['discharge'] / ( n_years * 365.25 * factor )
    df['variable_energy'] = df['variable_energy'] / ( n_years * 365.25 * factor )
    load = load / ( n_years * 365.25 * factor )

    # calculate the 'lost' energy
    df['all_energy'] = df['energy'] + df['variable_energy']
    battery_loss = df['charge'] * ( ( 1 - eta ) / eta) + (df['discharge']*(1-etad))
    generated = df['all_energy'] + df['base']
    # only true for existing
    df['lost'] = generated - load - battery_loss
    df['slost'] = battery_loss

    # check energy sum (should be 1.0, ie the load)
    df['energy_sum'] = df['wind_energy'] + df['pv_energy'] + df['base'] + df['variable_energy'] - df['lost'] - df['slost']

    max_charge_rate[label] = 0
    max_discharge_rate[label] = 0

    if args.bounds:
        bounds(df,args.bounds)

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
        if charge_rate > max_charge_rate[label]:
            max_charge_rate[label] = charge_rate
        if discharge_rate > max_discharge_rate[label]:
            max_discharge_rate[label] = discharge_rate

        # gradient
        gradient=0
        iw=np.where(wind_vals == row['f_wind'])[0][0]
        ip=np.where(pv_vals == row['f_pv'])[0][0]
        grad1=gradients[0][iw][ip]
        grad2=gradients[1][iw][ip]
        gradient=math.sqrt(grad1**2 + grad2**2)
        if not args.nolist:
            if (args.gradient and gradient>args.gradient) or not args.gradient:
                print('{}  {:.1f}    {:.1f}  {:4.1f} {:4.1f} {:5.2f}  {:5.2f}      {:.3f} {:.2f} {:.2f} {:.2f}     {:.3f} {:.3f}      {:.3f}   {:.2f}  {:.2f} {:.2f} {:.2f} {:.2f} {:4.1f}'.format(scenario[0:3], row['f_wind'], row['f_pv'], row['storage'], days2twh(row['storage'], one_day, hourly), charge_rate, discharge_rate, row['cost'], row['wind_energy'], row['pv_energy'], row['fraction'], row['energy'], discharge, charge, row['last'], row['lost'], row['slost'], row['energy_sum'], load, gradient ) )

    max_f_pv[label] = df['f_pv'].max()
    min_f_pv[label] = df['f_pv'].min()
    max_f_wind[label] = df['f_wind'].max()
    min_f_wind[label] = df['f_wind'].min()
    max_storage[label] = df['storage'].max()
    min_storage[label] = df['storage'].min()

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

print("Label charge_rate discharge_rate f_pv      f_wind")
print("      max         min            max  min  max  min  max  min")
for label in max_charge_rate:
    print("{}   {:4.1f}       {:4.1f}         {:4.1f} {:4.1f} {:4.1f} {:4.1f} {:4.1f} {:4.1f}".format(label, max_charge_rate[label], max_discharge_rate[label], max_f_pv[label], min_f_pv[label], max_f_wind[label], min_f_wind[label], max_storage[label], min_storage[label] ) )

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
        hours = np.arange(0, len(store_sorted))
        store_sorted.index = hours / 24
        store_sorted.plot(label='Store size: {}'.format(labels[label]) )

    plt.xlabel('Time (days) sorted by state of charge')
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

