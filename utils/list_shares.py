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
from os.path import exists
from skimage import measure

# custom code
import stats
import readers
import storage

def days2twh(days):
    twh = days / 0.81838
    return twh

def get_series(name, hourly, doIndex):
    path = output_dir + name + scenario
    store = pd.read_csv(path, header=0, index_col=0, squeeze=True)
    if doIndex:
        store.index = pd.DatetimeIndex(pd.to_datetime(store.index).date)
    return store

# main program

# process command line
parser = argparse.ArgumentParser(description='List shares files by directory')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--pstore', action="store_true", dest="pstore", help='Plot the sample store history ', default=False)
parser.add_argument('--pdemand', action="store_true", dest="pdemand", help='Plot the demand ', default=False)
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
parser.add_argument('--dir', action="store", dest="dir", help='Directory', default='adhoc')
parser.add_argument('--units', action="store", dest="units", help='Units', default='days')
parser.add_argument('--pv', action="store", dest="pv", help='A particular value of PV', default=None, type=float)
parser.add_argument('--wind', action="store", dest="wind", help='A particular value of wind', default=None, type=float)
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
demands={}
hydrogens={}
# loop round the files
output_dir = '/home/malcolm/uclan/output/' + args.dir + '/'

print('File f_wind f_pv storage    charge discharge cost  energy  ')
print('                 days twh   rate   rate            wind pv   fraction total discharged charged')
for path in glob.glob(output_dir + 'shares*.csv'):
    df = pd.read_csv(path, header=0, index_col=0)
    tolerence = 0.01
    if args.pv:
        df = df[(df['f_pv'] > args.pv-tolerence) & (df['f_pv'] < args.pv+tolerence)]
    if args.wind:
        df = df[(df['f_wind'] > args.wind-tolerence) & (df['f_wind'] < args.wind+tolerence)]

    filename = os.path.basename(path)
    scenario = filename[-7:]

    setting = readers.read_settings(output_dir + 'settings' + scenario )

    # calculate cost and energy
    n_years = int(setting['end']) - int(setting['start']) + 1
    # for hourly this is one hours energy
    one_day = float(setting['normalise'])
    hourly = setting['hourly']=='True'
    storage.generation_cost(df, 'caes', one_day, n_years, hourly, 'both', 'B'  )

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
            # rates in GW
            charge_rate = storage.days2capacity(row['charge_rate'], one_day * 1e-3, False)
            discharge_rate = storage.days2capacity(row['discharge_rate'], one_day * 1e-3, False)
            discharge = storage.days2energy(row['discharge'], one_day , number_of_days, False)
            charge = storage.days2energy(row['charge'], one_day , number_of_days, False)
        print('{}  {:.1f}    {:.1f}  {:4.1f} {:4.1f} {:5.2f}  {:5.2f}      {:.3f} {:.2f} {:.2f} {:.2f}     {:.3f} {:.3f}      {:.3f}'.format(scenario[0:3], row['f_wind'], row['f_pv'], row['storage'], days2twh(row['storage']), charge_rate, discharge_rate, row['cost'], row['wind_energy'], row['pv_energy'], row['fraction'], row['energy'], discharge, charge ) )

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
        store.plot(label='Store size: {}'.format(label) )

    plt.xlabel('Time')
    plt.ylabel('Storage days')
    plt.title('Store history ')
    plt.legend(loc='lower center', fontsize=15)
    plt.show()

    for label,store in stores.items():
        store_sorted = store.sort_values(ascending=False)
        if args.rolling >0:
            store_sorted = store_sorted.rolling(args.rolling, min_periods=1).mean()
        store_sorted.plot(label='Store size: {}'.format(label), use_index=False )

    plt.xlabel('Time sorted by state of charge')
    plt.ylabel('State of charge (days)')
    plt.title('Store history sorted by state of charge')
    plt.legend(loc='lower center', fontsize=15)
    plt.show()

    for label,store in durations.items():
        if args.rolling >0:
            store = store.rolling(args.rolling, min_periods=1).mean()
        store.plot(label='Store duration: {}'.format(label) )

    plt.xlabel('State of Charge (stored energy) in days')
    plt.ylabel('Time in days store contained this amount of energy')
    plt.title('Store Duration ')
    plt.legend(loc='upper left', fontsize=15)
    plt.show()
