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

# Functions to convert to and from actual capacity based on demandNNH.csv
def cf2gw(x):
    return x * generation_capacity
def gw2cf(x):
    return x / generation_capacity

def days2twh(days):
    twh = days / 0.81838
    return twh

# main program

# process command line
parser = argparse.ArgumentParser(description='List shares files by directory')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--pstore', action="store_true", dest="pstore", help='Plot the sample store history ', default=False)
parser.add_argument('--pdemand', action="store_true", dest="pdemand", help='Plot the demand ', default=False)
parser.add_argument('--dir', action="store", dest="dir", help='Directory', default='adhoc')
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

# loop round the files
output_dir = '/home/malcolm/uclan/output/individuals/' + args.dir + '/'

print('File f_pv f_wind storage    charge discharge cost  energy ')
print('                 days twh   rate   rate            wind pv   fraction total')
for path in glob.glob(output_dir + 'shares*.csv'):
    df = pd.read_csv(path, header=0, index_col=0)

    filename = os.path.basename(path)
    scenario = filename[-7:]

    setting = readers.read_settings(output_dir + 'settings' + scenario )

    # calculate cost and energy
    n_years = int(setting['end']) - int(setting['start']) + 1
    storage.generation_cost(df, 'caes', float(setting['normalise']), n_years, setting['hourly']=='True', 'both', 'B'  )

    # calculate energy
    df['energy'] = df['wind_energy'] + df['pv_energy']
    # calculate wind energy fraction
    df['fraction'] = df['wind_energy'] / df['energy']

    # each row of dataframe
    for index, row in df.iterrows():
        charge_rate = ( row['charge_rate'] * 1000 ) / 24.0
        discharge_rate = ( row['discharge_rate'] * 1000 ) / 24.0
        print('{}  {:.1f}  {:.1f}    {:.1f} {:.1f}  {:.2f}   {:.2f}      {:.3f} {:.2f} {:.2f} {:.2f}     {:.3f}  '.format(scenario[0:3], row['f_pv'], row['f_wind'], row['storage'], days2twh(row['storage']), charge_rate, discharge_rate, row['cost'], row['wind_energy'], row['pv_energy'], row['fraction'], row['energy'] ) )


