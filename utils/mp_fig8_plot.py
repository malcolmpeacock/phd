# Recreate figure 8

# library stuff
import sys
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy import interpolate
import math
from os.path import exists

# custom code
import stats
import readers
import storage
import bilinear2 as bil

def get_storage_line(df, storage_model, days, wind_parm='f_wind', pv_parm='f_pv'):
    if storage_model == 'new':
        storage_line = df[df['storage'] == days]
        storage_line = storage_line[['f_pv','f_wind','last']]
#       storage_line.columns = ['Ps', 'Pw', 'last']
        storage_line = storage_line.sort_values(['f_wind', 'f_pv'], ascending=[True, True])
    else:
        storage_line = storage.storage_line(df, days, args.sline, wind_parm, pv_parm)
    return storage_line


# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--low', action="store_true", dest="low", help='Low numbers of days', default=False)
parser.add_argument('--historic', action="store_true", dest="historic", help='Historic time series', default=False)
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
parser.add_argument('--dir', action="store", dest="dir", help='Directory for my files', default='allS')
parser.add_argument('--scenario', action="store", dest="scenario", help='Scenario E=existing, N=no heat', default='N')
parser.add_argument('--stype', action="store", dest="stype", help='Type of Storage: pumped, hydrogen, caes.', default='pumped')
parser.add_argument('--model', action="store", dest="model", help='Storage Model', default='old')
parser.add_argument('--last', action="store_true", dest="last", help='Only include configs which ended with store full', default=False)
parser.add_argument('--shore', action="store", dest="shore", help='Wind to base cost on both, on, off . default = both ', default='both')
parser.add_argument('--title', action="store", dest="title", help='Override the plot title', default=None)
parser.add_argument('--costmodel', action="store", dest="costmodel", help='Cost model A or B', default='A')

args = parser.parse_args()

demand_type = 'S'
if args.historic:
    demand_type = 'H'

lines = [25, 30, 40, 60]
if args.low:
    lines = [4, 5, 6, 7]
colours = ['red', 'yellow', 'green', 'blue']
etas = [75, 85]
styles = ['solid', 'dotted']
first = True
output_dir = "/home/malcolm/uclan/output"

ecount=0
for eta in etas:
    print('eta {} '.format(eta) )
    count=0
    # read in the settings file
    path = '{}/{}{:02d}/settings{}N{}.csv'.format(output_dir, args.dir, eta,args.scenario, demand_type)
    if exists(path):
        settings = readers.read_settings(path)
    else:
        settings = {'storage' : 'kf', 'baseload' : '0.0', 'start' : 1980, 'end': 2019, 'hourly': False }

    # read in mp shares data
    df = pd.read_csv("{}/{}{:02d}/shares{}N{}.csv".format(output_dir, args.dir,eta,args.scenario, demand_type))
    # calculate cost and energy
    n_years = int(settings['end']) - int(settings['start']) + 1
    storage.generation_cost(df, args.stype, n_years, float(settings['normalise']), settings['hourly']=='True', args.shore, args.costmodel )
    df['energy'] = df['wind_energy'] + df['pv_energy']
    df['fraction'] = df['wind_energy'] / df['energy']
    if args.last:
        df = df[df['last']==0.0]
        print('Only use configs where store ends full {} values'.format(len(df)))

    print('Synthetic time series {} values'.format(len(df)))
    for line in lines:
#       points = storage.storage_line(df,line, args.sline, 'f_wind', 'f_pv')
        points = get_storage_line(df, args.model, line, 'f_wind', 'f_pv')
        print('Line {} points {} last {} to {} zero {} '.format(line, len(points), points['last'].min(), points['last'].max(), len(points[points['last']==0.0]) ) )
#       print(points)
        x_var = 'f_wind'
        y_var = 'f_pv'
        if first:
            ax = points.plot(x=x_var,y=y_var,color=colours[count],linestyle=styles[ecount],label='Efficiency {} days storage {}'.format(eta, line))
            first = False
        else:
            points.plot(x=x_var,y=y_var,ax=ax,color=colours[count],linestyle=styles[ecount],label='Efficiency {} days storage {}'.format(eta, line))
        count+=1
    ecount+=1

if args.title:
    plt.title(args.title)
else:
    plt.title('Constant storage lines {} scenario {}'.format(args.dir, args.scenario ))
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()
