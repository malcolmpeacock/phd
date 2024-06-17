# Plot kf constant storage and MP on same plot

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

# custom code
import stats
import readers
import storage
import bilinear2 as bil

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--last', action="store_true", dest="last", help='Only include configs which ended with store full', default=False)
parser.add_argument('--newdays', action="store_true", dest="newdays", help='Use new set of storage days', default=False)
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
parser.add_argument('--dir', action="store", dest="dir", help='Directory for my files', default='kfig8')
parser.add_argument('--electric', action="store", dest="electric", help='Electricity H=historic, S=snythetic', default='H')
parser.add_argument('--data', action="store", dest="data", help='Data to use K=KF, S=MP shares, T=mp threshold', default='T')
parser.add_argument('--stype', action="store", dest="stype", help='Type of Storage: pumped, hydrogen, caes.', default='pumped')
parser.add_argument('--costmodel', action="store", dest="costmodel", help='Cost model A or B', default='A')
parser.add_argument('--shore', action="store", dest="shore", help='Wind to base cost on both, on, off . default = both ', default='both')
args = parser.parse_args()

lines = [25, 30, 40, 60]
if args.newdays:
    lines = [4, 5, 6, 7]
colours = ['red', 'yellow', 'green', 'blue']
etas = [75, 85]
styles = ['solid', 'dotted']
first = True

ecount=0
for eta in etas:
    print('eta {} '.format(eta) )
    count=0
    # read in mp shares data
    if args.data == 'S':
        mp = pd.read_csv("/home/malcolm/uclan/output/{}S{:02d}/sharesNN{}.csv".format(args.dir,eta,args.electric))
        settings = readers.read_settings("/home/malcolm/uclan/output/{}S{:02d}/settingsNN{}.csv".format(args.dir,eta,args.electric))
        n_years = 30
        storage.generation_cost(mp, args.stype, n_years, float(settings['normalise']), settings['hourly']=='True', args.shore, args.costmodel )
        mp['energy'] = mp['wind_energy'] + mp['pv_energy']
        mp['fraction'] = mp['wind_energy'] / mp['energy']
        for c in ['cost_gen', 'cost_store', 'yearly_store_min', 'yearly_store_max', 'lost', 'slost', 'area']:
            mp[c] = 1

        print('Synthetic time series {} values'.format(len(mp)))
        if args.last:
            mp = mp[mp['last']==0.0]
            print('Only use configs where store ends full {} values'.format(len(mp)))
    for line in lines:
        if args.data == 'K':
            # read kf data
            kf_filename = "/home/malcolm/uclan/data/kf/S{:02d}days{:02d}.csv".format(eta,line)
            s = pd.read_csv(kf_filename, header=0)
            s.columns = ['f_wind', 'f_pv', 'w', 'fg', 'storage']
            # sort the same way
            points = s.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
            x_var = 'f_wind'
            y_var = 'f_pv'
        if args.data == 'T':
            mp = pd.read_csv("/home/malcolm/uclan/output/{}/S{:02d}days{:02d}.csv".format(args.dir,eta,line))
            x_var = 'f_wind'
            y_var = 'f_pv'
            points = mp.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
        if args.data == 'S':
            points = storage.storage_line(mp,line, args.sline, 'f_wind', 'f_pv')
            x_var = 'f_wind'
            y_var = 'f_pv'
        #print(mp_line)
        if first:
            ax = points.plot(x=x_var,y=y_var,color=colours[count],linestyle=styles[ecount],label='Efficiency {} days storage {}'.format(eta, line))
            first = False
        else:
            points.plot(x=x_var,y=y_var,ax=ax,color=colours[count],linestyle=styles[ecount],label='Efficiency {} days storage {}'.format(eta, line))
        count+=1
    ecount+=1

plt.title('Constant storage lines {} {} {}'.format(args.dir, args.electric, args.data))
plt.xlabel('Wind Capacity ( days )')
plt.ylabel('Solar PV Capacity ( days )')
plt.show()
