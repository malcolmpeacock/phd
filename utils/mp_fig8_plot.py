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

# custom code
import stats
import readers
import storage
import bilinear2 as bil

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--low', action="store_true", dest="low", help='Low numbers of days', default=False)
parser.add_argument('--historic', action="store_true", dest="historic", help='Historic time series', default=False)
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
parser.add_argument('--dir', action="store", dest="dir", help='Directory for my files', default='allS')
parser.add_argument('--scenario', action="store", dest="scenario", help='Scenario E=existing, N=no heat', default='N')
parser.add_argument('--last', action="store_true", dest="last", help='Only include configs which ended with store full', default=False)
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

ecount=0
for eta in etas:
    print('eta {} '.format(eta) )
    count=0
    # read in mp shares data
    mp = pd.read_csv("/home/malcolm/uclan/output/{}{:02d}/shares{}N{}.csv".format(args.dir,eta,args.scenario, demand_type))
    if args.last:
        mp = mp[mp['last']==0.0]
        print('Only use configs where store ends full {} values'.format(len(mp)))

    print('Synthetic time series {} values'.format(len(mp)))
    for line in lines:
        points = storage.storage_line(mp,line, args.sline, 'f_wind', 'f_pv')
        print('Line {} points {} '.format(line, len(points) ) )
        x_var = 'Pw'
        y_var = 'Ps'
        if first:
            ax = points.plot(x=x_var,y=y_var,color=colours[count],linestyle=styles[ecount],label='Efficiency {} days storage {}'.format(eta, line))
            first = False
        else:
            points.plot(x=x_var,y=y_var,ax=ax,color=colours[count],linestyle=styles[ecount],label='Efficiency {} days storage {}'.format(eta, line))
        count+=1
    ecount+=1

plt.title('Constant storage lines {} scenario {}'.format(args.dir, args.scenario ))
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()
