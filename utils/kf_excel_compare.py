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
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--last', action="store_true", dest="last", help='Only use configs where store ends full', default=False)
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
parser.add_argument('--eta', action="store", dest="eta", help='Storage efficiency', default=75, type=int)
parser.add_argument('--etak', action="store", dest="etak", help='Storage efficiency for kf', default=0, type=int)
parser.add_argument('--dir', action="store", dest="dir", help='Directory for my files', default='fixed_scaleKF')
parser.add_argument('--electric', action="store", dest="electric", help='Electricity H=historic, S=snythetic', default='H')
parser.add_argument('--threshold', action="store_true", dest="threshold", help='Use contour files from threshold method') 
args = parser.parse_args()

# read in mp shares data
mp = pd.read_csv("/home/malcolm/uclan/output/{}/sharesEN{}S{:02d}.csv".format(args.dir,args.electric,args.eta))

if args.last:
    original = len(mp)
    mp = mp[mp['last']==0.0]
    print('Using only store full at end {} from {} '.format(len(mp), original))

lines = [25, 30, 40, 60]
colours = ['red', 'green', 'blue', 'purple']
first = True
count=0
kf_eta = args.eta
if args.etak != 0:
    kf_eta = args.etak

for line in lines:
    # read kf data
    kf_filename = "/home/malcolm/uclan/data/kf/S{:02d}days{:02d}.csv".format(kf_eta,line)
    s = pd.read_csv(kf_filename, header=0)
    s.columns = ['f_wind', 'f_pv', 'w', 'fg', 'storage']
    # sort the same way
    kf_line = s.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
    #print(kf_line)
    if args.threshold:
        mp = pd.read_csv("/home/malcolm/uclan/output/{}/S{:02d}days{:02d}.csv".format(args.dir,args.eta,line))
        x_var = 'f_wind'
        y_var = 'f_pv'
        mp_line = mp.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
    else:
        mp_line = storage.storage_line(mp, line, args.sline, 'f_wind', 'f_pv')
        x_var = 'Pw'
        y_var = 'Ps'
    if first:
        ax = mp_line.plot(x=x_var,y=y_var,color=colours[count],label='Storage MP {} days'.format(line))
        first = False
    else:
        mp_line.plot(x=x_var,y=y_var,ax=ax,color=colours[count],label='Storage MP {} days'.format(line))
    kf_line.plot(x='f_wind', y='f_pv', ax=ax,color=colours[count],linestyle='dotted',label='Storage KF {} days'.format(line))
    count+=1

plt.title('Constant storage lines {} {}'.format(args.dir, args.electric))
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()
