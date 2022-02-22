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
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
args = parser.parse_args()

# read kf data
kf_filename = "/home/malcolm/uclan/data/kf/Storage75.csv"
s75 = pd.read_csv(kf_filename, header=0)
s75.columns = ['f_wind', 'f_pv', 'w', 'fg', 'storage']

# sort the same way
kf = s75.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
#print(kf)

# read in mp shares data
mp = pd.read_csv("/home/malcolm/uclan/output/fixed_scaleKF/sharesENS.csv")
#mp = mp[mp['last']==mp['storage']]
#mp = mp.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
#print(mp)

threshold=0.6
lines = [25, 30, 40, 60]
colours = ['red', 'green', 'blue', 'purple']
first = True
count=0
for line in lines:
    kf_line = kf[(kf['storage']<line+threshold) & (kf['storage']>line-threshold)]
    #print(kf_line)
    mp_line = storage.storage_line(mp,line, args.sline, 'f_wind', 'f_pv')
    #print(mp_line)
    if first:
        ax = mp_line.plot(x='Pw',y='Ps',color=colours[count],label='Storage MP {} days'.format(line))
        first = False
    else:
        mp_line.plot(x='Pw',y='Ps',ax=ax,color=colours[count],label='Storage MP {} days'.format(line))
    kf_line.plot(x='f_wind', y='f_pv', ax=ax,color=colours[count],linestyle='dotted',label='Storage KF {} days'.format(line))
    count+=1

plt.title('Constant storage lines')
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()
