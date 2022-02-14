# Plots to illustrate model to find energy storage.

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

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
args = parser.parse_args()

# create data: time, and demand time series
xt = 0
time = np.arange(0,60)
demand = np.cos( (time + xt) / 2 ) + 1
demand = demand * 10

wind = demand * 0.65

pv = np.sin( (time + xt) / 2 ) + 1
pv = pv * 4.5

net = pv + wind - demand 

plt.plot(time, wind, label='wind')
plt.plot(time, demand, label='demand')
plt.plot(time, pv, label='pv')
plt.plot(time, net, label='net')
plt.title('Demand and Supply ')
plt.xlabel('Time (months)')
plt.ylabel('Energy generation')
plt.legend(loc='upper right')
plt.show()


eta=0.75
store=0
store_history=[]
store2=0
store_history2=[]

# only go to -1 so that the store array is the same size as time
# and the store starts at zero
for i in range(len(net)-1):
    store_history.append(store)
    store_history2.append(store2)
    if net[i]>0:
        store+=net[i] * eta
        store2+=net[i] * eta
    else:
        store+=net[i] / eta
        store2+=net[i] / eta

    if store2>0:
        store2 = 0

store_history.append(store)
store_history2.append(store2)


s = np.array(store_history)
s2 = np.array(store_history2)

# zero line
zero_s = np.full(2,0)
zero_t = np.array([0,60])
# max line
max_s = np.full(2,np.max(s))
max_t = np.array([0,60])
# min line
min_s = np.full(2,np.min(s2))
min_t = np.array([0,60])

# 
print('Max line {} Min line {} Last point {}'.format(max_s[0], min_s[0], s2[-1]) )

plt.plot(time, s, linestyle='dashed')
plt.plot(time, s2)
plt.plot(zero_t, zero_s, linestyle='dotted',label='Store Full')
plt.plot(max_t, max_s, linestyle='dotted',label='Store Excess')
plt.plot(min_t, min_s, linestyle='dotted',label='Store Empty')
plt.title('Storage size')
plt.xlabel('Time (months)')
plt.ylabel('Storage size (days)')
plt.legend(loc='lower left')
plt.show()
