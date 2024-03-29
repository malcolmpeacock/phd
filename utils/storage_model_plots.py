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

# model specific storage size

def model(net, capacity, start, eta):
    store_history=[]
    store = start
    store_history.append(store)
    for i in range(len(net)-1):
        if net[i]>0:
            store+=net[i] * eta
        else:
            store+=net[i] / eta

        if store>capacity:
            store = capacity
        if store<=0:
            print('Out of storage at {:.2f} for capacity {:.2f}'.format(i, capacity))
            return False, store_history
    if store>start:
        print('Viable Solution for capacity {:.2f} store {:.2f} back to start {:.2f}'.format(capacity, store, start))
    else:
        print('Non-Viable Solution for capacity {:.2f} as store end at {:.2f} below start {:.2f}'.format(capacity, store, start))
        return False, store_history
    return True, store_history

def find_storage(net, eta, a, b, c):
    count=0
    max_count=10
    last_viable = a+b
    next_try = b
    threshold = 0.2
    print('find_storage: last_viable {} next_try {} threshold {}'.format(last_viable, next_try, threshold) )
    while abs(last_viable - next_try) > threshold and count<max_count:
        print('last_viable {} next_try {} threshold {}'.format(last_viable, next_try, threshold) )
        viable, store = model(net, next_try, b, eta)
        if viable:
            last_viable = next_try
            next_try = next_try - ( last_viable - next_try ) / 2
        else:
            next_try = next_try + ( last_viable - next_try ) / 2
        count+=1

    if count >= max_count:
        print('finished without finding a solution at count {} '.format(count))
    return last_viable

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

if args.plot:
    plt.plot(time, wind, label='wind')
    plt.plot(time, demand, label='demand')
    plt.plot(time, pv, label='pv')
    plt.plot(time, net, label='net')
    plt.title('Demand and Supply ')
    plt.xlabel('Time (months)')
    plt.ylabel('Energy generation')
    plt.legend(loc='upper right')
    plt.show()

# original energy storage model

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

if args.plot:
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

min_store = -min_s[0]
with_diff = -( min_s[0] + s2[-1] )
capacities = [20, min_store, with_diff, with_diff+2, 25]
for capacity in capacities:
    print('Starts full')
    viable, hist = model(net, capacity, capacity, eta)
    print('Starts at 0.8')
    viable, hist = model(net, capacity, capacity*0.8, eta)
    print('Starts at 0.5')
    viable, hist = model(net, capacity, capacity*0.5, eta)

# new energy storage model ?
# don't know the upper limit of curtailment until we know the 
# capacity
# so perhaps set a loop as above which will converge on a viable
# solution ? or decide its not viable.
a = abs(s.min() )
b = s.max()
c = s[-1]
print('a {} b {} c {} '.format(a, b, c))
last_viable = find_storage(net, eta, a, b, c)
print('Found storage last_viable {}'.format(last_viable))
