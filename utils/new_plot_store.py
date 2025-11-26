# Plot the baseline electricity demand
# Print out stats about the baseline.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

import readers

base_dir = '/home/malcolm/uclan/output/new/kf2/'
scenario = 'hp4109'
# read demand

demand_filename = base_dir + 'demand' + scenario + '.csv'
demand = readers.read_demand(demand_filename, parm='demand_twh')

print('Demand: average {:.2f} total {:.2f} '.format(demand.sum()/40.0, demand.sum() ) )

# read store

store_filename = base_dir + 'store' + scenario + '.csv'
store = readers.read_demand(store_filename, parm='store')

print('store: mean {:.2f} total {:.2f} '.format(store.sum()/40.0, store.sum() ) )

# plot daily

demand = demand.resample('D').sum()
store = store.resample('D').sum()

demand.plot(color='purple', label='Demand')
store.plot(color='orange', label='Store State of Charge' )
#plt.title('Mix Paper Fig 3')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.show()

