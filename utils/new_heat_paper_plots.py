# Plot the baseline electricity demand
# Print out stats about the baseline.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

import readers

# process command line
parser = argparse.ArgumentParser(description='Plot fig13 for given filename.')
parser.add_argument('--filename', action="store", dest="filename", help='filename ', default='hourly_hpall')
args = parser.parse_args()

# read historical electricity demand for reference year
# (power in MW for each hour of the year)

demand_filename = '/home/malcolm/uclan/output/new/demand/' + args.filename + '.csv'
demand = readers.read_demand(demand_filename, parm='demand_twh')

print('Demand: max {} min {} total {} '.format(demand.max(), demand.min(), demand.sum() ) )

# read baseline

baseline_filename = '/home/malcolm/uclan/output/new/demand/{}.baseline.csv'.format(args.filename)
baseline = readers.read_demand(baseline_filename, parm='demand_twh')

print('Baseline demand: max {} min {} total {} '.format(baseline.max(), baseline.min(), baseline.sum() ) )

# plot daily

demand = demand.resample('D').sum()
baseline = baseline.resample('D').sum()

demand.plot(color='orange', label='All heating provided by heat pumps' )
baseline.plot(color='purple', label='demand with heating electricity removed')
plt.title('Journal Paper Fig 13')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.show()
