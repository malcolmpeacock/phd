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
parser = argparse.ArgumentParser(description='Plot baseline for given filename.')
parser.add_argument('--year', action="store", dest="year", help='Year ', default='2018')
parser.add_argument('--filename', action="store", dest="filename", help='filename ', default='base2018')
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
args = parser.parse_args()

# read historical electricity demand for reference year
# (power in MW for each hour of the year)

demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
original_demand = readers.read_espeni(demand_filename, args.reference)

# convert to TWh
original_demand = original_demand * 1e-6

print('Historic demand: max {} min {} total {} '.format(original_demand.max(), original_demand.min(), original_demand.sum() ) )

# read baseline

baseline_filename = '/home/malcolm/uclan/output/new/baseline/{}.csv'.format(args.filename)
baseline = readers.read_demand(baseline_filename, parm='demand_twh')
# convert to TWh
#baseline = baseline * 1e-6

print('Baseline demand: max {} min {} total {} '.format(baseline.max(), baseline.min(), baseline.sum() ) )

# plot daily

original_demand = original_demand.resample('D').sum()
baseline = baseline.resample('D').sum()

original_demand.plot(color='blue', label='Historic electricity demand time series ' )
baseline.plot(color='purple', label='Electricity {} with heating electricity removed'.format(args.filename))
plt.title('Removing existing heating electricity from the daily electricty demand series')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.show()
