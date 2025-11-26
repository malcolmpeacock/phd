# Plot the baseline electricity demand
# Print out stats about the baseline.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

import readers

# read electricity with 2018 heating technology

#existing_filename = '/home/malcolm/uclan/output/new/demand/existing09.csv'
existing_filename = '/home/malcolm/uclan/output/new/demand/existing06.csv'
existing = readers.read_demand(existing_filename, parm='demand_twh')

print('Existing Demand: average {:.2f} total {:.2f} '.format(existing.sum()/40.0, existing.sum() ) )

# read hp41

#hp41_filename = '/home/malcolm/uclan/output/new/demand/hp4109.csv'
hp41_filename = '/home/malcolm/uclan/output/new/demand/hp4106.csv'
hp41 = readers.read_demand(hp41_filename, parm='demand_twh')

print('hp41 demand: mean {:.2f} total {:.2f} '.format(hp41.sum()/40.0, hp41.sum() ) )

# plot daily

existing = existing.resample('D').sum()
hp41 = hp41.resample('D').sum()

hp41.plot(color='purple', label='41% of heating provided by heat pumps')
existing.plot(color='orange', label='2018 heating technology' )
plt.title('Mix Paper Fig 3')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# read ev

#ev_filename = '/home/malcolm/uclan/output/new/demand/ev09.csv'
ev_filename = '/home/malcolm/uclan/output/new/demand/ev06.csv'
ev = readers.read_demand(ev_filename, parm='demand_twh')

print('ev demand: mean {:.2f} total {:.2f} '.format(ev.sum()/40.0, ev.sum() ) )

# plot daily

ev = ev.resample('D').sum()

ev.plot(color='purple', label='2018 heating with EVs')
existing.plot(color='orange', label='2018 heating technology' )
plt.title('Mix Paper Fig 4')
plt.xlabel('day of the year', fontsize=15)
plt.ylabel('Daily Electricity Demand (Twh)', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# print yearly
ev = ev.resample('Y').sum()
hp41 = hp41.resample('Y').sum()
existing = existing.resample('Y').sum()

print('Yearly Variation')
print('Demand   min    max    range  mean  years' )
print('Existing {:.2f} {:.2f} {:2.2f} {:.2f} {:}'.format(existing.min(), existing.max(), existing.max() - existing.min(), existing.mean(), len(existing)))
print('HP41     {:.2f} {:.2f} {:2.2f} {:.2f} {:}'.format(hp41.min(), hp41.max(), hp41.max() - hp41.min(), hp41.mean(), len(hp41)))
print('EVS      {:.2f} {:.2f} {:2.2f} {:.2f} {:}'.format(ev.min(), ev.max(), ev.max() - ev.min(), ev.mean(), len(ev) ))
