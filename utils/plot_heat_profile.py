# python script to plot hourly profiles in heat_series.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
# custom code
import stats

# main program

# read in the data
profile_dir = "/home/malcolm/uclan/tools/python/scripts/heat/input/hourly/"

profile = 'rhpp'
filename = profile_dir + profile + '/hourly_factors_SFH.csv'
rhpp = pd.read_csv(filename, header=0, sep=',', index_col=0)

profile = 'bdew'
filename = profile_dir + profile + '/hourly_factors_SFH.csv'
bdew = pd.read_csv(filename, header=0, sep=',', index_col=0)

profile = 'flat'
filename = profile_dir + profile + '/hourly_factors_SFH.csv'
flat = pd.read_csv(filename, header=0, sep=',', index_col=0)

# normalize check
total_rhpp = rhpp['-5'].sum()
total_bdew = bdew['-5'].sum()
total_flat = flat['-5'].sum()
print('Total rhpp {:.2f} bdew {:.2f} flat {:.2f}'.format(total_rhpp, total_bdew, total_flat) )

# output plots

rhpp['-5'].plot(label='rhpp', color='blue')
bdew['-5'].plot(label='bdew', color='red')
flat['-5'].plot(label='flat', color='green')
plt.title('Hourly profiles heating -5')
plt.xlabel('Hour of the day', fontsize=15)
plt.ylabel('Percentage Heat Demand', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()


rhpp['10'].plot(label='rhpp', color='blue')
bdew['10'].plot(label='bdew', color='red')
flat['10'].plot(label='flat', color='green')
plt.title('Hourly profiles heating 10')
plt.xlabel('Hour of the day', fontsize=15)
plt.ylabel('Percentage Heat Demand', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()
