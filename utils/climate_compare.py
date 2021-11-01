# python script to compare different models impacted by climate change.

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
output_dir = "/home/malcolm/uclan/output/40years/"

filename = output_dir + 'yearly.csv'
no_climate = pd.read_csv(filename, header=0, sep=',', index_col=0)

filename = output_dir + 'yearlyC.csv'
with_climate = pd.read_csv(filename, header=0, sep=',', index_col=0)

no_climate['storage'].plot(label='Yearly Storage without climate change adjustment', color='blue')
with_climate['storage'].plot(label='Yearly Storage with climate change adjustment', color='red')
plt.title('Impact of climate change adjustment on long term storage')
plt.xlabel('year', fontsize=15)
plt.ylabel('Days of storage', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()

diff = no_climate['storage'] - with_climate['storage']
print('Diff mean {} min {} max {}'.format(diff.mean(), diff.min(), diff.max() ))
