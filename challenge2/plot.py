
# contrib code
import sys
import pandas as pd
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

# main program

# process command line

# read in the data
output_dir = "/home/malcolm/uclan/challenge2/output/"
# merged data file ( demand, weather, augmented variables )
merged_filename = '{}merged_pre_august.csv'.format(output_dir)
df_in = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
# print(df_in)
# merged data file ( demand, weather, augmented variables )
maxmin_filename = '{}maxmin_pre_august.csv'.format(output_dir)
df_out = pd.read_csv(maxmin_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# plot the demand and max min
df_out['max_demand'].plot(label='max_demand')
df_out['min_demand'].plot(label='min_demand')
df_in['demand'].plot(label='half hourly demand')
plt.title('min and max demand')
plt.xlabel('Half Hour of the month', fontsize=15)
plt.ylabel('Demand (MW)', fontsize=15)
plt.legend(loc='lower left', fontsize=15)
plt.show()

