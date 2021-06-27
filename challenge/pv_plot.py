# python script to investigate pv data.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
import pvlib
import statsmodels.api as sm

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Plot PV data.')
parser.add_argument('set', help='weather file eg set0')

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

monthly_pv_ghi = df['pv_ghi'].resample('M', axis=0).mean()
monthly_sun2 = df['sun2'].resample('M', axis=0).mean()

monthly_pv_ghi.plot(label='PV GHI')
monthly_sun2.plot(label='weather GHI sun2')
plt.title('Monthly PV and weather GHI')
plt.xlabel('Month', fontsize=15)
plt.ylabel('GHI (W/m2)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()
