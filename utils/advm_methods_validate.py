# python script to validate the heat demand methods against gas meter data

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
output_dir = "/home/malcolm/uclan/data/advanced_metering/testing/"
filename = output_dir + 'methods.csv'
df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# convert from Wh to kWh

df = df * 0.001

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(df['B'], df['gas'], 'BDEW', 2, True)
stats.print_stats(df['W'], df['gas'], 'Watson', 1, True)
stats.print_stats(df['H'], df['gas'], 'HDD 12.8', 1, True)
stats.print_stats(df['S'], df['gas'], 'HDD 15.5  ', 1, True)

# output plots

df['gas'].plot(label='Measured heat demand', color='blue')
df['B'].plot(label='Heat Demand BDEW', color='red')
df['W'].plot(label='Heat Demand Watson', color='green')
df['H'].plot(label='Heat Demand HDD 12.8', color='purple')
df['S'].plot(label='Heat Demand HDD 15.5', color='orange')
plt.title('Comparison of Synthetic Heat Demand Series and Commerical Gas data')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Heat Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# load duration curves

sorted_gas = df['gas'].sort_values(ascending=False)
sorted_B = df['B'].sort_values(ascending=False)
sorted_W = df['W'].sort_values(ascending=False)
sorted_H = df['H'].sort_values(ascending=False)
sorted_S = df['S'].sort_values(ascending=False)

sorted_gas.plot(label='Heat Demand Measured)', use_index=False, color='blue')
sorted_B.plot(label='Heat Demand BDEW', use_index=False, color='red')
sorted_W.plot(label='Heat Demand Watson', use_index=False, color='green')
sorted_H.plot(label='Heat Demand HDD 12.8', use_index=False, color='purple')
sorted_S.plot(label='Heat Demand HDD 15.5', use_index=False, color='orange')
plt.title('Heat Demand Methods sorted by demand - measured gas')
plt.xlabel('Day sorted by demand', fontsize=15)
plt.ylabel('Daily Heat Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(rotation=0)
plt.show()
