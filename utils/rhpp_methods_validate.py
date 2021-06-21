# python script to validate the heat demand methods against heat pump data

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
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
filename = output_dir + 'methods.csv'
df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# convert from Wh to kWh

df = df * 0.001

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(df['B'], df['pumps'], 'BDEW', 2, True)
stats.print_stats(df['W'], df['pumps'], 'Watson', 1, True)
stats.print_stats(df['H'], df['pumps'], 'HDD 12.8', 1, True)
stats.print_stats(df['S'], df['pumps'], 'HDD 15.5  ', 1, True)

stats.monthly_stats_header()
m_b = stats.monthly_stats(df['B'], df['pumps'], 'BDEW')
m_w = stats.monthly_stats(df['W'], df['pumps'], 'Watson')
m_h = stats.monthly_stats(df['H'], df['pumps'], 'HDD 12.8')
m_s = stats.monthly_stats(df['S'], df['pumps'], 'HDD 15.5')

plt.plot(range(12), m_b, label='BDEW')
plt.plot(range(12), m_w, label='Watson')
plt.plot(range(12), m_h, label='HDD 12.8')
plt.plot(range(12), m_s, label='HDD 15.5')
plt.xlabel('Month of the year', fontsize=15)
plt.ylabel('nRMSE', fontsize=15)
plt.title('Monthly variation in nRMSE')
plt.legend(loc='upper right', fontsize=15)
plt.show()


# output plots

df['pumps'].plot(label='Heat Demand from measured heat pumps', color='blue')
df['B'].plot(label='Heat Demand BDEW', color='red')
df['W'].plot(label='Heat Demand Watson', color='green')
df['H'].plot(label='Heat Demand HDD 12.8', color='purple')
df['S'].plot(label='Heat Demand HDD 15.5', color='orange')
plt.title('Comparison of Synthetic Heat Demand Series and Measured heat series')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Heat Demand (kWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()

# load duration curves

sorted_pumps = df['pumps'].sort_values(ascending=False)
sorted_B = df['B'].sort_values(ascending=False)
sorted_W = df['W'].sort_values(ascending=False)
sorted_H = df['H'].sort_values(ascending=False)
sorted_S = df['S'].sort_values(ascending=False)

sorted_pumps.plot(label='Heat Demand Measured)', use_index=False, color='blue')
sorted_B.plot(label='Heat Demand BDEW', use_index=False, color='red')
sorted_W.plot(label='Heat Demand Watson', use_index=False, color='green')
sorted_H.plot(label='Heat Demand HDD 12.8', use_index=False, color='purple')
sorted_S.plot(label='Heat Demand HDD 15.5', use_index=False, color='orange')
plt.title('Heat Demand Methods sorted by demand')
plt.xlabel('Day sorted by demand', fontsize=15)
plt.ylabel('Daily Heat Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(rotation=0)
plt.show()
