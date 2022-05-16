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
# process command line
parser = argparse.ArgumentParser(description='Plot methods vs gas smart meter data')
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()


# read in the data
output_dir = "/home/malcolm/uclan/data/advanced_metering/testing/"
filename = output_dir + 'methods.csv'
df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# convert from Wh to kWh

df = df * 0.001

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(df['B'], df['gas'], 'BDEW', 2, args.plot)
stats.print_stats(df['W'], df['gas'], 'Watson', 1, args.plot)
stats.print_stats(df['H'], df['gas'], 'HDD 12.8', 1, args.plot)
stats.print_stats(df['S'], df['gas'], 'HDD 15.5  ', 1, args.plot)

stats.monthly_stats_header()
m_b = stats.monthly_stats(df['B'], df['gas'], 'BDEW')
m_w = stats.monthly_stats(df['W'], df['gas'], 'Watson')
m_h = stats.monthly_stats(df['H'], df['gas'], 'HDD 12.8')
m_s = stats.monthly_stats(df['S'], df['gas'], 'HDD 15.5')

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

if args.rolling == 0:
    s_gas = df['gas']
    s_b = df['B']
    s_w = df['W']
    s_h = df['H']
    s_s = df['S']
else:
    window = args.rolling
    s_gas = df['gas'].rolling(window, min_periods=1).mean()
    s_b = df['B'].rolling(window, min_periods=1).mean()
    s_w = df['W'].rolling(window, min_periods=1).mean()
    s_h = df['H'].rolling(window, min_periods=1).mean()
    s_s = df['S'].rolling(window, min_periods=1).mean()
    print('Rolling average window {} '.format(window))
    print(s_gas)

s_gas.plot(label='Heat Demand from gas smart meters', color='blue')
s_b.plot(label='Heat Demand BDEW', color='red')
s_w.plot(label='Heat Demand Watson', color='green')
s_h.plot(label='Heat Demand HDD 12.8', color='purple')
s_s.plot(label='Heat Demand HDD 15.5', color='orange')
plt.title('Comparison of Heat Demand Series generated from temperature and Gas data')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Heat Demand (kWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
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
