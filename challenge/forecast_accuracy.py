# python script to assess the acuracy of the reanlysis ghi.

# contrib code
import sys
import pandas as pd
#from datetime import datetime
#from datetime import timedelta
#import pytz
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import argparse
import numpy as np
import math
#import torch
#import torch.nn as nn
# Import tensor dataset & data loader
#from torch.utils.data import TensorDataset, DataLoader
# Import nn.functional
#import torch.nn.functional as F

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Create pv forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print('sun2 {} vs pv_ghi {}'.format(df['sun2'].sum(), df['pv_ghi'].sum() ))
utils.print_metrics(df['pv_ghi'], df['sun2'])
print('sunw {} vs pv_ghi {}'.format(df['sunw'].sum(), df['pv_ghi'].sum() ))
utils.print_metrics(df['pv_ghi'], df['sunw'])
print(df)

# forecast accuracy measurement. -ve values = under predict
# acc2 = np.abs(df['pv_ghi'].values - df['sun2'].values)
acc2 = df['sun2'].values - df['pv_ghi'].values
df['acc2'] = acc2

# forecast disagreement measurement.
df['sundis'] =  (df['sun1'] - df['sun2']).abs() + (df['sun2'] - df['sun5']).abs() + (df['sun5'] - df['sun6']).abs() + (df['sun6'] - df['sun1']).abs()

# monthly mean temperature
month_temps = df['tempw'].resample('M', axis=0).mean()
print(month_temps)

# monthly forecast accuracy
month_acc = df['acc2'].resample('M', axis=0).mean()
print(month_acc)

if args.plot:
    plt.scatter(month_temps.values, month_acc.values, s=12, color='blue')
    plt.title('Irradiance Forecast inaccuracy vs monthly temperature')
    plt.xlabel('mean monthly temperature', fontsize=15)
    plt.ylabel('weather inaccuracy', fontsize=15)
    plt.show()

    plt.scatter(df['k'].values, df['acc2'].values, s=12, color='blue')
    plt.title('Irradiance Forecast inaccuracy vs K')
    plt.xlabel('K period', fontsize=15)
    plt.ylabel('weather inaccuracy', fontsize=15)
    plt.show()

    plt.scatter(df['sun2'].values, df['acc2'].values, s=12, color='blue')
    plt.title('Irradiance Forecast inaccuracy vs irradiance')
    plt.xlabel('Irradiance', fontsize=15)
    plt.ylabel('weather inaccuracy', fontsize=15)
    plt.show()

    plt.scatter(df['temp2'].values, df['acc2'].values, s=12, color='blue')
    plt.title('Irradiance Forecast inaccuracy vs temperature')
    plt.xlabel('Temperature (degrees C)', fontsize=15)
    plt.ylabel('weather inaccuracy', fontsize=15)
    plt.show()

    plt.scatter(df.index.values, df['acc2'].values, s=12, color='blue')
    plt.title('Irradiance Forecast inaccuracy vs hour of the year')
    plt.xlabel('time of year', fontsize=15)
    plt.ylabel('weather inaccuracy', fontsize=15)
    plt.show()

    plt.scatter(df['sundis'].values, df['acc2'].values, s=12, color='blue')
    plt.title('Irradiance Forecast inaccuracy vs stations disagreement')
    plt.xlabel('stations disagreement', fontsize=15)
    plt.ylabel('weather inaccuracy', fontsize=15)
    plt.show()

    month_acc.plot(label='inaccuracy', color='green')
    plt.title('Variation with month' )
    plt.xlabel('Month of the year', fontsize=15)
    plt.ylabel('Inaccuracy (w/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()


