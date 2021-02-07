# See how the data for the data challange correlates daily

# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import seaborn as sn

parser = argparse.ArgumentParser(description='Data correlations.')
parser.add_argument('set', help='data files eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# demand data
demand_filename = '{}demand_fixed_{}.csv'.format(output_dir, dataset)
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(demand)

# weather data
weather_filename = '{}weather_{}.csv'.format(output_dir, dataset)
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(weather)

# stick it all together
df = pd.concat([demand, weather], axis=1)
df['k'] = (df.index.hour * 2) + (df.index.minute / 30) + 1
df['k'] = df['k'].astype(int)
print(df['k'])

mean_temp2 = df['temp2'].resample('D', axis=0).mean()
demand_mean = df['demand'].resample('D', axis=0).mean()
demand_max = df['demand'].resample('D', axis=0).max()
demand_min = df['demand'].resample('D', axis=0).min()
# find which k has the highest daily temp
maxk=[]
for day in demand_max.index.date:
     day_data = df.loc[day.strftime('%Y-%m-%d')]
     idd = day_data['demand'].idxmax()
     maxk.append(day_data.loc[idd]['k'])
ks = pd.Series(maxk, index=demand_max.index, name='kmax')

k33 = df[df['k'] == 33].resample('D', axis=0).sum()
k34 = df[df['k'] == 34].resample('D', axis=0).sum()
matrix = pd.concat([mean_temp2, demand_mean, demand_max, demand_min, k33['demand'], k34['demand'], k33['temp2'],k34['temp2'],ks.astype(int) ], axis=1)
matrix.columns=['temp2', 'demand_mean', 'demand_max', 'demand_min', 'k33', 'k34', 'tempk33', 'tempk34', 'maxk']
print(matrix)

if args.plot:
    ax = matrix['temp2'].plot(label='pv_ghi', color='red')
    plt.ylabel('Mean Temperature', fontsize=15, color='red')
    ax2 = ax.twinx()
    ax2.set_ylabel('Temperature (Degres C)', fontsize=15)
    matrix['demand_mean'].plot(label='demand_mean', color='blue')
    matrix['demand_min'].plot(label='demand_min', color='orange')
    matrix['demand_max'].plot(label='demand_max', color='green')
    matrix['k33'].plot(label='k33', color='yellow')
    matrix['k34'].plot(label='k34', color='purple')
    plt.title('Daily Demand and Temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    fewdays = matrix['2018-06-01 00:00:00' : '2018-06-04 23:30:00']
    ax = fewdays['temp2'].plot(label='pv_ghi', color='red')
    plt.ylabel('Mean Temperature', fontsize=15, color='red')
    ax2 = ax.twinx()
    ax2.set_ylabel('Temperature (Degres C)', fontsize=15)
    fewdays['demand_mean'].plot(label='demand_mean', color='blue')
    fewdays['demand_min'].plot(label='demand_min', color='orange')
    fewdays['demand_max'].plot(label='demand_max', color='green')
    fewdays['k33'].plot(label='k33', color='yellow')
    fewdays['k34'].plot(label='k34', color='purple')
    plt.title('Daily Demand and Temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

print(matrix)

# calcuulate correlation matrix
corrMatrix = matrix.corr()
print (corrMatrix)

# plot
sn.heatmap(corrMatrix, annot=True)
plt.show()

# output data
output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}daily_values_{}.csv'.format(output_dir, dataset)

matrix.to_csv(output_filename, float_format='%.2f')
