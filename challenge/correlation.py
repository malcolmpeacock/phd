# See how the data for the data challange correlates.

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

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(df)

print(df.columns)

if args.plot:
    df['pv_ghi'].plot(label='pv_ghi', color='blue')
    df['cs_ghi'].plot(label='cs_ghi', color='red')
    df['poa_ghi'].plot(label='poa_ghi', color='orange')
    df['sun2'].plot(label='sun1', color='green')
    plt.title('Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    fewdays = df['2018-06-01 00:00:00' : '2018-06-04 23:30:00']
    fewdays['pv_ghi'].plot(label='pv_ghi', color='blue')
    fewdays['cs_ghi'].plot(label='cs_ghi', color='red')
    fewdays['poa_ghi'].plot(label='poa_ghi', color='orange')
    fewdays['sun2'].plot(label='sun1', color='green')
    plt.title('Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    lat =   [ 50.33,   50.5,  50.5,  51.0, 51.5,   50.0, 50.0 ]
    lon =   [-4.034, -4.375, -3.75, -3.75, -2.5, -4.375, -3.75]
    label = ['pv',     'w1',  'w2',  'w3', 'w4',   'w5', 'w6' ]

    fig, ax = plt.subplots()
    ax.scatter(lon, lat)
    for i, txt in enumerate(label):
        ax.annotate(txt, (lon[i], lat[i]))
    plt.show()

# convert to daily??
#matrix = df.resample('D', axis=0).mean()
matrix = df
print(matrix)

# calcuulate correlation matrix
corrMatrix = matrix.corr()
print (corrMatrix)

# plot
sn.heatmap(corrMatrix, annot=True)
plt.show()
