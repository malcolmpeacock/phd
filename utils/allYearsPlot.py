# All year sensitivity plot

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from icecream import ic 

# custom code
import stats
import readers
import storage

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--scenario', action="store", dest="scenario", help='Scenarion to plot', default='adhoc')
args = parser.parse_args()

output_dir = "/home/malcolm/uclan/output"

# load the demands
demands = {}
folder = 'heatpaper'
filename = 'GNS'
path = '{}/{}/demand{}.csv'.format(output_dir, folder, filename)
demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
#print(demand)

reference_year = demand[str(2018) + '-01-01' : str(2018) + '-12-31']
print(reference_year)
years = pd.Series(demand.index.year).unique()
feb28 = 31 + 28
# for each weather year ...
for year in years:
    demand_year = demand[str(year) + '-01-01' : str(year) + '-12-31']
    print('Year {}'.format(year) )
    if len(demand_year)==366:
        print('Leap year')
        values = np.delete(demand_year.values, feb28)
    else:
        print('Normal year')
        values = demand_year.values
    demands[year] = values

df = pd.DataFrame(data=demands, index=reference_year.index)
average = df.mean(axis=1)
print(average)
print(df)
for year in years:

    df[year].plot(color='blue')

average.plot(color='red')

plt.title('Daily average and variation of electricity demand with 40 years weather')
plt.xlabel('day', fontsize=15)
plt.ylabel('Demand (TWh / day)', fontsize=15)
plt.show()

