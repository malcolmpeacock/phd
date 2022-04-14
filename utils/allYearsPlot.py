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
parser = argparse.ArgumentParser(description='Plot all years of modified demand')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--median', action="store_true", dest="median", help='{plot median year instead of average', default=False)
parser.add_argument('--shift', action="store_true", dest="shift", help='Shift the days to match weekly pattern', default=False)
args = parser.parse_args()

output_dir = "/home/malcolm/uclan/output"

# load the demands
demands = {}
folder = 'heatpaper'
filename = 'GNS'
path = '{}/{}/demand{}.csv'.format(output_dir, folder, filename)
electric_demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
electric_demand.index = pd.DatetimeIndex(pd.to_datetime(electric_demand.index).date)
#print(demand)
normalise_factor = 728400.086232
demand = electric_demand * normalise_factor

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
print(df)

# median year
yearly = electric_demand.resample('Y').mean()
yearly.sort_values(ascending=False, inplace=True)
print('YEARLY')
print(yearly)
mean_year = yearly.index[19].year
print(mean_year)

for year in years:

    df[year].plot(color='blue')

if args.median:
    extra_line = df[mean_year]
    print(extra_line)
else:
    extra_line = df.mean(axis=1)
    print(extra_line)

extra_line.plot(color='red')

plt.title('Daily average and variation of electricity demand with 40 years weather')
plt.xlabel('day', fontsize=15)
plt.ylabel('Demand (TWh / day)', fontsize=15)
plt.show()

# scatter
first = True
for year in years:
    s_size = 5
    if first:
        ax = plt.scatter(x=df[year].index, y=df[year], color='blue', s=s_size)
    else:
        ax = plt.scatter(x=df[year].index, y=df[year], color='blue', ax=ax, s=s_size)
plt.title('Daily average and variation of electricity demand with 40 years weather')
plt.xlabel('day', fontsize=15)
plt.ylabel('Demand (TWh / day)', fontsize=15)
plt.show()

# monthly

demand_monthly = electric_demand.resample('M').mean()
print(demand_monthly)
group = demand_monthly.groupby(by=[demand_monthly.index.month]).sum()
print(group)

# read historical electricity demand for reference year

scotland_factor = 1.1    # ( Fragaki et. al )
demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_ref = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor * 1e-6
daily_electric_ref = electric_ref.resample('D').sum()

#electric_ref_monthly = electric_ref.resample('M').mean()
electric_ref_group = daily_electric_ref.groupby(by=[daily_electric_ref.index.month]).sum()
print(electric_ref_group)

print('Month  Historic  New    Percent   ')
print('       Total     Total  Increase  ')
for m in range(12):
    month = m+1
    print('{:02d}     {:.2f}     {:.2f}     {:.2f}     '.format(month, electric_ref_group[month], group[month], (group[month] - electric_ref_group[month])/group[month] ) )
