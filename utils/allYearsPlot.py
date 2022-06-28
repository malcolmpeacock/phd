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
parser.add_argument('--bogdan', action="store_true", dest="bogdan", help='Do each year in a different colour', default=False)
parser.add_argument('--existing', action="store_true", dest="existing", help='Add existing heat to scatter plotr', default=False)
args = parser.parse_args()

output_dir = "/home/malcolm/uclan/output"

# load the demands
demands = {}
existings = {}
folder = 'heatpaper'

# load demand with heatpumps
filename = 'GNS'
path = '{}/{}/demand{}.csv'.format(output_dir, folder, filename)
electric_demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
electric_demand.index = pd.DatetimeIndex(pd.to_datetime(electric_demand.index).date)
normalise_factor = 728400.086232
demand = electric_demand * normalise_factor
# load existing heat
filename = 'ENS'
path = '{}/{}/demand{}.csv'.format(output_dir, folder, filename)
electric_existing = pd.read_csv(path, header=0, index_col=0, squeeze=True)
electric_existing.index = pd.DatetimeIndex(pd.to_datetime(electric_existing.index).date)
#print(electric_existing)
existing = electric_existing * normalise_factor

reference_year = demand[str(2018) + '-01-01' : str(2018) + '-12-31']
#print(reference_year)
years = pd.Series(demand.index.year).unique()
feb28 = 31 + 28
# for each weather year ...
for year in years:
    demand_year = demand[str(year) + '-01-01' : str(year) + '-12-31']
    existing_year = existing[str(year) + '-01-01' : str(year) + '-12-31']
    #print('Year {}'.format(year) )
    if len(demand_year)==366:
        values = np.delete(demand_year.values, feb28)
        evalues = np.delete(existing_year.values, feb28)
    else:
        values = demand_year.values
        evalues = existing_year.values
    demands[year] = values * 1e-6
    existings[year] = evalues* 1e-6

#new_index = reference_year.index
new_index = pd.Index(reference_year.index.dayofyear)
#print(new_index)
df = pd.DataFrame(data=demands, index=new_index)
#print(df)
edf = pd.DataFrame(data=existings, index=new_index)
#print(edf)

# median year
yearly = electric_demand.resample('Y').mean()
yearly.sort_values(ascending=False, inplace=True)
mean_year = yearly.index[19].year

for year in years:

    df[year].plot(color='blue')

if args.median:
    extra_line = df[mean_year]
else:
    extra_line = df.mean(axis=1)

extra_line.plot(color='red')

plt.title('Daily variation of electricity demand with 40 years weather')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Demand (TWh / day)', fontsize=15)
plt.show()

# scatter
s_size = 9
for year in years:
    ax1 = plt.scatter(x=df[year].index, y=df[year], color='green', s=s_size, marker='+', linewidths=1)
    if args.existing:
#       plt.scatter(x=edf[year].index, y=edf[year], color='blue', s=s_size)
        ax2 = plt.scatter(x=edf[year].index, y=edf[year], edgecolors='blue', s=9, facecolors='none', marker='^')
        plt.legend( (ax1, ax2), ('41% Electric Heat Pumps', 'Existing Heating'), loc='upper center', fontsize=15)

#TODO legend manually as can't do for each plot
#if args.existing:
#    plt.legend(loc='upper right', fontsize=15)
plt.title('Daily variation of electricity demand with 40 years weather')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Demand (TWh / day)', fontsize=15)
plt.show()

# as 2 plots
plt.subplot(2, 1, 1)
for year in years:
    ax = plt.scatter(x=df[year].index, y=df[year], color='green', s=s_size)
    plt.ylim(0.4, 1.5)
plt.title('Daily variation of electricity demand with 40 years weather')
plt.ylabel('Demand (TWh / day)', fontsize=12)
plt.subplot(2, 1, 2)
for year in years:
    plt.scatter(x=edf[year].index, y=edf[year], color='blue', s=s_size)
    plt.ylim(0.4, 1.5)
plt.xlabel('Day of the year', fontsize=12)
plt.ylabel('Demand (TWh / day)', fontsize=12)
plt.show()


# monthly

demand_monthly = electric_demand.resample('M').mean()
group = demand_monthly.groupby(by=[demand_monthly.index.month]).sum()

# read historical electricity demand for reference year

scotland_factor = 1.1    # ( Fragaki et. al )
demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_ref = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor * 1e-6
daily_electric_ref = electric_ref.resample('D').sum()

#electric_ref_monthly = electric_ref.resample('M').mean()
electric_ref_group = daily_electric_ref.groupby(by=[daily_electric_ref.index.month]).sum()

print('Month  Historic  New    Percent   ')
print('       Total     Total  Increase  ')
for m in range(12):
    month = m+1
    print('{:02d}     {:.2f}     {:.2f}     {:.2f}     '.format(month, electric_ref_group[month], group[month], (group[month] - electric_ref_group[month])/group[month] ) )
