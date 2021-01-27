# python script to model the advanced metering gas data using the 4 methods 
# for validation

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob
import os.path

# custom code
import stats
import readers

# main program

heat_dir = "/home/malcolm/uclan/tools/python/scripts/heat/output/"

methods = { 'B' : 'BDEW', 'W' : 'Watson', 'S' : 'HDD15.5', 'H' : 'HDD12.8' }
years = ['2004', '2005', '2006']

years_dfs = []

print ('Reading heat series ...')
# for each year ...
for year in years:
    methods_dfs = []
    # for each method ...
    for method in methods:
        # read in the heat time series
        filename = "{}{}/GBRef2018Weather{}I-{}bdew.csv".format(heat_dir, year, year, method)
        heat = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True, usecols=['time','space'] )
        # convert to daily
        heat = heat.resample('D').sum()
        heat = heat.rename('f_' + method)
        heat.index = pd.DatetimeIndex(heat.index.date)
        methods_dfs.append(heat)
    # combine the methods into one df
    combined = pd.concat(methods_dfs, axis=1)
    years_dfs.append(combined)

# combine the years into one df
df = pd.concat(years_dfs, axis=0)

# add in colunms of zeros to contain the added in time series for each method
for method in methods:
    df[method] = 0
# add in column of zeros to contain the added in time series for the gas
df['gas'] = 0
print(df)

print ('Reading building files ...')
advm_dir = '/home/malcolm/uclan/data/advanced_metering/'

# for each building file ....
for name in glob.glob(advm_dir + '*.csv'):
    filename = os.path.basename(name)
#   print(filename)
    location = filename[0:9]
    print(location)

    # read heat demand and temp
    house, stats = readers.read_advm(name, location)
#   print(house)
    house = house.resample('D').sum()
#   house.index = pd.DatetimeIndex(house.index.date)
#   print(house)
    print('{} nans {} fixed {}'.format(location,stats['nans'],stats['fixed']) )

    # expand the house time series to the whole period
    # and create a time series of 1's and zeros for which days
    # the house was monitored.
    house_heat = house['value']
    total_heat = house_heat.sum()
    ones = house_heat * 0.0 + 1.0
    ones = ones.reindex(df.index, fill_value = 0.0)
    ones = ones.rename('ones')
#   print(ones)
#   print(ones.loc['2004-06-01'])
    house_heat = house_heat.reindex(df.index,fill_value = 0.0)
#   print(house_heat)
#   print(house_heat.loc['2004-06-01'])
    # add in calculated synthetic time series for each method
    for method in methods:
        # make this zero when this house is not monitored
        used_factors = df['f_' + method] * ones
        # normalize factor over the monitoring period
        norm_factor = 1 / used_factors.sum()
        df[method] = df[method] + (used_factors * total_heat * norm_factor)
    # add in actual time series
    df['gas'] = df['gas'] + house_heat

print(df)

output_dir = "/home/malcolm/uclan/data/advanced_metering/testing/"
df.to_csv(output_dir + 'methods.csv')
