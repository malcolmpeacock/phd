# python script to model the rhpp heat pump data using the 4 methods 
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
years = ['2012', '2013', '2014', '2015' ]

years_dfs = []
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
# add in column of zeros to contain the added in time series for the pumps
df['pumps'] = 0
print(df)

rhpp_dir = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/csv/'

# for each house file ....
for name in glob.glob(rhpp_dir + 'processed*'):
    filename = os.path.basename(name)
#   print(filename)
    location = filename[14:18]
    print(location)

    # read heat demand and temp
    house, stats = readers.read_rhpp(name, location)
    house = house.resample('D').sum()
    house.index = pd.DatetimeIndex(house.index.date)
#   print(house)
    # If no nans in heat and electricity
    if stats['nans_Hhp'] == 0 and stats['nans_Ehp'] == 0:
        # Add in the circulation heat if not already there.
        # TODO check this works !!!
        house.loc[house.pump_heat_already_in_Hhp==0.0, 'Hhp'] = house.loc[house.pump_heat_already_in_Hhp==0.0, 'Hhp'] + house.loc[house.pump_heat_already_in_Hhp==0.0, 'Modelled_Hcircpump']
        # expand the house time series to the whole period
        # and create a time series of 1's and zeros for which days
        # the house was monitored.
        house_heat = house['Hhp']
        total_heat = house_heat.sum()
        ones = house_heat * 0.0 + 1.0
        ones = ones.reindex(df.index, fill_value = 0.0)
        ones = ones.rename('ones')
#       print(ones)
#       print(ones.loc['2014-03-02'])
        house_heat = house_heat.reindex(df.index,fill_value = 0.0)
#       print(house_heat)
#       print(house_heat.loc['2014-03-02'])
        # add in calculated synthetic time series for each method
        for method in methods:
            # make this zero when this house is not monitored
            used_factors = df['f_' + method] * ones
            # normalize factor over the monitoring period
            norm_factor = 1 / used_factors.sum()
            df[method] = df[method] + (used_factors * total_heat * norm_factor)
        # add in actual time series
        df['pumps'] = df['pumps'] + house_heat

print(df)

output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
df.to_csv(output_dir + 'methods.csv')
