# python script to use the rhpp heat pump data to and the COP time series 
# from heat_series to generate an electricity demand time series to compare
# with the real one.

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob
import os.path
import numpy as np

# custom code
import stats
import readers

# main program

heat_dir = "/home/malcolm/uclan/tools/python/scripts/heat/output/"

years = ['2012', '2013', '2014', '2015' ]
method = 'W'

years_dfs = []
# for each year ...
for year in years:
    # read in the heat time series
    filename = "{}{}/GBRef2018Weather{}I-{}bdew.csv".format(heat_dir, year, year, method)
    heat = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True, usecols=['time','ASHP_floor','ASHP_radiator','GSHP_floor','GSHP_radiator', 'temperature', 'soiltemp'] )
    heat.index = pd.DatetimeIndex(pd.to_datetime(heat.index.strftime("%Y-%m-%d %H:%M") ) )
    years_dfs.append(heat)

# combine the years into one df
df = pd.concat(years_dfs, axis=0)

# add in colunms of zeros to contain the synthetic electricity time series
df['synthetic'] = 0
# add in column of zeros to contain the actual electricity in time series
df['real'] = 0
# add in column of zeros to contain the actual heat demand to do cop later
df['heat_GSHP'] = 0
df['elec_GSHP'] = 0
df['heat_ASHP'] = 0
df['elec_ASHP'] = 0
print(df)

# read the sources file which tells use ASHP/GSHP etc
rhpp_sources_file = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/mrdoc/excel/sourcesink.csv'
sources = pd.read_csv(rhpp_sources_file, index_col=0)
print(sources)

# read sites csv to determine locations in the B cropped sample
# ( these have less data errors )
rhpp_sites_file = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/mrdoc/excel/8151_site_list.csv'
sites = pd.read_csv(rhpp_sites_file, index_col=0, usecols=[0,5], squeeze=True)
sites = sites.rename('B2crop')
print(sites)

sinks = { 'name' : [], 'min' : [], 'max' : [], 'mean' : [], 'type' : [] }

gshp_cops = np.empty(0)
gshp_deltat = np.empty(0)
ashp_cops = np.empty(0)
ashp_deltat = np.empty(0)

# process the houses
rhpp_dir = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/csv/'

# for each house file ....
for name in glob.glob(rhpp_dir + 'processed*'):
    filename = os.path.basename(name)
    location = filename[14:18]
    print('Location: {}'.format(location) )
    # skip if its not part of the B sample cropped
    if sites['RHPP' + location] == 0:
        print('Skipping {}, not in B2'.format(location) )
        continue

    # read heat demand and temp
    house, stats = readers.read_rhpp(name, location)
#   house = house.resample('D').sum()
#   house.index = pd.DatetimeIndex(house.index.date)
#   print(house)
    # If no nans in heat and electricity
    if stats['nans_Hhp'] == 0 and stats['nans_Ehp'] == 0:
        # get heat pump type
        loc_key = 'RHPP' + location
        source = sources.loc[loc_key]
#       print(source)
        hp = source['Heat.pump.type']
        sink = source['Emitter.type']
#       house.loc[house.pump_heat_already_in_Hhp==0.0, 'Hhp'] = house.loc[house.pump_heat_already_in_Hhp==0.0, 'Hhp'] + house.loc[house.pump_heat_already_in_Hhp==0.0, 'Modelled_Hcircpump']
        house_heat = house['Hhp']
#       print(house_heat)
        house_electric = house['Ehp']
#       print(house_electric)
        print('Ehp Zero {}'.format(len(house[house['Ehp'] < 0.0000001])))
        # get the cops and deltats
        house_cop = house['Hhp'] / house['Ehp']
#       house_cop = house_cop[~house_cop.isin([np.nan, np.inf, -np.inf]).any()]
#       house_cop = house_cop.dropna()
#       n_gt10 = len(house_cop[house_cop > 10.0])
#       if n_gt10 > 10:
            #print('WARNING COP > 10 {} for {}'.format(n_gt10, location))
            # can't drop the cops > 10 or the deltats will be different
            # length so do it later.
#       print(house_cop.values)
#       print(df.loc[house.index])
        if hp == 'GSHP':
            gshp_cops = np.concatenate([gshp_cops, house_cop.replace(np.inf, 0).values])
            house_deltat = house['Tsf'] - df.loc[house.index]['soiltemp']
            gshp_deltat = np.concatenate([gshp_deltat, house_deltat.values])
        else:
            ashp_cops = np.concatenate([ashp_cops, house_cop.replace(np.inf, 0).values])
            house_deltat = house['Tsf'] - df.loc[house.index]['temperature']
            ashp_deltat = np.concatenate([ashp_deltat, house_deltat.values])
#       print(house_deltat)
#       print(all_deltat)
       
        # expand the house time series to the whole period
        # and create a time series of 1's and zeros for which days
        # the house was monitored.
        house_electric = house_electric.reindex(df.index,fill_value = 0.0)
        house_heat = house_heat.reindex(df.index,fill_value = 0.0)
#       print('house_heat')
#       print(house_heat)
#       print(house_heat.loc['2014-03-02'])
#       print('house_electric')
#       print(house_electric)
#       print(house_electric.loc['2014-03-02'])
        # sink temp
        sinks['name'].append(location)
        sinks['max'].append(house['Tsf'].max())
        sinks['min'].append(house['Tsf'].min())
        sinks['mean'].append(house['Tsf'].mean())
        # add in real electric time series
        df['real'] = df['real'] + house_electric
        # calculate synthetic series = heat / cop
        print('HP {} sink {}'.format(hp, sink) )
        sinks['type'].append(sink)
        # if both underfloor and radiators then assume 50-50
        if sink == 'Both':
            print('BOTH')
            synthetic = (house_heat / ( df[hp+'_floor'] * 2 ) ) + (house_heat / ( df[hp+'_radiator'] * 2 ) )
        else:
            if sink == 'Radiators':
                print('Radiators')
                synthetic = house_heat / ( df[hp+'_radiator'] )
            else:
                print('Floor')
                synthetic = house_heat / ( df[hp+'_floor'] )
        # add in synthetic electric time series
        df['synthetic'] = df['synthetic'] + synthetic
        # store by GSHP / ASHP
        df['elec_'+hp] = df['elec_'+hp] + house_electric
        df['heat_'+hp] = df['heat_'+hp] + house_heat

print(df)

output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
df.to_csv(output_dir + 'electric.csv')

df = pd.DataFrame(data=sinks)
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
df.to_csv(output_dir + 'sinks.csv')

# gshp cop
data = { 'cop' : gshp_cops, 'deltat' : gshp_deltat }
df = pd.DataFrame(data=data)
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
df.to_csv(output_dir + 'gshp.csv')

# ashp cop
data = { 'cop' : ashp_cops, 'deltat' : ashp_deltat }
df = pd.DataFrame(data=data)
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
df.to_csv(output_dir + 'ashp.csv')
