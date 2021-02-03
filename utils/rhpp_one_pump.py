# python script to create a COP curve from the data for one heat pump.    

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
# 5299 COP curve is wrong way!
location = '5299'
# 5311 COP curver is OK
#location = '5311'

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

print(df)
rhpp_sources_file = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/mrdoc/excel/sourcesink.csv'
sources = pd.read_csv(rhpp_sources_file, index_col=0)
print(sources)

rhpp_dir = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/csv/'

# house file
name = "{}processed_rhpp{}.csv".format(rhpp_dir, location)
print('Location: {}'.format(location) )

# read heat demand and temp
house, stats = readers.read_rhpp(name, location)
print(stats)

# join the df to match the house index
house = house.join(df, how='left')
# only include the points with some electricity
house = house[ house['Ehp'] > 0.0]
# only include the points with some heat
house = house[ house['Hhp'] > 0.0]

print(house) 

ax = house['temperature'].plot(label='Reanalysis temperature', color='green')
plt.ylabel('Temperature (degrees C)', fontsize=15)
ax2 = ax.twinx()
ax2.set_ylabel('Demand (kWh)',color='red', fontsize=15)
house['Hhp'].plot(label='Measured heat demand', color='blue')
house['Ehp'].plot(label='Measured electicity demand', color='red')
plt.title('Measured heat demand for one house')
plt.xlabel('Hour of the year', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# Get sink temperature
loc_key = 'RHPP' + location
source = sources.loc[loc_key]
hp = source['Heat.pump.type']
sink = source['Emitter.type']
print('HP {} sink {}'.format(hp, sink) )
# TODO should use soiltemp for gshp, first example is ASHP
# if both underfloor and radiators then assume 50-50
if sink == 'Both':
    sink_temp = 35.0
    print('BOTH')
else:
    if sink == 'Radiators':
        sink_temp = 40.0
        print('Radiators')
    else:
        sink_temp = 30.0
        print('Floor')

# delta T
house['deltat'] = (house['temperature'] * -1.0) + sink_temp
# calculate COP
house['cop'] = house['Hhp'] / house['Ehp']
#house = house.sort_values('deltat', axis=0)
print(house)

# fit a regression line through the cops
rmodel = sm.OLS(house['cop'].to_numpy(), sm.add_constant(house['deltat'].to_numpy()))
residual_results = rmodel.fit()
res_const = residual_results.params[0]
res_grad = residual_results.params[1]
x = np.array([house['deltat'].min(),house['deltat'].max()])
y = res_const + res_grad * x

plt.scatter(house['deltat'], house['cop'], s=12)
plt.plot(x, y, color='red')
plt.title('RHPP location {} COP vs DELTA T'.format(location))
plt.xlabel('Temperature Difference (degrees C)')
plt.ylabel('COP')
plt.show()

