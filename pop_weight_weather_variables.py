# read netcdf of weather parameters from ERA 5 and weight by population.

import os
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset, num2date
from shapely.geometry import Point
import numpy as np
from sklearn.linear_model import LassoCV

# heat stuff
import heat.scripts.preprocess as preprocess

# ERA5 file name
#year = '2018'
#year = '2009'
year = '2019'
filename = '/home/malcolm/uclan/tools/python/scripts/heat/input/weather/ERA5_parms{}.nc'.format(year)

# Read the netCDF file
print('Reading netcdf file ...')
nc = Dataset(filename)
print(nc.variables)
time = nc.variables['time'][:]
time_units = nc.variables['time'].units
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]
u10 = nc.variables['u10'][:]
v10 = nc.variables['v10'][:]
wind = np.sqrt(np.square(u10) + np.square(v10))

variables = {}
variables['wind'] = wind
variables['temp'] = nc.variables['t2m'][:]
variables['temp_dp'] = nc.variables['d2m'][:]
variables['total_precip'] = nc.variables['tp'][:]
variables['surface_pressure'] = nc.variables['sp'][:]
variables['total_cloud_cover'] = nc.variables['tcc'][:]
variables['ghi'] = nc.variables['ssrd'][:]
variables['thermal'] = nc.variables['strd'][:]
variables['clear_sky'] = nc.variables['ssrdc'][:]
times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

print('Getting Mapped population ... ')

heat_path = "/home/malcolm/uclan/tools/python/scripts/heat/"
input_path = heat_path + 'input'
interim_path = heat_path + 'interim'
mapped_population = preprocess.map_population(input_path, interim_path, 'GB', False, year, 'I', False)
print(mapped_population)
total_population = mapped_population.sum()

# Transform to pd.DataFrame
print('Creating DataFrame ...')
wdata={}
for key, var in variables.items():
    print('Weighting {} by population ...'.format(key))

    df = pd.DataFrame(data=var.reshape(len(time), len(latitude) * len(longitude)), index=pd.DatetimeIndex(times, name='time'), columns=pd.MultiIndex.from_product([latitude, longitude], names=('latitude', 'longitude')))
    print(df)
    location_series=[]
    for location in mapped_population.index.tolist():
        # if this location is part of GB then its in mapped_population so we
        # want it, otherwise drop it
        if location in mapped_population.index:
            print('Location found {} {} {}'.format(key, location[0], location[1]) )
            df_wo = df[(location[0], location[1])]
#           print(df_wo)
            pop =  mapped_population[location]
#           print(pop)
            location_series.append(df_wo * pop / total_population)
        else:
            print('Location zero {} {} {}'.format(location[0], location[1]) )
    dfl = pd.concat(location_series, axis=1)
#   print(dfl)
    wdata[key] = dfl.sum(axis=1)
#   print(wdata[key])

dft = pd.DataFrame(data=wdata)
print(dft)

#dft.to_pickle('dft.pickle')

output_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(year)
dft.to_csv(output_file, sep=',', decimal='.', float_format='%g')
