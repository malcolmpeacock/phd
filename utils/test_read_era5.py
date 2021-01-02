import os
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset, num2date
from shapely.geometry import Point

#filename = 'ERA5-2018-Hourly-Temp.nc'
filename = '/home/malcolm/uclan/tools/python/input/weather/ERAI_wind.nc'
#variable_name = '2m_temperature'
variable_name = 't2m'
# Read the netCDF file
nc = Dataset(filename)
print(nc.variables)
time = nc.variables['time'][:]
time_units = nc.variables['time'].units
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]
variable = nc.variables[variable_name][:]
times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
# Transform to pd.DataFrame
df = pd.DataFrame(data=variable.reshape(len(time), len(latitude) * len(longitude)),
                  index=pd.DatetimeIndex(times, name='time'),
                  columns=pd.MultiIndex.from_product([latitude, longitude],
                                                     names=('latitude', 'longitude')))
print(df)
