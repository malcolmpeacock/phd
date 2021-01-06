# Python modules
import os
import shutil
import pandas as pd
import numpy as np
import math
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
from netCDF4 import Dataset, num2date

# custom code
import stats

# Read the netCDF file
def read_netcdf(filename):

    print(filename)
    nc = Dataset(filename)
#   print(nc.variables)
    time = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    # latitude = nc.variables['latitude'][:]
    # longitude = nc.variables['longitude'][:]
    latitude = nc.variables['latitude']
#   print(latitude)
#   print(latitude[0])
#   print("latitude {}".format(latitude[:]))
    longitude = nc.variables['longitude']
#   print(longitude)
#   height = nc.variables['height']
#   print(height)
    print("latitude {} longitude {} height {} ".format(latitude[0], longitude[0], height[0]))
    wind_speed = nc.variables[name][:]
    print(type(wind_speed))
    # print(wind_speed)
    status_flag = nc.variables['f_' + name][:]
    # try to set the first and last values if NaN
#   if status_flag[0] == 4:
#   wind_speed[0] = wind_speed[1]
    # print(status_flag)
    times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
#   data = {'windspeed' : wind_speed, 'status': status_flag}
    data = wind_speed
    # Transform to pd.DataFrame
#   df = pd.DataFrame(data=data, index=pd.DatetimeIndex(times, name='time'))
    df = pd.Series(data=data, index=pd.DatetimeIndex(times, name='time'), name=name)
    # interpolate missing values
#   df = df.interpolate()
    # incase the first or last value was NaN use back forward fill
#   df = df.fillna(method='ffill')
#   df = df.fillna(method='bfill')
    # resample to hourly
#   df = df.resample('H').mean()
#   print('TOWER')
#   print(df)
    return df

# the netcdf data

filename = "/home/malcolm/uclan/output/correlation/weather/ERA1I_weather_2018.nc"

df = read_netcdf(filename)
print(df)
