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

# Rad the ERA5 windspeeds
def read_era5(filename):
    nc = Dataset(filename)
    time = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    latitude = nc.variables['latitude'][:]
    longitude = nc.variables['longitude'][:]
    u10 = nc.variables['u10'][:]
#   print('U10')
#   print(u10)
    v10 = nc.variables['v10'][:]
    # calculate wind speed from u and v components
    wind10 = np.sqrt(np.square(u10) + np.square(v10))
    u100 = nc.variables['u100'][:]
    v100 = nc.variables['v100'][:]
#   print('U100')
#   print(u100)
    wind100 = np.sqrt(np.square(u100) + np.square(v100))
    times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    wind10 = wind10.reshape(len(time), len(latitude) * len(longitude))
    wind100 = wind100.reshape(len(time), len(latitude) * len(longitude))
#   data = { 'wind10' : wind10, 'wind100' : wind100 }
    # Transform to pd.DataFrame
    df10 = pd.DataFrame(data=wind10,
                      index=pd.DatetimeIndex(times, name='time'),
                      columns=pd.MultiIndex.from_product([latitude, longitude],
                                                         names=('latitude', 'longitude')))
#   print('df10')
#   print(df10)
    tower10 = df10[(52.5,4.5)]
#   print('tower10')
#   print(tower10)
    df100 = pd.DataFrame(data=wind100,
                      index=pd.DatetimeIndex(times, name='time'),
                      columns=pd.MultiIndex.from_product([latitude, longitude],
                                                         names=('latitude', 'longitude')))
    tower100 = df100[(52.5,4.5)]
    df = pd.concat([tower10, tower100], axis=1,keys=['wind10', 'wind100'])

    return df

# Read the tall tower netCDF file
def read_netcdf(directory,name,yearmonth):

    filename = os.path.join(directory, name, name + '_' + yearmonth + '.nc')
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
    height = nc.variables['height']
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
    df = df.interpolate()
    # incase the first or last value was NaN use back forward fill
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    # resample to hourly
    df = df.resample('H').mean()
    print('TOWER')
    print(df)
    return df

# power law using wind speed from 2 heights to work out alpha first
# df    dataframe 
# h1    height of wind speeds in v1
# h2    height of wind speeds in v2
# href  height of wind speeds in vref
# h     height you want the wind speed at 
# v     column to populate in df

def power_law_given_2heights(df,h1,h2,v1,v2,href,vref,h,v):
    # calculate the power law exponent alpha from h1,h2,v1,v2
    base = h2 / h1
    vd = v2 / v1
    # log vd to base = ln(vd) / ln(base)
    alpha = np.log(vd) / np.log(base)
#   alpha = np.exp(log_alpha)
    print('ALPHA')
    print(alpha)
    # create an array of the same length filled with h/href
    n = np.empty(len(df))
    n.fill(h / href )
    # apply the power law using the alpha calculated above to 
    # get the speed v from vref 
    df[v] = vref * np.power(n, alpha)
    return df

# power law given alpha
#   df    - dataframe
#   alpha - power law exponent
#   h1    - height for wind speeds in v1
#   h2    - height where you want the wind speed
#   v2    - where to put the new windspeed in df

def power_law_given_alpha(df, alpha, h1, h2, v1, v2):
    df[v2] = v1 * ( (h1 / h2 ) ** alpha )
    return df

# log law 
#   df    - dataframe
#   href  - height for wind speeds in vref
#   h     - height where you want the wind speed
#   v     - where to put the new windspeed in df

def log_law(df, z0, href, vref, h, v):
    df[v] = (math.log(h/z0) / math.log(href/z0) ) * vref
    return df

# Read ERA5 wind
erafile="/home/malcolm/uclan/data/era5/highwind_2010_12.nc"
era5 = read_era5(erafile)
print('ERA5')
print(era5)

# Read the tall tower data

dir1 = "/home/malcolm/uclan/data/talltowers/owez/10minutely"

df1 = read_netcdf(dir1,'windagl21S1','201012')
print(df1)
df2 = read_netcdf(dir1,'windagl70S1','201012')
print(df2)
df3 = read_netcdf(dir1,'windagl116S1','201012')
print(df3)
tower = pd.concat([df1, df2, df3], axis=1,keys=['wind21', 'wind70', 'wind116' ])
print('Tower')
print(tower)

# defaults
# surface roughness for ocean
z0 = 0.03
alpha = 0.14

# tower data - log law for 70m from 21m
tower = log_law(tower, z0, 21, tower['wind21'], 70, 'log70')

# tower data - log law for 116m from 21m
tower = log_law(tower, z0, 21, tower['wind21'], 116, 'log116')

# tower data - power law for 70m from 21m
tower = power_law_given_alpha(tower, alpha, 21, 70, tower['wind21'], 'power70')
# tower data - power law for 116m from 21m
tower = power_law_given_alpha(tower, alpha, 21, 116, tower['wind21'], 'power116')
print(tower)

# era5 data  - log law for 70m from 10m
era5 = log_law(era5, z0, 10, era5['wind10'], 70, 'log70')
# era5 data  - log law for 116 from 10m
era5 = log_law(era5, z0, 10, era5['wind10'], 116, 'log116')

# era5 data  - power law using alpha from 2 heights then 70m from 10m
era5 = power_law_given_2heights(era5,10,100,era5['wind10'],era5['wind100'],10,era5['wind10'],70,'power70')
# era5 data  - power law using alpha from 2 heights then 116 from 100m
era5 = power_law_given_2heights(era5,10,100,era5['wind10'],era5['wind100'],100,era5['wind100'],116,'power116')
print(era5)

# output wind plots for 70m

tower['wind70'].plot(label='Tower Wind Speed at 70m')
tower['log70'].plot(label='Log law from 21m to 70m')
tower['power70'].plot(label='Power law from 21m to 70m')
era5['log70'].plot(label='ERA5 Log law from 10m to 70m')
era5['power70'].plot(label='ERA5 Power law using 2 heights 70m')
plt.title('Amsterdam Offshore Tower windspeed for December 2010')
plt.xlabel('Day of the month')
plt.ylabel('Wind Speed (M/S)')
plt.legend(loc='upper right')
plt.show()

# output stats for 70m

stats.print_stats_header()
stats.print_stats(tower['log70'], tower['wind70'], 'Log law 70m', 1, True, 'Log law wind speed (m/s)', 'Actual wind speed (m/s)' )
stats.print_stats(tower['power70'], tower['wind70'], 'Power law 70m', 1, True, 'Power law wind speed (m/s)', 'Actual wind speed (m/s)' )
stats.print_stats(era5['log70'], tower['wind70'], 'ERA5 log law', 1, True, 'Power law wind speed (m/s)', 'Actual wind speed (m/s)' )
stats.print_stats(era5['power70'], tower['wind70'], 'ERA5 Power law', 1, True, 'Power law wind speed (m/s)', 'Actual wind speed (m/s)' )

# output wind plots for 116m

tower['wind116'].plot(label='Tower Wind Speed at 116m')
tower['log116'].plot(label='Log law from 21m to 116m')
tower['power116'].plot(label='Power law from 21m to 116m')
era5['log116'].plot(label='ERA5 Log law from 10m to 116m')
era5['power116'].plot(label='ERA5 Power law using 2 heights 116m')
plt.title('Amsterdam Offshore Tower windspeed for December 2010')
plt.xlabel('Day of the month')
plt.ylabel('Wind Speed (M/S)')
plt.legend(loc='upper right')
plt.show()

# output stats for 116m

stats.print_stats_header()
stats.print_stats(tower['log116'], tower['wind116'], 'Log law 116m', 1, True, 'Log law wind speed (m/s)', 'Actual wind speed (m/s)' )
stats.print_stats(tower['power116'], tower['wind116'], 'Power law 116m', 1, True, 'Power law wind speed (m/s)', 'Actual wind speed (m/s)' )
stats.print_stats(era5['log116'], tower['wind116'], 'ERA5 log law', 1, True, 'Power law wind speed (m/s)', 'Actual wind speed (m/s)' )
stats.print_stats(era5['power116'], tower['wind116'], 'ERA5 Power law', 1, True, 'Power law wind speed (m/s)', 'Actual wind speed (m/s)' )
