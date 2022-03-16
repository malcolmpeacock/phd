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

    nc = Dataset(filename)
    time = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    print(type(times[0]) )
#   print(nc.variables)
    latitude = nc.variables['latitude'][:]
#   print(latitude)
    lat_dim = latitude.shape
#   print(lat_dim)
    longitude = nc.variables['longitude'][:]
    month_number = nc.variables['month_number'][:]
    year = nc.variables['year'][:]
    temp = nc.variables['tas'][:]
    print(' lat {} lon {} month_number {} year {} temp {}'.format(len(latitude), len(longitude), len(month_number), len(year), len(temp) ) )
    # Transform to pd.DataFrame
    df = pd.DataFrame(data=temp.reshape(len(year), lat_dim[0] * lat_dim[1]),
                      index=pd.DatetimeIndex(times, name='time'),
                      columns=pd.MultiIndex.from_arrays([latitude.reshape(lat_dim[0] * lat_dim[1]), longitude.reshape(lat_dim[0] * lat_dim[1])],
                                                         names=('latitude', 'longitude')))

    return df

# process command line
parser = argparse.ArgumentParser(description='List contents of netcdf file')
parser.add_argument('folder', help='folder')
parser.add_argument('file', help='Filename')
parser.add_argument('--kfgen', action="store_true", dest="kfgen", help='Use KF generation from matlab', default=False)

args = parser.parse_args()

filename = "/home/malcolm/uclan/data/{}/{}".format(args.folder, args.file)

df = read_netcdf(filename)
print(df)

#df1980 = df.loc[
