# program to verify the affect of waves on surface roughness.
# since I've not found and data with windspeed at different heights
# and wave data at the same time its not done yet!

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
#from netCDF4 import Dataset, num2date

# custom code
import stats
import readers

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

# Read the crown estates data

gabbard_filename = "/home/malcolm/uclan/data/crown_estates/greatergabbard/1478/DataSets/6841/MetMast_Clean_GG_Data_2009.csv"

measured = readers.read_crown(gabbard_filename)
print(measured)
print(measured.columns)

# Midas marine
midas_filename = '/home/malcolm/uclan/output/wind/marine_lat51.1_long1.8_2009.csv'
midas = pd.read_csv(midas_filename, header=0, parse_dates=[0], index_col=0 )
midas.index =midas.index.map(lambda t: t.strftime('%Y-%m-%d %H:%M:00'))
midas.index = pd.DatetimeIndex(midas.index)
print(midas)
print(midas.columns)
quit()

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
