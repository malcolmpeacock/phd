# Calculat a temperature by blinear interpolation of the surrounding ERA5
# grid squares .

# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
import math
from scipy import interpolate

# Custom scripts
import heat.scripts.download as download 
import heat.scripts.read as read
import heat.scripts.preprocess as preprocess
import heat.scripts.demand as demand
import heat.scripts.cop as cop
import heat.scripts.write as write

import utils.bilinear as bil

# inputs
location_latitude = 53.7746
location_longitude = -3.03647
location = 'BlackpoolSquiresGate'

home_path = os.path.realpath('heat')
#home_path = os.path.realpath('../')

input_path = os.path.join(home_path, 'input')
interim_path = os.path.join(home_path, 'interim')
output_path = os.path.join(home_path, 'output','loctemp', location)

os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
os.environ["ECMWF_API_KEY"] = "45a818c5dfcb79f44a3a5dd3217a2866"
os.environ["ECMWF_API_EMAIL"] = "MPeacock2@uclan.ac.uk"

year = 2018

def distance(x1,y1,x2,y2):
    d = math.sqrt( ( x1-x2 )**2 + ( y1-y2)**2 )
    return d

# weather
download.temperatures(input_path, year, year)

# get air temperature at 6 hourly intvervals
t = read.temperature(input_path, year, year, 't2m')

# take the mean of the 4 temperatures to get daily
t = t.resample('D').mean()

# convert to degrees C
t = t - 273.15

print(t.head())

df, x, y = bil.bilinear(location_latitude, location_longitude, t)

plt.scatter(x,y)
plt.text(x[0], y[0], 'G(x1,y1)', fontsize=12)
plt.text(x[1], y[1], 'G(x2,y2)', fontsize=12)
plt.text(x[2], y[2], 'G(x2,y1)', fontsize=12)
plt.text(x[3], y[3], 'G(x1,y2)', fontsize=12)
plt.text(x[4], y[4], 'Blackpool(xp,yp)', fontsize=12)
plt.title('Location of Blackpool Squires Gate on ERA5 Grid')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()

print(df)

output_file = output_path + str(year) + '.csv'
df.to_csv(output_file, sep=',', decimal='.', float_format='%g')
