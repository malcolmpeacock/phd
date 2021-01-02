# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
from netCDF4 import Dataset, num2date

# Custom scripts
import scripts.download as download 
import scripts.read as read
import scripts.preprocess as preprocess
import scripts.demand as demand
import scripts.cop as cop
import scripts.write as write
import scripts.metadata as metadata
import scripts.plot as plots
import scripts.electric as electric
from scripts.misc import localize

home_path = os.path.realpath('../')

temp_path = os.path.join(home_path, 'temp')
# interim_path = os.path.join(home_path, 'interim')
# output_path = os.path.join(home_path, 'output', version)

# output_file = os.path.join(output_path, 'heatCopRef' + str(ref) + 'weather' + str(year) + methods[method].replace(" ", "") + '.csv')

os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
os.environ["ECMWF_API_KEY"] = "45a818c5dfcb79f44a3a5dd3217a2866"
os.environ["ECMWF_API_EMAIL"] = "MPeacock2@uclan.ac.uk"

year = '2018'

# weather
file = os.path.join(temp_path, 'temperatureSD' + year)

if not os.path.isfile(file):

    # Call the general weather download function with temperature specific parameters
    # oper = HRES sub daily
    download.weather_sd(date="{}-01-01/to/{}-12-31".format(year, year),
                param="167.128/236.128",
                stream="oper",
                target=file)

else:
    print('{} already exists. Download is skipped.'.format(file))

# Read the netCDF file
nc = Dataset(file)
time = nc.variables['time'][:]
time_units = nc.variables['time'].units
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]
variable = nc.variables[variable_name][:]
times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
# Transform to pd.DataFrame
df = pd.DataFrame(data=variable.reshape(len(time), len(latitude) * len(longitude)), index=pd.DatetimeIndex(times, name='time'), columns=pd.MultiIndex.from_product([latitude, longitude], names=('latitude', 'longitude')))
print(df)
