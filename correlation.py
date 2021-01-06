# Get lots of variables from the 2018 weather and see how they correlate
# with the 2018 gas demand.

# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import seaborn as sn

# Custom scripts
import heat.scripts.download as download 
import heat.scripts.read as read
from utils import readers
#import scripts.preprocess as preprocess
#import scripts.demand as demand
#import scripts.cop as cop
#import scripts.write as write
#import scripts.metadata as metadata

population_path = "/home/malcolm/uclan/tools/python/scripts/heat/input/populaiton"
output_path = "/home/malcolm/uclan/output/correlation"

year = 2018
hour = 1
grid = 'I'

# get weather, if not already present
download.weather_era5(output_path, year, hour, grid, 'weather', [ '2m_temperature', 'soil_temperature_level_4', '10m_u_component_of_wind', '10m_v_component_of_wind', 'surface_pressure', 'total_cloud_cover', 'instantaneous_large_scale_surface_precipitation_fraction', 'large_scale_rain_rate', 'precipitation_type', 'surface_solar_radiation_downwards', 'surface_solar_radiation_downward_clear_sky' ], [ 60, -8, 48, 2, ])

# read weather
parameters = {
            'air_temp': 't2m',
            'soil_temp': 'stl4',
            'wind_u': 'u10',
            'wind_v': 'v10',
            'pressure': 'sp',
            'cloud': 'tcc',
            'precipitation': 'ilspf',
            'rain': 'lsrr',
#           'precipitation_type': 'ptype',
            'irradiance': 'ssrd',
            'irradiance_cs': 'ssrdc'
        }
weather = pd.concat(
            [read.weather_era5(output_path, year, hour, grid, 'weather', parameter) for parameter in parameters.values()],
            keys=parameters.keys(), names=['parameter', 'latitude', 'longitude'], axis=1
        )
print(weather)
weather.to_pickle("/home/malcolm/uclan/output/correlation/weather/test.pickle")

# map population to the weather grid

population_filename = os.path.join(output_path, "population")
if os.path.isfile(population_filename):
    s = pd.read_pickle(population_filename)
    print("mapped population read from file")
else:
    population = read.population("/home/malcolm/uclan/tools/python/scripts/heat/input")
    weather_grid = None
    mapped_population = {}
    weather_data = weather['air_temp']

    # Make GeoDataFrame from the weather data coordinates
    weather_grid = gpd.GeoDataFrame(index=weather_data.columns)
    weather_grid['geometry'] = weather_grid.index.map(lambda i: Point(reversed(i)))

    # Set coordinate reference system to 'latitude/longitude'
    weather_grid.crs = {'init': 'epsg:4326'}

    # Make polygons around the weather points
    weather_grid['geometry'] = weather_grid.geometry.apply(lambda point: point.buffer(.75 / 2, cap_style=3))

    # Make list from MultiIndex (this is necessary for the spatial join)
    weather_grid.index = weather_grid.index.tolist()

    # Filter population data by country to cut processing time
    gdf = population[population['CNTR_CODE'] == 'UK'].copy()

    # Align coordinate reference systems
    print(' aligning coords .....')
    gdf = gdf.to_crs({'init': 'epsg:4326'})

    # Spatial join
    # This must map the population onto the weather grid since
    # the UK weather grid contains 128022 points!
    print(' spatial join .....')
    gdf = gpd.sjoin(gdf, weather_grid, how="left", op='within')

    # Sum up population
    s = gdf.groupby('index_right')['TOT_P'].sum()
    # Write results to interim path
    s.to_pickle(population_filename)

mapped_population = s

print(mapped_population)
# weight weather by population
# iterate through population df gettting lat, long
# if lat,long in weather * by population/sum(population)

# weather.loc[weather['latitude'] == 60.0 and weather['longtude'] == 1.75]
#mapped_population.index = mapped_population['index_right']
tot = mapped_population.sum()
newdata = {}
for parm in parameters.keys():
    newdata[parm] = []

# for each latitude and longitude ...
for i,row in mapped_population.iteritems():
    lat=i[0]
    lon=i[1]
#   print(lat,lon,row)
    # for each parameter ...
    for parm in parameters.keys():
        wparm = weather[parm]
        xx=wparm[lat, lon]
#       print(xx)
        # multiply by population for the grid square
        xx=xx * row
#       print(xx) 
        newdata[parm].append(xx)

dfs = {}
for parm in parameters.keys():
    print('PARM {}'.format(parm))
    for p in range(len(newdata[parm])):
        if p==0:
            dfs[parm] = newdata[parm][0]
        else:
            dfs[parm] = dfs[parm] + newdata[parm][p]
        print(p)
#       print(dfs[parm])

newdf = pd.concat(dfs, axis=1)
# divide by total population
newdf = newdf / tot
print(newdf)

# add correlation variables:
#   hour of the day - can't have it because gas only daily!
#   month of the year
newdf['month'] = newdf.index.month
#   (day of the year )
#   previous days temperature
#   wind speed from u and v
newdf['wind'] = np.sqrt(np.square(newdf['wind_u']) + np.square(newdf['wind_v']))

print(newdf.index)
#index = pd.DatetimeIndex(newdf.index)
#newdf.index = index
print(newdf)

# convert to daily because only have daily gas
matrix = newdf.resample('D', axis=0).mean()
print(matrix)

# add gas
# read historic gas demand
year = '2018'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + year + '.csv'
gas = readers.read_gas(gas_filename)

# Convert gas energy from kWh to MWh
gas = gas * (10 ** -3)
print(gas)

matrix['gas'] = gas.values

print(matrix)
print(matrix.columns)

corrMatrix = matrix.corr()
print (corrMatrix)

sn.heatmap(corrMatrix, annot=True)
plt.show()
