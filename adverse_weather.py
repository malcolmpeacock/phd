# read MET Office Adverse weather files.

import os
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset, num2date
from shapely.geometry import Point
import numpy as np

# heat stuff
import heat.scripts.preprocess as preprocess

# TODO - this needs to work on whatever the adverse grid is not heat
#      - also this needs to be in the heat series program for heat demand
#   pv generation and wind don't need a population weighting.

def read_population(weather):
    input_path = '/home/malcolm/uclan/tools/python/scripts/heat/input'
    directory = 'Version 2_0_1'
    filename = 'GEOSTAT_grid_POP_1K_2011_V2_0_1.csv'
    population_file = '/home/malcolm/uclan/output/population.adverse.pickle'

    if not os.path.isfile(population_file):
        # Read population data
        df = pd.read_csv(os.path.join(input_path, directory, filename),
                         usecols=['GRD_ID', 'TOT_P', 'CNTR_CODE'],
                         index_col='GRD_ID')

        # Make GeoDataFrame from the the coordinates in the index
        population = gpd.GeoDataFrame(df)
        population['geometry'] = df.index.map(lambda i: Point(
            [1000 * float(x) + 500 for x in reversed(i.split('N')[1].split('E'))]
        ))

        # Transform coordinate reference system to 'latitude/longitude'
        population.crs = {'init': 'epsg:3035'}

        # Filter population data by country to cut processing time
        gdf = population[population['CNTR_CODE'] == 'UK'].copy()

        # Align coordinate reference systems
        print(' aligning coords .....')
        gdf = gdf.to_crs({'init': 'epsg:4326'})

        mapped_population = {}

        # Make GeoDataFrame from the weather data coordinates
        weather_grid = gpd.GeoDataFrame(index=weather_data.columns)
        weather_grid['geometry'] = weather_grid.index.map(lambda i: Point(reversed(i)))

        # Set coordinate reference system to 'latitude/longitude'
        weather_grid.crs = {'init': 'epsg:4326'}

        # Make polygons around the weather points
        weather_grid['geometry'] = weather_grid.geometry.apply(lambda point: point.buffer(.75 / 2, cap_style=3))

        # Spatial join
        # This must map the population onto the weather grid since
        # the UK weather grid contains 128022 points!
        print(' spatial join .....')
        gdf = gpd.sjoin(gdf, weather_grid, how="left", op='within')

        # Sum up population
        s = gdf.groupby('index_right')['TOT_P'].sum()

        # Remove NI if GB
#       if country == 'GB':
#           s = split_northern_ireland(s)
#       if country == 'NI':
#           s = split_northern_ireland(s,True)
        # Write results to interim path
        s.to_pickle(filename)

    else:

        s = pd.read_pickle(filename)
        print('{} already exists and is read from disk.'.format(file))

    mapped_population = s

    return mapped_population

# ERA5 file name
event = 'winter_wind_drought'
period = '5'
warming = '2-4'
eno = '1'
parm = 'windspeed'
filename = '/home/malcolm/uclan/data/adverse/{}_uk_return_period_1_in_{}_years_duration_gwl1{}degC_event{}_{}.nc'.format(event, period, warming, eno, parm)

# Read the netCDF file
print('Reading netcdf file {} ...'.format(filename))
nc = Dataset(filename)
print(nc.variables)
time = nc.variables['time'][:]
time_units = nc.variables['time'].units
latitude = nc.variables['latitude'][:]
longitude = nc.variables['longitude'][:]
wind = nc.variables['wind_speed'][:]

times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)

# TODO can't use this because its a different grid so need to clone the 
# population mapping part.
print('Getting Mapped population ... ')

heat_path = "/home/malcolm/uclan/tools/python/scripts/heat/"
input_path = heat_path + 'input'
interim_path = heat_path + 'interim'
year = '2018'
mapped_population = preprocess.map_population(input_path, interim_path, 'GB', False, year, 'I', False)
print(mapped_population)
total_population = mapped_population.sum()

# Transform to pd.DataFrame
print('Creating DataFrame ...')

df = pd.DataFrame(data=wind.reshape(len(time), len(latitude) * len(longitude)), index=pd.DatetimeIndex(times, name='time'), columns=pd.MultiIndex.from_product([latitude, longitude], names=('latitude', 'longitude')))
print(df)
quit()
print('Weighting {} by population ...'.format(key))
location_series=[]
for location in mapped_population.index.tolist():
    # if this location is part of GB then its in mapped_population so we
    # want it, otherwise drop it
    if location in mapped_population.index:
        print('Location found {} {} {}'.format(key, location[0], location[1]) )
        df_wo = df[(location[0], location[1])]
#       print(df_wo)
        pop =  mapped_population[location]
#       print(pop)
        location_series.append(df_wo * pop / total_population)
    else:
        print('Location zero {} {} {}'.format(location[0], location[1]) )
dfl = pd.concat(location_series, axis=1)
print(dfl)
wdata = dfl.sum(axis=1)
print(wdata)

dft = pd.DataFrame(data=wdata)
print(dft)

#dft.to_pickle('dft.pickle')

output_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(year)
dft.to_csv(output_file, sep=',', decimal='.', float_format='%g')
