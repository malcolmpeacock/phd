# Script to create wind generation time series
# Input: wind speeds from ERA5 or MIDAS
#        turbine power curve and location
# Output:hourly time series of wind generation in yearly files.

# Python modules
import os
import math
# import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
import numpy as np
from netCDF4 import Dataset, num2date

# Custom scripts
from utils.readers import read_midas_hourly
import utils.bilinear as bil
import heat.scripts.download as download
import heat.scripts.read as read
from utils.midas_locations import get_locations
from utils.sanity import sanity

def power_test(ws):
    return ws+2.3

# def power_from_curve(ws,curve):
def power_from_curve(ws):
  exactmatch=curve[curve.Windspeedms==ws]
  if not exactmatch.empty:
      return exactmatch.HSkw
  else:
      aboves = curve[curve.Windspeedms>ws]
      # above cut out speed
      if len(aboves) == 0:
          return 0.0
      belows = curve[curve.Windspeedms<ws]
      # below cut in speed
      if len(belows) == 0:
          return 0.0
      lowerneighbour_ind = curve[curve.Windspeedms<ws].Windspeedms.idxmax()
#     upperneighbour_ind = curve[curve.Windspeedms>ws].Windspeedms.idxmin()
      row = curve.iloc[lowerneighbour_ind]
      return row.HSkw

locations = get_locations()

midas_dir = "/home/malcolm/uclan/data/midas/"
output_dir = "/home/malcolm/uclan/output/wind/";

# process command line

parser = argparse.ArgumentParser(description='Generate wind energy time series.')
parser.add_argument('year', type=int, help='Weather year')
parser.add_argument('--nyears', type=int, action="store", dest="nyears", help='Number of years', default=1 )
parser.add_argument('--location', action="store", dest="location", help='Location to generate for z for all: ', default='z' )
parser.add_argument('--weather', action="store", dest="weather", help='Weather source: midas, era5 or adv', default='midas' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--debug', action="store_true", dest="debug", help='Debug 30 values of wind speed only', default=False)

args = parser.parse_args()

years=[]
for year in range(args.year,args.year+args.nyears):
    years.append(str(year))

if args.location == 'z':
    generators={}
    for key, location in locations.items():
        if location['wind'] == 'y':
            generators[key] = location
else:
    generators={ args.location : locations[args.location]}

#print(generators)

weather_source = args.weather

print('Options- weather: {} '.format( weather_source))

# heat_path = os.path.realpath('heat')

# input_path = os.path.join(home_path, 'input')
# output_path = os.path.join(home_path, 'output', version)
input_path = "/home/malcolm/uclan/input/"

# read turbine power curve
power_file = "/home/malcolm/uclan/data/n902500power.csv"
curve = pd.read_csv(power_file, header=0, usecols=['Windspeedms','HSkw'] )
#print(curve)
if args.plot:
    curve.plot(x='Windspeedms',y='HSkw')
    plt.title('Wind turbine power curve')
    plt.xlabel('Wind speed (m/s)', fontsize=15)
    plt.ylabel('Power (kW)', fontsize=15)
    plt.show()
test1 = power_from_curve(2.0)
test2 = power_from_curve(3.5)
test3 = power_from_curve(4.2)
test4 = power_from_curve(25.2)
#print("ws 2.0 {} ws 3.5 {} ws 4.2 {} ws 25.2 {}".format(test1,test2,test3,test4))
# heights of 70m, 80, 100m given in docs
turbine_height = 100.0
surface_roughness = 0.03
turbine_rated_power = 2500
wind_height = 10.0
# to get the wind speed at hub height.
log_law_factor = math.log(turbine_height/surface_roughness) / math.log(wind_height/surface_roughness)

power_years = {}

# for each year ...
for year in years:
    wind_locations = {}
    surface_roughness = {}
# weather
    # MIDAS weather
    if weather_source == 'midas':
        print('Using weather data from MIDAS')
        # for each location ...
        for key, generator in generators.items():
            if generator["wind"] == 's':
                print('Marine station')
                # read weather file previously processed from midas marine
                filename = generator['name'] + '_' + year + ".csv"
                midas_marine = pd.read_csv(output_dir+filename, header=0, parse_dates=[0], index_col=0 )
                # TODO sanity ??
                # convert from knots to m/s
                wind_hourly = 0.514444 * midas_marine['windspeed']
                print(wind_hourly)
                # calculate surface roughness
                d = generator['depth']
                g = 9.8
                rootgd = math.sqrt(g * d)
                beta = midas_marine['msr_wave_height'] / ( rootgd * midas_marine['msr_wave_per'].astype(float))
                z0 = (beta * wind_hourly * wind_hourly ) / g
                print(z0)
                le05 = len(z0[z0<0.5])
                le01 = len(z0[z0<0.1])
                le001 = len(z0[z0<0.01])
                print("Z0 max {:.2f} min {:.2f} less than 0.5 {} less than 0.1 {} less than 0.01 {}".format(z0.max(), z0.min(), le05, le01, le001))
                # set a minimum surface roughness
                z0[z0<0.0001] = 0.0001
                wind_locations[key] = wind_hourly
                surface_roughness[key] = z0
            else:
                print('Land station')
                # read midas weather file
                filename = "midas-open_uk-hourly-weather-obs_dv-201908_" + generator['name'] + "_qcv-1_" + year + ".csv"
                wind_hourly = read_midas_hourly(midas_dir+filename,['wind_speed'])
                # sanity checks
                wind_hourly = sanity(wind_hourly,year,generator['name'])
                # convert from knots to m/s
                wind_hourly['wind_speed'] = 0.514444 * wind_hourly['wind_speed']
                wind_hourly = wind_hourly.squeeze()
                wind_hourly.index.rename('time', inplace=True)
                wind_locations[key] = wind_hourly
    else:
        if weather_source == 'adv':
        # Adverse climate change weather events
            print('Using advserse weather event data for climate change')
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
            df = pd.DataFrame(data=wind.reshape(len(time), len(latitude) * len(longitude)), index=pd.DatetimeIndex(times, name='time'), columns=pd.MultiIndex.from_product([latitude, longitude], names=('latitude', 'longitude')))
            print(df)
            # for each location ...
            for key, generator in generators.items():
                # blinearly interpolate wind from weather grid
                coords = generator['coords']
                lat = coords[0]
                lon = coords[1]
                bl,x,y = bil.bilinear(lat, lon, df)
                wind = bl['t_interp']
                wind_hourly = wind.rename('wind_speed')
                wind_locations[key] = wind_hourly

        else:
        # ERA5 weather
            print('Using weather data from ERA5')
            parameters = {
                'wind_u': 'u10',
                'wind_v': 'v10'
            }

            download.weather_era5(input_path, year, 1, 'I', 'wind',  ['10m_u_component_of_wind', '10m_v_component_of_wind'], [ 60, -8, 50, 2, ])
            df_u = read.weather_era5(input_path, year, 1, 'I', 'wind','u10')
            df_v = read.weather_era5(input_path, year, 1, 'I', 'wind','v10')
            print(df_v)
            # for each location ...
            for key, generator in generators.items():
                # blinearly interpolate wind from weather grid
                coords = generator['coords']
                lat = coords[0]
                lon = coords[1]
                bl_u,x,y = bil.bilinear(lat, lon, df_u)
                bl_v,x,y = bil.bilinear(lat, lon, df_v)
                wind_u = bl_u['t_interp']
                wind_v = bl_v['t_interp']
                wind_squared = wind_u.pow(2) + wind_u.pow(2)
                wind_hourly = wind_squared.pow(1/2).rename('wind_speed')
                wind_locations[key] = wind_hourly

    # calculate power from the weather

    power_locations = {}
    for key, generator in generators.items():
        print("Calculating power for {}".format(key))
        wind_hourly = wind_locations[key]
        if args.debug:
            wind_hourly = wind_hourly.head(30)
            print(wind_hourly)
        if args.plot:
            wind_hourly.plot()
            plt.title('Wind speed for : ' + str(year) + ' ' +generator['name'])
            plt.xlabel('Hour', fontsize=15)
            plt.ylabel('Wind speed (m/s)', fontsize=15)
            plt.show()
        # convert wind speed to turbine height using log law.
        if generator["wind"] == 's':
            z0 = surface_roughness[key]
            toverz0 = turbine_height * np.reciprocal(z0)
            woverz0 = wind_height * np.reciprocal(z0)
            log_law_factor = np.log(toverz0) / np.log(woverz0)
            wind_hourly = wind_hourly * log_law_factor
        else:
            wind_hourly = wind_hourly * log_law_factor
        # generate wind energy series
        power=[]
        #   this didn't work
        #   power_hourly = wind_hourly.apply(power_from_curve)
        for index, row in  wind_hourly.items():
            power.append(power_from_curve(row))
        power_series = pd.Series(power,index=wind_hourly.index,name='power')

        # stick power and wind speed into the same dataframe
        power_hourly = pd.concat([wind_hourly,power_series],axis=1)
        power_hourly.columns = ['wind_' + key, 'power_'+key]

        if args.plot:
            ax = power_hourly['wind_' + key].plot(color='blue')
            plt.ylabel('Wind speed (m/s)', fontsize=15, color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Power(kW)',color='red', fontsize=15)
            power_hourly['power_' + key].plot(ax=ax2,color='red')
            plt.title('Wind power : ' + str(year) + ' ' +generator['name'])
            plt.xlabel('Hour', fontsize=15)
            plt.show()

        power_locations[key] = power_hourly

    power_years[year] = pd.concat([power_locations[key] for key in generators.keys()] ,axis=1)

# stick all the years together at the end
df = pd.concat([power_years[year] for year in years], axis=0 )
# Timestamp
index = pd.DatetimeIndex(df.index)
df.index = index.strftime('%Y-%m-%dT%H:%M:%SZ')

if args.debug:
    print(df)
output_file = weather_source + str(args.year) + str(args.nyears) + ".csv"
df.to_csv(output_dir + output_file, sep=',', decimal='.', float_format='%g')
