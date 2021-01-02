# Script to create PV generation time series
# Input: irradiance from ERA5 or MIDAS
#        
# Output:hourly time series of pv generation in yearly files.

# Python modules
import os
import math
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
from utils.readers import read_midas_irradiance
import utils.bilinear as bil
import heat.scripts.download as download
import heat.scripts.read as read
from utils.midas_locations import get_locations
from utils.sanity import sanity
import pytz
import pvlib
import numpy as np

def ghi2irradiance(site_location, tilt, surface_azimuth, in_ghi):
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(times=in_ghi.index)
    # Get the direct normal component of the solar radiation
    disc = pvlib.irradiance.disc(
        in_ghi,
        solar_position['apparent_zenith'],
        in_ghi.index.dayofyear)
    in_dni = disc['dni']
    # Get the diffuse component of the solar radiation
    in_dhi = in_ghi - in_dni * np.cos(np.radians(solar_position['apparent_zenith']))
    # Get the irradiance on the plane of the solar array.
    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=in_dni,
        ghi=in_ghi,
        dhi=in_dhi,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])
    # return poa
    return POA_irradiance['poa_global']

locations = get_locations()

midas_dir = "/home/malcolm/uclan/data/midas/"

# process command line

parser = argparse.ArgumentParser(description='Generate pv energy time series.')
parser.add_argument('year', type=int, help='Weather year')
parser.add_argument('--nyears', type=int, action="store", dest="nyears", help='Number of years', default=1 )
parser.add_argument('--location', action="store", dest="location", help='Location to generate for z for all: ', default='z' )
parser.add_argument('--weather', action="store", dest="weather", help='Weather source', default='midas' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--debug', action="store_true", dest="debug", help='Debug 30 values of irradiance only', default=False)

args = parser.parse_args()

years=[]
for year in range(args.year,args.year+args.nyears):
    years.append(str(year))

if args.location == 'z':
    generators={}
    for key, location in locations.items():
        if location['pv'] == 'y':
            generators[key] = location
else:
    generators={ args.location : locations[args.location]}

print(generators)

weather_source = args.weather

print('Options- weather: {} '.format( weather_source))

input_path = "/home/malcolm/uclan/input/"
interim_path = "/home/malcolm/uclan/interim/"

tilt = 30
surface_azimuth = 180

power_years = {}

# for each year ...
for year in years:
    pv_locations = {}
# weather
    # MIDAS weather
    if weather_source == 'midas':
        print('Using weather data from MIDAS')
        # for each location ...
        for key, generator in generators.items():
            # read midas weather file
            filename = "midas-open_uk-radiation-obs_dv-201908_" + generator['name'] + "_qcv-1_" + year + ".csv"
#           pv_hourly = read_midas_irradiance(midas_dir+filename,['glbl_irad_amt', 'difu_irad_amt', 'direct_irad'])
            pv_hourly = read_midas_irradiance(midas_dir+filename,['glbl_irad_amt'])
            # sanity checks
            pv_hourly = sanity(pv_hourly,year,generator['name'])
            # convert from KJ to w/ms
            pv_hourly['glbl_irad_amt'] = pv_hourly['glbl_irad_amt'] * 0.2777777777
            # convert to a series (GHI only)
            pv_hourly = pv_hourly.squeeze()
#           print(pv_hourly)
            pv_locations[key] = pv_hourly
    # ERA5 weather
    else:
        print('Using weather data from ERA5')
        download.weather_era5(input_path, year, 1, 'I', 'pv',  ['surface_solar_radiation_downwards', 'surface_solar_radiation_downward_clear_sky'], [ 60, -8, 48, 2, ])
        df_ir = read.weather_era5(input_path, year, 1, 'I', 'pv','ssrd')
        df_cs = read.weather_era5(input_path, year, 1, 'I', 'pv','ssrdc')
        # for each location ...
        for key, generator in generators.items():
            # blinearly interpolate irradiance from weather grid
            coords = generator['coords']
            lat = coords[0]
            lon = coords[1]
            print('Bilinear interpolation: {} {} {} '.format(key, lat, lon))
            df_bl,x,y = bil.bilinear(lat, lon, df_ir)
            era5_ir = df_bl['t_interp']
            # convert from J/m2 to wh/m2
            era5_ir = era5_ir * 0.000277778
            pv_locations[key] = era5_ir


    # calculate power from the weather

    power_locations = {}
    for key, generator in generators.items():
        pv_hourly = pv_locations[key]
        # reduce to only 30 values for debugging
        if args.debug:
            pv_hourly = pv_hourly.head(30)
            print(pv_hourly)
        # sanity check
        print(len(pv_hourly))
        print(pv_hourly.nlargest())
        # Create location object to store lat, lon, timezone
        coords = generator['coords']
        lat = coords[0]
        lon = coords[1]
        site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('Europe/London'))
        # get irradiance on the tilted surface
        # uses the DISC method to get DNI and then derives DHI from it
        surface_ghi = ghi2irradiance(site_location, tilt, surface_azimuth, pv_hourly)
        if args.plot:
            surface_ghi.plot()
            plt.title('Irradiance for : ' + str(year) + ' ' +generator['name'])
            plt.xlabel('Hour', fontsize=15)
            plt.ylabel('Irradiance (w/m2)', fontsize=15)
            plt.show()

        # generate pv energy series
        # TODO multiply by performance ratio (PR) of 0.85 ??
        surface_ghi = surface_ghi * 0.85
#       power=[]
#       power_series = pd.Series(surface_ghi,index=pv_hourly.index,name='power')
        power_series = surface_ghi.rename('power')

        # stick power and irradiance into the same dataframe
        power_hourly = pd.concat([pv_hourly,power_series],axis=1)
        power_hourly.columns = ['ghi_' + key, 'power_'+key]

        if args.plot:
            # plot irradiance on horizontal
            ax = power_hourly['ghi_' + key].plot(color='blue')
            plt.ylabel('GHI (w/m2)', fontsize=15, color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Power(kW)',color='red', fontsize=15)
            # plot pv power (after POA and PR)
            power_hourly['power_' + key].plot(ax=ax2,color='red')
            plt.title('PV power : ' + str(year) + ' ' +generator['name'])
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
output_dir = "/home/malcolm/uclan/output/pv/";
output_file = weather_source + str(args.year) + str(args.nyears) + ".csv"
df.to_csv(output_dir + output_file, sep=',', decimal='.', float_format='%g')
