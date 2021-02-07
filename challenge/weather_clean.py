# python script to clean weather data and split into the historic and
# the forecast weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
import pvlib
# Import tensor dataset & data loader
# Import nn.functional

# custom code
import utils

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
    return POA_irradiance['poa_global'], solar_position['apparent_zenith']

# main program

# process command line

parser = argparse.ArgumentParser(description='Clean weather data.')
parser.add_argument('set', help='weather file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"

# weather data
weather_filename = "{}weather_train_{}.csv".format(input_dir,dataset)
print('Cleaning {} {}'.format(dataset, weather_filename) )
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# up sample to 30 mins (with interpolation)
weather = weather.resample('30min').interpolate()

print(weather)

# pv data ( to create a weather file of same length )
pv_filename = "{}pv_train_{}.csv".format(input_dir,dataset)
print('Reading {} {}'.format(dataset, pv_filename) )
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(pv)

# Weather
n_missing = utils.missing_times(weather, '60min')
if n_missing>0:
    print("Missing rows in weather {}".format(n_missing) )
for col in weather.columns:
    if weather[col].isna().sum() >0:
        print("ERROR nans in {}".format(col))
        print(weather[col].isnull().values)
        quit()

# TODO add ( bilinear interpolation ) for ( 50.33, -4.034 )
# TODO add mean weather of the 6 points
#weather['temp_mean'] = weather['temp_location3']
#weather['solar_mean'] = weather['solar_location3']

# clear sky irradiance
# Create location object to store lat, lon, timezone
# for location of solar farm in devon.
lat = 50.33
lon = -4.034
#site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('Europe/London'))
site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('UTC'))
times = pv.index
# Generate clearsky data using the Ineichen model, which is the default
    # The get_clearsky method returns a dataframe with values for GHI, DNI,
    # and DHI
clearsky = site_location.get_clearsky(times)
#clearsky.index = pv.index
print(clearsky)
print(times)
print(clearsky.index)
weather['cs_ghi'] = clearsky['ghi'].fillna(0)

#pv = pd.concat([pv,clearsky['ghi']],axis=1)
#pv = pv.append(clearsky)

# get irradiance on the tilted surface - guessing 30
tilt = 30
surface_azimuth = 180
# uses the DISC method to get DNI and then derives DHI from it
# NB: this is for location 2 - should it be mean ?
weather['poa_ghi'], weather['zenith'] = ghi2irradiance(site_location, tilt, surface_azimuth, weather['solar_location2'])

# holiday indicator
holidays = ['2017-01-01', '2017-01-02', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-29', '2017-08-28', '2017-12-25', '2017-12-26', '2018-01-01', '2018-03-30', '2018-05-07', '2018-05-28', '2018-08-27', '2018-12-25', '2018-12-26']
weather['holiday'] = 0
for holiday in holidays:
#   if holiday in weather.index.date:
    weather[holiday+' 00:00:00' : holiday+' 23:30:00']['holiday'] = 1

days = weather.resample('D', axis=0).mean().index.date
for day in days:
    print(day)
    if day.weekday() > 4:
        day_str = day.strftime('%Y-%m-%d')
        print('weekend: ' + day_str)
        weather[day_str+' 00:00:00' : day_str+' 23:30:00']['holiday'] = 1

# plot weather
if args.plot:
    weather['temp_location1'].plot(label='temperature 1', color='red')
    weather['temp_location2'].plot(label='temperature 2', color='blue')
    plt.title('temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Temp (degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    fewdays = weather['2018-06-01 00:00:00' : '2018-06-04 23:30:00']
    fewdays['solar_location1'].plot(label='sun 1', color='red')
    fewdays['solar_location2'].plot(label='sun 2', color='green')
    fewdays['cs_ghi'].plot(label='clear sky ghi', color='blue')
    fewdays['poa_ghi'].plot(label='sun 2 poa 30 south', color='orange')
    plt.title('Solar Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()


print(weather.columns)
weather.columns = ['temp3', 'temp6', 'temp2', 'temp4', 'temp5', 'temp1', 'sun3', 'sun6', 'sun2', 'sun4', 'sun5', 'sun1', 'cs_ghi', 'poa_ghi', 'zenith', 'holiday']

history = weather.loc[pv.index]
forecast_index = pd.date_range(start = history.last_valid_index() + pd.Timedelta(minutes=30) , end= history.last_valid_index() + pd.Timedelta(days=6, hours=23, minutes=30), freq='30min')
forecast = weather.loc[forecast_index]
print(forecast)

output_dir = "/home/malcolm/uclan/challenge/output/"

# output historic weather
output_filename = '{}weather_{}.csv'.format(output_dir, dataset)
history.to_csv(output_filename, float_format='%.2f')
# output weather forecast
output_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')
