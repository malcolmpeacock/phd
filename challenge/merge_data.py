# python script to merge all the data into one csv and add additional stuff
# like holidays and csc_ghi.
# also output the forecast file for next weeks weather.

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
output_dir = "/home/malcolm/uclan/challenge/output/"

# demand data
demand_filename = '{}demand_fixed_{}.csv'.format(output_dir, dataset)
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(demand)

# pv data
pv_filename = '{}pv_fixed_{}.csv'.format(output_dir, dataset)
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(pv)

# weather data
weather_filename = '{}weather_fixed_{}.csv'.format(output_dir, dataset)
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(weather)

history = weather.loc[pv.index]
forecast_index = pd.date_range(start = history.last_valid_index() + pd.Timedelta(minutes=30) , end= history.last_valid_index() + pd.Timedelta(days=6, hours=23, minutes=30), freq='30min')
forecast = weather.loc[forecast_index]
# add in a last 30 minutes row, because the weather upsample ended at the hour
last_index = forecast.last_valid_index() + pd.Timedelta(minutes=30)
last_row = pd.DataFrame(forecast[-1:].values, index=[last_index], columns=forecast.columns)
forecast = forecast.append(last_row)
print(forecast)

# stick it all together
df = pd.concat([pv, history], axis=1)
df = df.join(demand, how='inner')
#df = pd.concat([demand, pv, history], axis=1)
print(df)

# clear sky irradiance
# Create location object to store lat, lon, timezone
# for location of solar farm in devon.
lat = 50.33
lon = -4.034
#site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('Europe/London'))
site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('UTC'))
times = df.index
# Generate clearsky data using the Ineichen model, which is the default
    # The get_clearsky method returns a dataframe with values for GHI, DNI,
    # and DHI
clearsky = site_location.get_clearsky(times)
#clearsky.index = pv.index
print(clearsky)
print(times)
print(clearsky.index)
df['cs_ghi'] = clearsky['ghi'].fillna(0)

# get irradiance on the tilted surface - guessing 30
tilt = 30
surface_azimuth = 180
# uses the DISC method to get DNI and then derives DHI from it
# NB: this is for location 2 - should it be mean ?
df['poa_ghi'], df['zenith'] = ghi2irradiance(site_location, tilt, surface_azimuth, df['sun2'])

# holiday indicator
holidays = ['2017-01-01', '2017-01-02', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-29', '2017-08-28', '2017-12-25', '2017-12-26', '2018-01-01', '2018-03-30', '2018-05-07', '2018-05-28', '2018-08-27', '2018-12-25', '2018-12-26']
df['holiday'] = 0
for holiday in holidays:
#   if holiday in weather.index.date:
#   df[holiday+' 00:00:00' : holiday+' 23:30:00']['holiday'] = 1
    df.loc[holiday+' 00:00:00' : holiday+' 23:30:00','holiday'] = 1

days = df.resample('D', axis=0).mean().index.date
for day in days:
#   print(day)
    if day.weekday() > 4:
        day_str = day.strftime('%Y-%m-%d')
#       print('weekend: ' + day_str)
#       df[day_str+' 00:00:00' : day_str+' 23:30:00']['holiday'] = 1
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','holiday'] = 1

# add period, k
df['k'] = (df.index.hour * 2) + (df.index.minute / 30) + 1
df['k'] = df['k'].astype(int)

# plot weather
if args.plot:
    df['temp1'].plot(label='temperature 1', color='red')
    df['temp2'].plot(label='temperature 2', color='blue')
    plt.title('temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Temp (degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    fewdays = df['2018-06-01 00:00:00' : '2018-06-04 23:30:00']
    fewdays['sun1'].plot(label='sun 1', color='red')
    fewdays['sun2'].plot(label='sun 2', color='green')
    fewdays['cs_ghi'].plot(label='clear sky ghi', color='blue')
    fewdays['poa_ghi'].plot(label='sun 2 poa 30 south', color='orange')
    plt.title('Solar Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()


print(df.columns)
#weather.columns = ['temp3', 'temp6', 'temp2', 'temp4', 'temp5', 'temp1', 'sun3', 'sun6', 'sun2', 'sun4', 'sun5', 'sun1', 'cs_ghi', 'poa_ghi', 'zenith', 'holiday', 'k']

output_dir = "/home/malcolm/uclan/challenge/output/"

# output merged data.
output_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df.to_csv(output_filename, float_format='%.2f')

# output weather forecast
output_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')

