# python script to merge all the data into one csv and add additional stuff
# like holidays and csc_ghi.
# also output the forecast file for next weeks weather.
#
# pv_ghi     global horizontal irradiance from the pv data
# pv_power   power from the pv data
# pv_temp    panel temperature from the pv data
# temp1, 6   temperature from weather locations 1 to 6
# sun1, 6    irradiance from weather locations 1 to 6
# tempw      temperature at pv location weighted average of 4 surrounding weather
# sunw       irradiance at pv location weighted average of 4 surrounding weather
# tempm      mean temperature from all 6 weather locations
# k          half hour period of the day  ( based on UTC)
# dsk        half hour period of the day  ( accounting for clocks changing)
# zenith     solar zenith angle ( its getting dark if > 87 )
# cs_ghi     clear sky theorectical GHI based on sun position
# poa_ghi    ghi on plain of array for sun2 assuming 30 tilt south facing
# demand     demand data
# sh         school holiday =1, 0 otherwise
# ph         public holiday =1, 0 otherwise
# wd         day of the week

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

def sethols(df):
    # school holidays
    sh = ['2017-12-18', '2017-12-19', '2017-12-20', '2017-12-21', '2017-12-22', '2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28', '2017-12-29', '2017-12-30', '2018-01-01', '2018-02-12', '2018-02-13', '2018-02-14', '2018-02-15', '2018-02-16', '2018-03-30', '2018-04-02', '2018-04-03', '2018-04-04', '2018-04-05', '2018-04-06', '2018-04-09', '2018-04-10', '2018-04-11', '2018-04-12', '2018-04-13', '2018-07-30', '2018-07-31', '2018-08-01', '2018-08-02', '2018-08-03', '2018-08-06', '2018-08-07', '2018-08-08', '2018-08-09', '2018-08-10', '2018-08-13', '2018-08-14', '2018-08-15', '2018-08-16', '2018-08-17', '2018-08-20', '2018-08-21', '2018-08-22', '2018-08-23', '2018-08-24', '2018-08-27', '2018-08-28', '2018-08-29', '2018-08-30', '2018-08-31', '2018-10-22', '2018-10-23', '2018-10-24', '2018-10-25', '2018-10-26', '2018-12-21', '2018-12-24', '2018-12-25', '2018-12-26', '2018-12-27', '2018-12-28', '2018-12-31', '2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-07']
    df['sh'] = 0
    for holiday in sh:
        df.loc[holiday+' 00:00:00' : holiday+' 23:30:00','sh'] = 1
    df['sh'] = df['sh'].astype(int)

    # public holidays indicator
    ph = ['2017-01-01', '2017-01-02', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-29', '2017-08-28', '2017-12-25', '2017-12-26', '2018-01-01', '2018-03-30', '2018-05-07', '2018-05-28', '2018-08-27', '2018-12-25', '2018-12-26']
    df['ph'] = 0
    for holiday in ph:
        df.loc[holiday+' 00:00:00' : holiday+' 23:30:00','ph'] = 1
    df['ph'] = df['ph'].astype(int)

    # day of the week
    df['wd'] = 0
    days = df.resample('D', axis=0).mean().index.date
    for day in days:
        day_str = day.strftime('%Y-%m-%d')
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','wd'] = day.weekday()
    df['wd'] = df['wd'].astype(int)

    # dst indicator
    df['dsk'] = df['k'] + 2 * (df.index.hour - df.index.tz_localize('UTC').tz_convert(tz=pytz.timezone('Europe/London')).hour + 1)
    df.loc[df['dsk']==95, 'dsk'] = 47
    df.loc[df['dsk']==96, 'dsk'] = 48

def get_zenith(site_location, index):
    solar_position = site_location.get_solarposition(times=index)
    return solar_position['apparent_zenith']

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
parser.add_argument('--raw', action="store_true", dest="raw", help='Use the original uncorrected data', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
output_dir = "/home/malcolm/uclan/challenge/output/"

# demand data
demand_filename = '{}demand_fixed_{}.csv'.format(output_dir, dataset)
if args.raw:
    demand_filename = '{}demand_train_{}.csv'.format(input_dir, dataset)
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(demand)

# pv data
pv_filename = '{}pv_fixed_{}.csv'.format(output_dir, dataset)
if args.raw:
    pv_filename = '{}pv_train_{}.csv'.format(input_dir, dataset)
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
if args.raw:
    pv.columns = ['pv_ghi','pv_power','pv_temp']
print(pv)

# weather data
weather_filename = '{}weather_fixed_{}.csv'.format(output_dir, dataset)
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# add period, k
weather['k'] = utils.index2ks(weather.index)
# set holidays

sethols(weather)

# clear sky irradiance
# Create location object to store lat, lon, timezone
# for location of solar farm in devon.
lat = 50.33
lon = -4.034
#site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('Europe/London'))
site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('UTC'))
times = weather.index
weather['zenith'] = get_zenith(site_location, times)
# Generate clearsky data using the Ineichen model, which is the default
# The get_clearsky method returns a dataframe with values for GHI, DNI,
# and DHI
clearsky = site_location.get_clearsky(times)
weather['cs_ghi'] = clearsky['ghi'].fillna(0)
print(weather)

# split off the historical part of the weather and the forecast
history = weather.loc[pv.index]
forecast_index = pd.date_range(start = history.last_valid_index() + pd.Timedelta(minutes=30) , end= history.last_valid_index() + pd.Timedelta(days=6, hours=23, minutes=30), freq='30min')
forecast = weather.loc[forecast_index]
# add in a last 30 minutes row, because the weather upsample ended at the hour
last_index = forecast.last_valid_index() + pd.Timedelta(minutes=30)
last_row = pd.DataFrame(forecast[-1:].values, index=[last_index], columns=forecast.columns)
forecast = forecast.append(last_row)
# add the holiday flag
#sethols(forecast)
forecast.index.rename('datetime', inplace=True)

print(forecast)

# stick it all together
df = pd.concat([pv, history], axis=1)
df = df.join(demand, how='inner')
#sethols(df)
print(df)


# get irradiance on the tilted surface - guessing 30
tilt = 30
surface_azimuth = 180
# uses the DISC method to get DNI and then derives DHI from it
# NB: this is for location 2 - should it be mean ?
df['poa_ghi'], df['zenith'] = ghi2irradiance(site_location, tilt, surface_azimuth, df['sun2'])


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
    fewdays['sun5'].plot(label='sun 5', color='purple')
    fewdays['sun6'].plot(label='sun 6', color='yellow')
    fewdays['sunw'].plot(label='sun w', color='black')
    fewdays['cs_ghi'].plot(label='clear sky ghi', color='blue')
    fewdays['poa_ghi'].plot(label='sun 2 poa 30 south', color='orange')
    fewdays['pv_ghi'].plot(label='pv ghi', color='brown')
    plt.title('Solar Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    plt.scatter(df['tempm'].values, df['demand'].values, s=12, color='blue')
    plt.title('Demand vs temperature')
    plt.xlabel('Demand (MWh)', fontsize=15)
    plt.ylabel('Temperature (Degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()


print(df.columns)
for col in df.columns:
    if df[col].isna().sum() >0:
        print("ERROR nans in {}".format(col))
        print(df[col].isnull().values)
        quit()


output_dir = "/home/malcolm/uclan/challenge/output/"

# output merged data.
output_filename = '{}merged_{}.csv'.format(output_dir, dataset)
if args.raw:
    output_filename = '{}merged_raw_{}.csv'.format(output_dir, dataset)
df.to_csv(output_filename, float_format='%.2f')

# output weather forecast
output_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
if args.raw:
    output_filename = '{}forecast_raw_{}.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')

