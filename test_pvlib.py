# 
# Test program for pvlib and midas
# -read midas ghi, dhi, dni
# -use pv lib to 
#    get clear sky and convert to location asimuth tilt
#    convert midas to location asimuth tilt
#  plot over a day:
#    midas values
#    clear sky values
#    both sets of converted values (to azimuth, tilt)

import pvlib
import pandas as pd
import pytz
import numpy as np
import matplotlib.pyplot as plt
from utils.readers import read_midas_irradiance
import heat.scripts.download as download
import heat.scripts.read as read
import utils.bilinear as bil
import math

# From weather GHI ( Global Horizonal Irradiance ), get the irradiance
# on the plane of the solar array.

def ghi2poa(site_location, tilt, surface_azimuth, in_ghi):
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(times=in_ghi.index)
    # calculate ratio of irradiance on POA to horizontal
    ratio = pvlib.irradiance.poa_horizontal_ratio(tilt, surface_azimuth, solar_position['apparent_zenith'], solar_position['azimuth'])
#   print('in_ghi')
#   print(in_ghi)
#   print('RATIO BEFORE')
#   print(ratio)
    # TODO implelement max_zenith check
#   print('Zenith')
#   print(solar_position['apparent_zenith'])
    zenith_too_big = solar_position['apparent_zenith'] > 87.0
    ratio[zenith_too_big] = 0.0
#   print('RATIO AFTER')
#   print(ratio)
    poa_ghi = in_ghi * ratio
    df = pd.concat([solar_position['apparent_zenith'], solar_position['azimuth'], ratio, in_ghi, poa_ghi], axis=1, keys=['zenith', 'azimuth', 'ratio', 'ghi', 'poa'])
    output_dir = "/home/malcolm/uclan/output/pv/";
    output_file = date + ".csv"
    df.to_csv(output_dir + output_file, sep=',', decimal='.', float_format='%g', date_format='%Y-%m-%d %H:%M:%S')
    return poa_ghi

# From weather GHI ( Global Horizonal Irradiance ), get the irradiance
# on the plane of the solar array.

def cosine(x):
    return math.cos(math.radians(x))

def ghi2irradiance(site_location, tilt, surface_azimuth, in_ghi):
    print('ghi2irradiance')
    print(in_ghi.index)
    print(in_ghi)
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(times=in_ghi.index)
    # Get the direct normal component of the solar radiation
    disc = pvlib.irradiance.disc(
        in_ghi,
        solar_position['apparent_zenith'],
        in_ghi.index.dayofyear)
    in_dni = disc['dni']
    print(in_dni)
    # Get the diffuse component of the solar radiation
#   in_dhi = pvlib.irradiance.get_ground_diffuse(tilt, in_ghi)
    # TODO could also to this as dhi = ghi - dni cos(zenith)
#   in_dhi = in_ghi - in_dni.apply(lambda x: math.cos(x))
#   in_dhi = in_ghi - in_dni.apply(cosine)
    in_dhi = in_ghi - in_dni * np.cos(np.radians(solar_position['apparent_zenith']))
    print(in_dhi)
    # Get the irradiance on the plane of the solar array.
    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=in_dni,
        ghi=in_ghi,
        dhi=in_dhi,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])
    # write to file
    pd_dni = pd.Series(in_dni)
    pd_dni.index = in_ghi.index
    print(pd_dni)
    df = pd.concat([solar_position['apparent_zenith'], in_ghi, POA_irradiance['poa_global'], in_dhi, pd_dni], axis=1, keys=['zenith', 'ghi', 'poa', 'dhi', 'dni'])
    output_dir = "/home/malcolm/uclan/output/pv/";
    output_file = date + ".csv"
    # return poa
    return POA_irradiance['poa_global']

def ghi2poa2(site_location, tilt, surface_azimuth, in_ghi, cs):
    print('ghi2poa2')
    print(in_ghi)
    print(cs)
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(times=in_ghi.index)
    # Get the direct normal component of the solar radiation
    disc = pvlib.irradiance.dirindex(
        in_ghi,
        cs['ghi'],
        cs['dni'],
        solar_position['apparent_zenith'],
        in_ghi.index)
#   in_dni = disc['dni']
    in_dni = disc
    print(in_dni)
    # Get the diffuse component of the solar radiation
#   in_dhi = pvlib.irradiance.get_ground_diffuse(tilt, in_ghi)
    # TODO could also to this as dhi = ghi - dni cos(zenith)
#   in_dhi = in_ghi - in_dni.apply(lambda x: math.cos(x))
#   in_dhi = in_ghi - in_dni.apply(cosine)
    in_dhi = in_ghi - in_dni * np.cos(np.radians(solar_position['apparent_zenith']))
    print(in_dhi)
    # Get the irradiance on the plane of the solar array.
    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=in_dni,
        ghi=in_ghi,
        dhi=in_dhi,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])
    return POA_irradiance['poa_global']

# Calculate clear-sky GHI and transpose to plane of array
# Define a function so that we can re-use the sequence of operations with
# different locations
def get_irradiance(site_location, tilt, surface_azimuth, input_irradiance, times):
    print('get_irradiance')
    print(times)
    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(times=times)
    # Use the get_total_irradiance function to transpose the GHI to POA
    POA_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=surface_azimuth,
        dni=input_irradiance['dni'],
        ghi=input_irradiance['ghi'],
        dhi=input_irradiance['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'])
    # Return DataFrame with only GHI and POA
    return POA_irradiance['poa_global']

# Create location object to store lat, lon, timezone
lat = 53.7746
lon = -3.03647
site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('Europe/London'))

# read ERA5
input_path = "/home/malcolm/uclan/input/"
interim_path = "/home/malcolm/uclan/interim/"
year = '2018'

download.weather_era5(input_path, year, 1, 'I', 'pv',  ['surface_solar_radiation_downwards', 'surface_solar_radiation_downward_clear_sky'], [ 60, -8, 50, 2, ])
df_ir = read.weather_era5(input_path, year, 1, 'I', 'pv','ssrd')
df_ir,x,y = bil.bilinear(lat, lon, df_ir)
era5_ir = df_ir['t_interp']
df_cs = read.weather_era5(input_path, year, 1, 'I', 'pv','ssrdc')
df_cs,x,y = bil.bilinear(lat, lon, df_cs)
era5_cs = df_cs['t_interp']
# convert from J/m2 to wh/m2
era5_ir = era5_ir * 0.000277778
era5_cs = era5_cs * 0.000277778
# print(era5_ir)
# print(era5_cs)

# read midas
midas_file = "/home/malcolm/uclan/data/midas/midas-open_uk-radiation-obs_dv-201908_cornwall_01395_camborne_qcv-1_2018.csv"
pv_hourly = read_midas_irradiance(midas_file,['glbl_irad_amt', 'difu_irad_amt', 'direct_irad'])
# convert from kJ/m2 to wh/m2
pv_hourly = pv_hourly * 0.2777777777
# print(pv_hourly)

totals = { 'cs' : {}, 'cs_poa' : {}, 'midas' : {}, 'midas_poa' : {}, 'era' : {}, 'era_cs' : {} }

dates = ['2018-03-21', '2018-06-21', '2018-09-21', '2018-12-21']
for date in dates:
# Creates one day's worth of 60 min intervals
    times = pd.date_range(date, freq='60min', periods=24,
                          tz=site_location.tz)
    # Generate clearsky data using the Ineichen model, which is the default
    # The get_clearsky method returns a dataframe with values for GHI, DNI,
    # and DHI
    clearsky = site_location.get_clearsky(times)

    tilt = 30
    surface_azimuth = 180

    surface_irradiance = get_irradiance(site_location, tilt, surface_azimuth, clearsky, times)
    df = pd.DataFrame({'GHI': clearsky['ghi'], 'POA': surface_irradiance})
#   print(df)

    pv_day = pv_hourly[date+' 00:00:00' : date+' 23:00:00']
    # doing this to the whole thing gives to daylight saving time issues?
#   pv_day.index = pv_day.index.tz_localize('Europe/London')
    pv_day.index = pd.DatetimeIndex(data=pv_day.index.values,freq='60T',tz='Europe/London')
#   print(pv_day)
    # uses the DISC method to get DNI and then derives DHI from it (best)
    surface_ghi = ghi2irradiance(site_location, tilt, surface_azimuth, pv_day['glbl_irad_amt'])
    # just calculates the ration to the tilted surface (all bit high)
#   surface_ghi = ghi2poa(site_location, tilt, surface_azimuth, pv_day['glbl_irad_amt'])
    # uses the DIRINDEX method to get DNI and then derives DHI from it
    # December one goes a bit whacky here.
#   surface_ghi = ghi2poa2(site_location, tilt, surface_azimuth, pv_day['glbl_irad_amt'], clearsky)
#   print(surface_ghi)
    era5_day_ir = era5_ir[date+' 00:00:00' : date+' 23:00:00']
    era5_day_cs = era5_cs[date+' 00:00:00' : date+' 23:00:00']
    era5_day_ir.index.tz_localize('Europe/London')
    era5_day_cs.index.tz_localize('Europe/London')
#   print(era5_day_ir)

    df['GHI'].plot(label='Clear Sky GHI')
    df['POA'].plot(label='Clear Sky Plane Of Array')
    surface_ghi.plot(label='midas_ghi POA')
    pv_day['glbl_irad_amt'].plot(label='midas_ghi')
    era5_day_ir.plot(label='ERA5 GHI')
    era5_day_cs.plot(label='ERA5 Clear Sky GHI')
    plt.title('Solar Irradiance - Blackpool Squires Gate: ' + date)
    plt.xlabel('Time')
    plt.ylabel('Irradiance (W/m2)')
    plt.legend(loc='upper right')
    plt.show()

    total_ghi = df['GHI'].sum()
    print('Total GHI {}'.format(total_ghi) )
    totals['cs'][date] = df['GHI'].sum()
    totals['cs_poa'][date] = df['POA'].sum()
    totals['midas'][date] = pv_day['glbl_irad_amt'].sum()
    totals['midas_poa'][date] = surface_ghi.sum()
    totals['era'][date] = era5_day_ir.sum()
    totals['era_cs'][date] = era5_day_cs.sum()

print('Year cs    cs_poa midas midas_poa era era_cs')
for date in dates:
    print("{} {:4.1f} {:4.1f} {:4.1f} {:4.1f} {:4.1f} {:4.1f}".format(date,totals['cs'][date],totals['cs_poa'][date],totals['midas'][date],totals['midas_poa'][date],totals['era'][date],totals['era_cs'][date]) )
#   print("{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(year,totals['cs'][date],totals['cs_poa'][date],totals['midas'][date],totals['midas_poa'][date],totals['era'][date],totals['era_cs'][date]) )
