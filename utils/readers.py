# read various files of mine
#

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import stats

def clean(series):
    # replace zeros with NaN
    series = series.replace(0, float("NaN"))
    # replace missing values (NaN) by interpolation
    series = series.interpolate()
    return series

# read the electricity suppply from Elexon

def read_elexon_supply(filename):
    demand = pd.read_csv(filename, header=0, usecols=['date','period','WIND'], index_col=False )
    # each period is half an hour so multiply by 0.5
    demand.index = pd.DatetimeIndex(pd.to_timedelta((demand['period']-1) * 0.5, unit='H') + pd.to_datetime(demand['date'],format='%Y%m%d')).tz_localize('UTC')
    demand = demand.drop(['date', 'period'], axis=1)
    # multiply by 0.5 to convert Mw over a half hour period to Mwh
    demand = demand.resample('H').sum() * 0.5
    return demand

# read the electricity demand from national grid

def read_electric(filename):
    demand = pd.read_csv(filename, header=0, parse_dates=[0], usecols=['SETTLEMENT_DATE','SETTLEMENT_PERIOD','EMBEDDED_WIND_GENERATION', 'EMBEDDED_SOLAR_GENERATION', 'ENGLAND_WALES_DEMAND', 'EMBEDDED_WIND_CAPACITY', 'EMBEDDED_SOLAR_CAPACITY'] )
    # each period is half an hour so multiply by 0.5
    demand.index = pd.DatetimeIndex(pd.to_timedelta((demand['SETTLEMENT_PERIOD']-1) * 0.5, unit='H') + pd.to_datetime(demand['SETTLEMENT_DATE'])).tz_localize('UTC')
    demand = demand.drop(['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'], axis=1)
    # multiply by 0.5 to convert Mw over a half hour period to Mwh
    demand = demand.resample('D').sum() * 0.5
    return demand

def read_electric_hourly(filename):
    demand = pd.read_csv(filename, header=0, parse_dates=[0], usecols=['SETTLEMENT_DATE','SETTLEMENT_PERIOD','EMBEDDED_WIND_GENERATION', 'EMBEDDED_SOLAR_GENERATION', 'ENGLAND_WALES_DEMAND'] )
    # each period is half an hour so multiply by 0.5
    dt = pd.to_timedelta((demand['SETTLEMENT_PERIOD']-1) * 0.5, unit='H') + pd.to_datetime(demand['SETTLEMENT_DATE'])
    dt = pd.DatetimeIndex(dt)
    demand.index = dt.tz_localize('UTC')
    demand = demand.drop(['SETTLEMENT_DATE', 'SETTLEMENT_PERIOD'], axis=1)

    demand = demand.astype(float)
    # remove dupes due to daylight savings.
    demand = demand.drop_duplicates()
    # add in missing rows due to daylight savings.
    demand = demand.resample('30T').mean()
    demand = demand.interpolate()

    # multiply by 0.5 to convert Mw over a half hour period to Mwh
    demand = demand.resample('H').sum() * 0.5
    return demand

def read_demand(filename, parm='electricity'):
    demand = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, usecols=['time', parm], squeeze=True )
    return demand

def read_copheat(filename, parms=['electricity']):
    demand = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0, usecols=['time']+parms, squeeze=True )
    return demand

# Renewables ninja individual generation file

def read_ninja(filename):
    ninja = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'),index_col=0, squeeze=True, usecols=[0,2], comment='#')
    # create a datetime index so we can plot
    # (the round stops us getting a 1 in the minutes )
    ninja.index = pd.DatetimeIndex(pd.to_datetime(ninja.index).round('H'))
    return ninja

# Renewables ninja whole country generation file

def read_ninja_country(filename):
    ninja = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'),index_col=0, comment='"')
    # create a datetime index so we can plot
    # (the round stops us getting a 1 in the minutes )
    ninja.index = pd.DatetimeIndex(pd.to_datetime(ninja.index).round('H'))
    return ninja

# note we only have daily gas
def read_gas(filename):
    gas = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y').replace(tzinfo=pytz.timezone('Europe/London')), index_col=0, squeeze=True, usecols=[1,3] )
    gas = gas.astype('float')
    # reverse it (december was first! )
    gas = gas.iloc[::-1]
    # get rid of time so we just have a date
    gas.index = pd.DatetimeIndex(pd.to_datetime(gas.index).date)
    # take the average of multiple values at same date. really should take
    # the recently amended one.
    # gas = gas.resample('D').mean()
    gas = gas.resample('D').mean()
    # get smallest values to check for missing data
    print('GAS: SMALLEST')
    print(gas.nsmallest())
    # replace zeros with NaN
    gas = gas.replace(0.0, float("NaN"))
    # replace missing values (NaN) by interpolation
    gas = gas.interpolate()
    # print(gas['2016-07-28'])
    return gas

def dt_parse(y,m,d):
    dt = '{}-{}-{}T00:00:00Z'.format(y, m.zfill(2), d.zfill(2))
    return dt

def read_hadcet(filename):
    hadcet = pd.read_csv(filename, header=0, sep=',', parse_dates={'datetime': [0,1,2]}, date_parser=dt_parse, index_col='datetime', squeeze=True)
    # create a datetime index so we can plot
    hadcet.index = pd.DatetimeIndex(pd.to_datetime(hadcet.index).date)
    # convert to degrees (is in tenths)
    hadcet = hadcet / 10.0;
    return hadcet

def read_midas(filename):
    f = open(filename)
    while f.readline() != 'data\n':
        pass

    midas = pd.read_csv(f, header=0, sep=',', parse_dates=[0], index_col=0, usecols=['ob_end_time', 'max_air_temp','min_air_temp'], skipfooter=1)

    f.close()
    return midas

def read_midas_hourly(filename, parms=['air_temperature']):
    f = open(filename)
    while f.readline() != 'data\n':
        pass

    midas = pd.read_csv(f, header=0, sep=',', parse_dates=[0], index_col=0, usecols=['ob_time']+parms, skipfooter=1)

    f.close()
    return midas

def read_midas_irradiance(filename, parms=['glbl_irad_amt']):
    # open file and skip to start of data
    f = open(filename)
    while f.readline() != 'data\n':
        pass

#   midas = pd.read_csv(f, header=0, sep=',', parse_dates=[0], index_col=0, usecols=['ob_end_time','id_type']+parms, skipfooter=1)
    midas = pd.read_csv(f, header=0, sep=',', parse_dates=[0], index_col=0, usecols=['ob_end_time']+parms, skipfooter=1)
    # only include records with DCNN
#   midas = midas[midas['id_type'] == 'DCNN']
    # rename the index
    midas.index.rename('time', inplace=True)
    # remove entries at 23:59 which are bad data
    # eg sutton bonnington 2018-01-01
    midas = midas[midas.index.minute != 59]
    f.close()
    return midas

def read_loctemp(filename):
#   temp = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d').replace(tzinfo=pytz.timezone('Europe/London')), index_col=0)
    temp = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0)
    temp = temp.astype('float')
    return temp

#   Columns of interest:
#    0  ob_time 
#    1  latitude
#    2  longitude
#    5  version (we need 1)
#   13  wind speed (knots)
#   25  air temperature degree c ( to nearest 0.1 )
#   28  sea temperature degree c ( to nearest 0.1 )
#   40  measured wave period. ( 0.1 sec )
#   41  measured wave height. ( 0.1 m )
#   42  ship direction, 0=stopped

def read_midas_marine(filename):
#   midas = pd.read_csv(filename, sep=',', parse_dates=[0], index_col=0, usecols=[0,1,2,5,13,25,28,40,41,42], dtype=np.float32, header=None ).fillna(0
    midas = pd.read_csv(filename, sep=',', parse_dates=[0], index_col=0, usecols=[0,1,2,5,13,25,28,40,41,42], header=None )
#   midas = pd.read_csv(filename, sep=',', parse_dates=[0], index_col=0, usecols=[0,1,2,5,13,25,28,40,41,42], dtype=np.float32, header=None )
#   midas = pd.read_csv(filename, sep=',', parse_dates=[0], index_col=0, usecols=[0,1,2,5,13,42], header=None, dtype={13 : np.int32})
#   midas = pd.read_csv(filename, sep=',', parse_dates=[0], index_col=0, usecols=[0,1,2,5,13,42], header=None, dtype={13 : 'int32'}, na_values = [' '])
    midas.columns = ['latitude', 'longitude', 'version', 'windspeed', 'air_temperature', 'sea_temperature','msr_wave_per', 'msr_wave_height','shipdir']
    midas.index.rename('time', inplace=True)
    # filter to get only version=1 which is met office quality checked
    midas = midas[midas['version'] == 1]
    # change type of windspeed to float
    midas['windspeed'] = pd.to_numeric(midas['windspeed'], errors='coerce').astype(float)
    midas['msr_wave_per'] = pd.to_numeric(midas['msr_wave_per'], errors='coerce').astype(float)
    midas['msr_wave_height'] = pd.to_numeric(midas['msr_wave_height'], errors='coerce').astype(float)
    midas['air_temperature'] = pd.to_numeric(midas['air_temperature'], errors='coerce').astype(float)
    # only get records where we have a windspeed
    notnull = pd.notnull(midas['windspeed'])
    midas = midas[notnull]
    print(midas.info())
    return midas

def read_crown(filename):
#   crown = pd.read_csv(filename, sep=' ', comment='*', parse_dates=[0,1], index_col=0, header=0 )
#   crown = pd.read_csv(filename, sep=',', header=0 )
    crown = pd.read_csv(filename, sep=',', header=0, parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M'),index_col=0)
    # hourly mean windspeed, since we have 10 minutes data
    crown = crown.resample('H').mean()
    return crown

def read_clnr(filename):
    clnr = pd.read_csv(filename, sep=',', header=0, parse_dates=[2], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S'))
    print(clnr)
    return clnr

    

def rhpp_dt_parse(y,m,d,H,M):
    dt = '{}-{}-{} {}:{}:00'.format(y, m.zfill(2), d.zfill(2), H.zfill(2), M.zfill(2))
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")

def read_rhpp(filename, location):
    rhpp = pd.read_csv(filename, header=0, sep=',', parse_dates={'datetime': [1,2,3,4,5]}, date_parser=rhpp_dt_parse, index_col='datetime')
#   print(rhpp.columns)
    rhpp = rhpp.drop(['Matlab_time'], axis=1)
    # create a datetime index so we can plot
    rhpp.index = pd.DatetimeIndex(pd.to_datetime(rhpp.index))
    # do some analysis
    start = rhpp.index[0]
    end = rhpp.index[-10]
    heat_nans = rhpp['Hhp'].isna().sum()
    elec_nans = rhpp['Ehp'].isna().sum()
    tin_nans = rhpp['Tin'].isna().sum()
    tsf_nans = rhpp['Tsf'].isna().sum()
#   print('Start {} End {} Nans Hhp {} Ehp {} Tin {} Tsf {}'.format(start, end, heat_nans, elec_nans, tin_nans, tsf_nans) )
    analysis = { 'start' : start, 'end': end, 'nans_Hhp': heat_nans, 'nans_Ehp': elec_nans, 'nans_tin': tin_nans, 'nans_tsf': tsf_nans, 'location' : location }
    rhpp = rhpp.resample('H').mean()
    
    return rhpp, analysis

def read_advm(filename, location):
    advm = pd.read_csv(filename, sep=',', header=0, parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y'), index_col=0, error_bad_lines=False, )
    # merge all the half hourly columns into rows
    advm = advm.melt(ignore_index=False)
    # combine the date and time columns into a new index
    advm.index = pd.DatetimeIndex(pd.to_datetime(advm.index.strftime("%Y-%m-%d") + ' ' + advm['variable']))
    # sort by the new index
    advm = advm.sort_index()
    # drop the old time column
    advm = advm.drop(['variable'], axis=1)
    nans = advm['value'].isna().sum()
    advm = advm.interpolate()
    advm = advm.fillna(method='bfill')
    advm = advm.fillna(method='ffill')
    fnans = advm['value'].isna().sum()
    analysis = { 'nans' : nans, 'location' : location, 'fixed' : fnans }
    return advm, analysis
