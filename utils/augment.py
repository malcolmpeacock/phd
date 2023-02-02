# augment weather variables with time dependent ones.
import pvlib
import pytz
import pandas as pd
import numpy as np

def augment(df, timed=True):
    # convert temperature to degrees C from kelvin
    df['temp'] = df['temp'] - 273.15

    if timed:
        # public holidays indicator 2017 - 2020
        ph = ['2017-01-01', '2017-01-02', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-29', '2017-08-28', '2017-12-25', '2017-12-26', '2018-01-01', '2018-03-30', '2018-05-07', '2018-05-28', '2018-08-27', '2018-12-25', '2018-12-26', '2019-01-01', '2019-04-19', '2019-04-22', '2019-05-06', '2019-05-27', '2019-08-26', '2019-12-25', '2019-12-26', '2020-01-01', '2020-04-10', '2020-04-13', '2020-05-08', '2020-05-25', '2020-08-31', '2020-12-25', '2020-12-28' ]
        df['ph'] = 0
        for holiday in ph:
            df.loc[holiday+' 00:00:00' : holiday+' 23:30:00','ph'] = 1
        df['ph'] = df['ph'].astype(int)

        # day of the week
        df['wd'] = 0
        # binary dow flag
        for d in range(7):
            d_key = 'd{}'.format(d)
            df[d_key] = 0
        # day of year
        df['doy'] = 0
        # month of year
        df['month'] = 0
        # hour of the day
        df['hour'] = df.index.hour
        # weekend
        df['weekend'] = 0

    # cooling degree hours ( 22 is Bloomfield et. al. )
    df['cdh'] = (df['temp'] - 22.0).clip(0.0)
    # heating degree hours ( Bloomfield et. al. use 15.5 )
    df['hdh'] = (15.5 - df['temp']).clip(0.0)

    # solar zenith at population centre of GB in Leicestershire 
    # 2011 census. ( could work this out )
    lat = 52.68
    lon = 1.488

    site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('UTC'))
    solar_position = site_location.get_solarposition(times=df.index)
    df['zenith'] = solar_position['apparent_zenith']
    # weather dependent ghi
    df['ghi_w'] = df['clear_sky'] - df['ghi']
    df['ghi_w2'] = df['ghi_w'] **2
    df['ghi_w3'] = df['ghi_w'] **3
    # cooling power of the wind ( National Grid )
    df['cp'] = np.sqrt(df['wind']) * (18.3 - df['temp'] ).clip(0.0)

    days = pd.Series(df.index.date).unique()
    # loop round each day ...

    yesterday_temp = 0.0
    daybefore_temp = 0.0

    for day in days:
        day_str = day.strftime('%Y-%m-%d')

        # daily temperature
        daily_temp = df['temp'].resample('D', axis=0).mean()
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','dailytemp'] = daily_temp.loc[day_str+' 00:00:00']

        # day before yesterdays daily temperature
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','tempdb'] = daybefore_temp
        daybefore_temp = yesterday_temp

        # yesterdays daily temperature
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','tempyd'] = yesterday_temp
        yesterday_temp = daily_temp.loc[day_str+' 00:00:00']
    
        if timed:
            # day of week
            wd = day.weekday()
            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','wd'] = wd
            df['wd'] = df['wd'].astype(int)
            # binary day flag
            d_key = 'd{}'.format(wd)
            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00',d_key] = 1

            # day of year
            doy = day.timetuple().tm_yday
            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','doy'] = doy

            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','month'] = day.month
            df['month'] = df['month'].astype(int)

            # weekend
            if wd in [5,6]:
                df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','weekend'] = 1

