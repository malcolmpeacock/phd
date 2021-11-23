# python script to merge all the data into one csv and add additional stuff
# like holidays and csc_ghi.
# also output the forecast file for next weeks weather.
#
# k          half hour period of the day  ( based on UTC)
# dsk        half hour period of the day  ( accounting for clocks changing)
# demand     demand data
# wd         day of the week
# week       week number with 1 as the newest
# season     0=winter, 1=spring, 2=summer, 3=autumn
# dtype      0,6=day of week, 7=ph, 8=Christmas, 9=December 26th
# month      month of the year, january=1,
# dailytemp  mean daily temperature of the day
# tempyd     mean daily temperature of the day before

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MaxAbsScaler
import matplotlib

# custom code
import utils

# feature identification using lasso
def lasso(input_df, output, plot=False):
#   print('LASSO #################')
    transformer = MaxAbsScaler().fit(input_df.values)
    X = transformer.transform(input_df.values)
    transformer = MaxAbsScaler().fit(output.values.reshape(-1, 1))
    y = transformer.transform(output.values.reshape(-1, 1))
#   print('normalized Y')
#   print(y)
    # defaults : tol=1e-4, max_iter=1000, 
#   reg = LassoCV(max_iter=8000, tol=1e-1)
    reg = LassoCV(max_iter=8000, tol=0.1)
    reg.fit(X, y)
#   print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
#   print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = input_df.columns)
    imp_coef = coef.sort_values()
#   print(imp_coef)
    if plot:
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind = "barh")
        plt.title("Feature importance using Lasso Model")
        plt.show()
    return coef

# feature correlation
def correlation(input_df, output, plot=False):
#   print(input_df.index)
#   print(output.index)
    coef = {}
    for column in input_df.columns:
        coef[column] = output.corr(input_df[column])
    return(coef)

# get series of k's
def index2ks(index):
    k = (index.hour * 2) + (index.minute / 30) + 1
    return k.astype(int)

# cube ( for wind power )
def add_cube(df, parameter='windspeed1'):
    df[parameter+'_cube'] = df[parameter].pow(3)
    

# lagged demand
def add_lags(df, nlags, parameter='demand'):
    count=0
    for lag in range(nlags):
        count+=1
        previous = df[parameter].copy().values[0:len(df)-count]
#       print(previous)
        zero = np.zeros(count)
#       print(zero)
        lag_values = np.concatenate([zero,previous])
        print('df {} prev {} zeros {} new {}'.format(len(df), len(previous), len(zero), len(lag_values)))
#       print(lag_values)
        df[parameter+'_lag'+str(count)] = lag_values

def add_variance(df, parameter):
    parms = []
    for p in range(5):
        parms.append(parameter + str(p+1))
    vmax = df[parms].max(axis=1)
    vmin = df[parms].min(axis=1)
    df[parameter+'_var'] = vmax - vmin

def augment(df):

    # trend
    df['trend'] = np.arange(len(df))
    # day of the week
#   df['wd'] = 0
    # day type
#   df['dtype'] = 0
    # season default to winter = 0
    df['season'] = 0
    # day of year
#   df['doy'] = 0
    # month of year
    df['month'] = 0
    # "day of year" ranges for the northern hemisphere
    spring = range(80, 172)
    summer = range(172, 264)
    autumn = range(264, 355)


    # heating degree hours
    df['hdh'] = (14.8 - df['temperature1']).clip(0.0)

    # daily mean temperature
    daily_temp = df['temperature1'].resample('D', axis=0).mean()

    # it doesn't matter what these are initialsed to as the weather starts
    # years before the demand and pv
    yesterday_temp = 0.0
    daybefore_temp = 0.0

    days = pd.Series(df.index.date).unique()
    # loop round each day ...
    ld_level = 0.0
    for day in days:
        day_str = day.strftime('%Y-%m-%d')

        # daily temperature
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','dailytemp'] = daily_temp.loc[day_str+' 00:00:00']
        # day before yesterdays daily temperature
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','tempdb'] = daybefore_temp
        daybefore_temp = yesterday_temp
        # yesterdays daily temperature
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','tempyd'] = yesterday_temp
        yesterday_temp = daily_temp.loc[day_str+' 00:00:00']

        # day of week and day of year
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','wd'] = day.weekday()
        doy = day.timetuple().tm_yday
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','doy'] = doy

        if doy in spring:
            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','season'] = 1
        if doy in summer:
            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','season'] = 2
        if doy in autumn:
            df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','season'] = 3
    
        dtype = day.weekday()
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','dtype'] = dtype

        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','month'] = day.month
            
    df['wd'] = df['wd'].astype(int)
    df['dtype'] = df['dtype'].astype(int)
    df['season'] = df['season'].astype(int)
    df['month'] = df['month'].astype(int)

    # half hour period
    df['k'] = index2ks(df.index)

    # dst indicator
    df['dsk'] = df['k'] + 2 * (df.index.hour - df.index.tz_localize('UTC').tz_convert(tz=pytz.timezone('Europe/London')).hour + 1)
    df.loc[df['dsk']==95, 'dsk'] = 47
    df.loc[df['dsk']==96, 'dsk'] = 48

    # lagged weather variables
    for p in range(5):
        add_lags(df, 1, 'solar_irradiance' + str(p+1)) 
        add_lags(df, 1, 'windspeed_east' + str(p+1)) 
        add_lags(df, 1, 'windspeed_north' + str(p+1))
        add_lags(df, 1, 'spec_humidity' + str(p+1)) 
        add_lags(df, 1, 'temperature' + str(p+1)) 

    # cyclic versions of the time variables
    df['s_k'] = np.sin(df['k'].values * (360 / 48 ) )
    df['c_k'] = np.cos(df['k'].values * (360 / 48 ) )

    # variance of weather variables
    add_variance(df, 'solar_irradiance') 
    add_variance(df, 'windspeed') 
    add_variance(df, 'temperature') 
    add_variance(df, 'spec_humidity') 
    # lagged variance
    add_lags(df, 1, 'solar_irradiance_var') 
    add_lags(df, 1, 'windspeed_var') 
    add_lags(df, 1, 'temperature_var') 
    add_lags(df, 1, 'spec_humidity_var') 

    # cube of windspeed
    add_cube(df, 'windspeed1') 

# main program

# process command line

parser = argparse.ArgumentParser(description='Merge and augment data.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--weather', action="store", dest="weather", help='Number of weather locations', default=5, type=int)

args = parser.parse_args()

# read in the data
input_dir = "/home/malcolm/uclan/challenge2/data/"
output_dir = "/home/malcolm/uclan/challenge2/output/"

# max/min ( what we are trying to predict )
maxmin_filename = 'MW_Staplegrove_CB905_MW_target_variable_half_hourly_max_min_real_power_MW_pre_august.csv'
maxmin = pd.read_csv(input_dir+maxmin_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

maxmin.columns = ['max_demand', 'min_demand']
print('Max and Min Demand')
print(maxmin)

# weather (hourly)
weathers=[]
for nw in range(args.weather):
    nw_str = str(nw+1)
    weather_filename = 'df_staplegrove_{}_hourly.csv'.format(nw_str)
    weather = pd.read_csv(input_dir+weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    print(weather)
    # up sample to 30 mins (with interpolation)
    # weather = weather.resample('30min').interpolate(method='cubic')
    weather = weather.resample('30min').interpolate()
    # add in a last 30 minutes row, because the weather upsample ended at the hour
    last_index = weather.last_valid_index() + pd.Timedelta(minutes=30)
    last_row = pd.DataFrame(weather[-1:].values, index=[last_index], columns=weather.columns)
    weather = weather.append(last_row)
    # combine the two windspeeds
    wind_squared = weather['windspeed_east'].pow(2) + weather['windspeed_north'].pow(2)
    weather['windspeed'] = wind_squared.pow(1/2)
    # add the station number onto the parameter name
    new_cols = []
    for col in weather.columns:
        new_cols.append(col+nw_str)
    weather.columns = new_cols
    weathers.append(weather)

# stick all the years together at the end
weather = pd.concat(weathers, axis=1 )
print(weather)

# demand data
demand_filename = 'MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_pre_august.csv'
demand = pd.read_csv(input_dir+demand_filename, header=0, sep=',', parse_dates=[0], index_col=0 )
demand.columns = ['demand']
add_lags(demand, 4)
print(demand)

# demand for the forecast period
demand_filename = 'MW_Staplegrove_CB905_MW_observation_variable_half_hourly_real_power_MW_august.csv'
demandf = pd.read_csv(input_dir+demand_filename, header=0, sep=',', parse_dates=[0], index_col=0)
demandf.columns = ['demand']
add_lags(demandf, 4)
print(demandf)

augment(weather)

# stick it all together
df = demand.join(weather, how='inner')
fdf = demandf.join(weather, how='inner')

print(df)
print(fdf)
print(df.columns)
utils.add_diffs(df, maxmin)

# plot weather
if args.plot:
    df['temperature1'].plot(label='temperature', color='red')
    plt.title('temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Temp (degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    df['solar_irradiance1'].plot(label='solar_irradiance1')
    df['solar_irradiance1_lag1'].plot(label='solar_irradiance_lag1')
    plt.title('Solar Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (W/m2)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

# plot demand
    df['demand'].plot(label='demand')
    maxmin['max_demand'].plot(label='max_demand')
    maxmin['min_demand'].plot(label='min_demand')
    plt.title('demand')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

lass_max = {}
lass_min = {}
lass_maxd = {}
lass_mind = {}
corr_max = {}
corr_min = {}
# lasso max demand
print('Lasso max_demand')
coeffs = lasso(df, maxmin['max_demand'], plot=args.plot)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
    lass_max[col] = value
# lasso min demand
print('Lasso min_demand')
coeffs = lasso(df, maxmin['min_demand'], plot=args.plot)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
    lass_min[col] = value

# lasso max demand diff
print('Lasso max_demand diff')
coeffs = lasso(df, maxmin['max_diff'], plot=args.plot)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
    lass_maxd[col] = value
# lasso min demand diff
print('Lasso min_demand diff')
coeffs = lasso(df, maxmin['min_diff'], plot=args.plot)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
    lass_mind[col] = value

# correlation max_demand
print('Correlation max_demand')
coeffs = correlation(df, maxmin['max_demand'], plot=args.plot)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
    corr_max[col] = value

# correlation max_demand
print('Correlation min_demand')
coeffs = correlation(df, maxmin['min_demand'], plot=args.plot)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
    corr_min[col] = value

#data = { 'corr_max' : pd.Series(corr_max), 'corr_min' : pd.Series(corr_min), 'lass_max': pd.Series(lass_max), 'lass_min' : pd.Series(lass_min) }
df_parms = pd.concat([pd.Series(corr_max), pd.Series(corr_min), pd.Series(lass_max), pd.Series(lass_min), pd.Series(lass_maxd), pd.Series(lass_mind) ], keys = ['corr_max', 'corr_min', 'lass_max', 'lass_min', 'lass_maxd', 'lass_mind'], axis=1)
print(df_parms)

# sanity check
print(df.columns)
for col in df.columns:
    if df[col].isna().sum() >0:
        print("ERROR nans in {}".format(col))
        print(df[df[col].isnull()])
        quit()


output_dir = "/home/malcolm/uclan/challenge2/output/"

# output maxmin data.
maxmin.index.name = 'time'
output_filename = 'maxmin_pre_august.csv'
maxmin.to_csv(output_dir+output_filename, float_format='%.8f')
# output merged data.
df.index.name = 'time'
output_filename = 'merged_pre_august.csv'
df.to_csv(output_dir+output_filename, float_format='%.8f')

# output weather forecast
fdf.index.name = 'time'
output_filename = 'merged_august.csv'
fdf.to_csv(output_dir+output_filename, float_format='%.8f')

# output correlation values
output_filename = 'correlation.csv'
df_parms.to_csv(output_dir+output_filename, float_format='%.2f')
