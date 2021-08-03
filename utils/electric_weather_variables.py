# augment weather with zenith, month, hour, doy etc.
# train ML model on 2018 weather to predict demand
# compare with actual 2017 and 2009 and synthetic addition of heat.

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib
import pytz
from sklearn.ensemble import RandomForestRegressor
import pvlib

# custom code
import stats
import readers
import stats

# augment weather variables with time dependent ones.

def augment(df):
    
    # public holidays indicator
    ph = ['2017-01-01', '2017-01-02', '2017-04-14', '2017-04-17', '2017-05-01', '2017-05-29', '2017-08-28', '2017-12-25', '2017-12-26', '2018-01-01', '2018-03-30', '2018-05-07', '2018-05-28', '2018-08-27', '2018-12-25', '2018-12-26', '2019-01-01', '2019-04-19', '2019-04-22', '2019-05-06', '2019-05-27', '2019-08-26', '2019-12-25', '2019-12-26', '2020-01-01', '2020-04-10', '2020-04-13', '2020-05-08', '2020-05-25', '2020-08-31', '2020-12-25', '2020-12-28' ]
    df['ph'] = 0
    for holiday in ph:
        df.loc[holiday+' 00:00:00' : holiday+' 23:30:00','ph'] = 1
    df['ph'] = df['ph'].astype(int)

    # day of the week
    df['wd'] = 0

   # day of year
    df['doy'] = 0
    # month of year
    df['month'] = 0
    # hour of the day
    df['hour'] = df.index.hour

    # solar zenith at population centre of GB in Leicestershire 
    # 2011 census. ( could work this out )
    lat = 52.68
    lon = 1.488

    site_location = pvlib.location.Location(lat, lon, tz=pytz.timezone('UTC'))
    solar_position = site_location.get_solarposition(times=df.index)
    df['zenith'] = solar_position['apparent_zenith']

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

        # day of week and day of year
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','wd'] = day.weekday()
        df['wd'] = df['wd'].astype(int)

        doy = day.timetuple().tm_yday
        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','doy'] = doy

        df.loc[day_str+' 00:00:00' : day_str+' 23:30:00','month'] = day.month
        df['month'] = df['month'].astype(int)

# feature identification using lasso - not working 
def lasso(input_df, output, plot=False):
    print('LASSO #################')
    X = normalize(input_df.values)
    y = normalize(output.values.reshape(-1, 1))
    print('normalized Y')
    print(y)
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
    print(input_df.index)
    print(output.index)
    coef = {}
    for column in input_df.columns:
        coef[column] = output.corr(input_df[column])
    return(coef)

# using sklearn Random Forest to predict the demand from weather

def forecast_demand(input_df, output, input_f):
    X_train = input_df
    y_train = output
    X_test = input_f
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
#   print(y_pred)
    # put the index of the year forecast on but may need to change this.
    return pd.Series(y_pred, index=output.index)

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2018.csv'
weather = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

augment(weather)

year = '2018'
# read historical electricity demand for reference year
# TODO - could we include the actual Scottish demand here?

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_ref = demand_ref['ENGLAND_WALES_DEMAND']
daily_electric_ref = electric_ref.resample('D').sum()

print(electric_ref)

# correlation of the various forecasting parameters with the demand
coeffs = correlation(weather, electric_ref, plot=True)
print('Hourly Feature correlation')
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

daily_weather = weather.resample('D').mean()
coeffs = correlation(daily_weather, daily_electric_ref, plot=True)
print('Daily Feature correlation')
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# lasso
# coeffs = lasso(weather, electric_ref, plot=True)
# print(coeffs)
# print('Daily Lasso')
# coeffs = lasso(daily_weather, daily_electric_ref, plot=True)
# print(coeffs)

# read historical electricity demand for 2017

year = '2017'
demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_2017 = demand_ref['ENGLAND_WALES_DEMAND']
# give it the same index as 2018 so we can plot
electric_2017.index = electric_ref.index
print(electric_2017)
daily_electric_2017 = electric_2017.resample('D').sum()

# read historical electricity demand for 2009

demand_filename = '/home/malcolm/uclan/data/electricity/demanddata_2009.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_2009 = demand_ref['ENGLAND_WALES_DEMAND']
# give it the same index as 2018 so we can plot
electric_2009.index = electric_ref.index
print(electric_2009)
daily_electric_2009 = electric_2009.resample('D').sum()

# 2017 weather variables
weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2018.csv'
weather17 = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
#weather17.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')
weather17.index = weather.index

augment(weather17)

# train ML to predict 2018 electric from weather
# ( add more variables till it predicts 2017 well )
# use it to predict what if we'd had 2017 weather. ( needs weather vars for 2017 )
# NOTES: only using dailytemp gives R=0.78 for daily, need hour for hourly 
#variables = ['dailytemp', 'hour', 'temp_dp']
variables = ['dailytemp', 'hour']
#variables = ['dailytemp']
#variables = weather.columns
predicted17 = forecast_demand(weather[variables], electric_ref, weather17[variables])
daily_predicted17 = predicted17.resample('D').sum()

# subtract 2018 heat demand from 2018 electric and add 2017 heat.
heat_that_is_electric_2018 = 0.06
heat_that_is_electric_2017 = 0.06

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather2018I-Bbdew_resistive.csv'
resistive_heat_2018 = readers.read_demand(demand_filename)
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2017/GBRef2018Weather2017I-Bbdew_resistive.csv'
resistive_heat_2017 = readers.read_demand(demand_filename)
resistive_heat_2017.index = resistive_heat_2018.index

# historic 2018 with the existing electric heat removed and 2017 added
mod_electric_ref = electric_ref - (resistive_heat_2018 * heat_that_is_electric_2018) + (resistive_heat_2017 * heat_that_is_electric_2017)
daily_mod_electric_ref = mod_electric_ref.resample('D').sum()

# compare it to the actual 2017 electric series (hourly and daily )
daily_electric_ref.plot(color='blue', label='Electricity demand time series 2018')
daily_electric_2017.plot(color='red', label='Electricity demand time series 2017')
daily_predicted17.plot(color='green', label='Random Forrest Prediction of 2017')
daily_mod_electric_ref.plot(color='yellow', label='Synthetic time series with 2017 weather')
daily_electric_2009.plot(color='orange', label='Electricity demand time series 2009')
plt.title('Daily electricty demand series from weather')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Mwh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# compare hourly series
print('Hourly Comparison to 2017 electricity demand')
stats.print_stats_header()
stats.print_stats(electric_ref, electric_2017, '2018 Electric', nvars=1, plot=True, xl='Electricity demand 2018', yl='Electricity demand 2017')
stats.print_stats(predicted17, electric_2017, '2017 ML', nvars=1, plot=True, xl='Electricity demand ML', yl='Electricity demand 2017')
stats.print_stats(mod_electric_ref, electric_2017, '2017 Heat only', nvars=1, plot=True, xl='Electricity demand synthetic', yl='Electricity demand 2017')

print('Hourly Storage Metrics')
print('2017 demand {}'.format(stats.esmv(electric_2017)) )
print('2018 demand {}'.format(stats.esmv(electric_ref)) )
print('2017 ML RF  {}'.format(stats.esmv(predicted17)) )
print('2017 heat   {}'.format(stats.esmv(mod_electric_ref)) )
print('2009 demand {}'.format(stats.esmv(electric_2009)) )

# compare daily series
daily_mod_electric_ref = mod_electric_ref.resample('D').sum()
print('Daily Comparison to 2017 electricity demand')
stats.print_stats_header()
stats.print_stats(daily_electric_ref, daily_electric_2017, '2018 Electric', nvars=1, plot=False, xl='Electricity demand 2018', yl='Electricity demand 2017')
stats.print_stats(daily_predicted17, daily_electric_2017, '2017 ML', nvars=1, plot=False, xl='Electricity demand ML', yl='Electricity demand 2017')
stats.print_stats(daily_mod_electric_ref, daily_electric_2017, '2017 Heat only', nvars=1, plot=False, xl='Electricity demand synthetic', yl='Electricity demand 2017')
stats.print_stats(daily_electric_2009, daily_electric_2017, '2009 Electric', nvars=1, plot=False, xl='Electricity demand 2009', yl='Electricity demand 2017')

# plot daily series
