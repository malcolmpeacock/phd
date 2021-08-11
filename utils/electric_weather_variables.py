# augment weather with zenith, month, hour, doy etc.
# train ML model on 2018 weather to predict demand
# compare with actual 2017 and 2009 and synthetic addition of heat.
# 

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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MaxAbsScaler

# custom code
import stats
import readers
import stats

def regression(X, y):
    estimator = LinearRegression()
#   model = make_pipeline(PolynomialFeatures(1),estimator)
#   fit = model.fit(X, y)
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
    print(fit.score(pf,y))
    print(estimator.coef_)
    print(estimator.intercept_)
#   p = fit.predict(Xp)
    return coeffs[1], coeffs[2]

# split cooling energy by CDD

def cooling(temperature, cooling_energy):
    print(cooling_energy)
    print(temperature)
    cdd = (temperature - 15.0).clip(0.0)
    print(cdd)
    cooling = cdd * ( cooling_energy / cdd.sum() )
    print('Cooling Degree Hours {}'.format(cdd.sum() ) )
    print(cooling)
    return cooling

# augment weather variables with time dependent ones.

def augment(df):
    # convert temperature to degrees C from kelvin
    df['temp'] = df['temp'] - 273.15
    
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

    # cooling degree hours
    df['cdh'] = (df['temp'] - 22.0).clip(0.0)
    # heating degree hours
    df['hdh'] = (14.8 - df['temp']).clip(0.0)
    print(df[['cdh', 'hdh']])

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

# feature identification using lasso
def lasso(input_df, output, plot=False):
    print('LASSO #################')
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

def get_demand(year, espini):
    if espini:
        demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
        demand = readers.read_espeni(demand_filename, year)
        electric = demand / 1000000.0
    else:
        # TODO - could we include the actual Scottish demand here?
        scotland_factor = 1.1
        # read historical electricity demand for reference year
        if year=='2009':
            demand_filename = '/home/malcolm/uclan/data/electricity/demanddata_2009.csv'
        else:
            demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'
        demand_ref = readers.read_electric_hourly(demand_filename)
        # Convert to TWh
        electric = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor / 1000000.0
    return electric

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2018.csv'
weather18 = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather18.index = pd.DatetimeIndex(weather18.index).tz_localize('UTC')

augment(weather18)
#print(weather['hour'])
#print(weather['hour'].isna().sum())

espini = True
electric_2018 = get_demand('2018', espini)
print(electric_2018)
# electric_2018 = demand_ref['ENGLAND_WALES_DEMAND']
daily_electric_2018 = electric_2018.resample('D').sum()

print(electric_2018)
print('Historic demand 2018: max {} min {} total {} '.format(electric_2018.max(), electric_2018.min(), electric_2018.sum() ) )

# correlation of the various forecasting parameters with the demand
coeffs = correlation(weather18, electric_2018, plot=True)
print('Hourly Feature correlation')
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

daily_weather18 = weather18.resample('D').mean()
coeffs = correlation(daily_weather18, daily_electric_2018, plot=True)
print('Daily Feature correlation')
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# lasso
print('Hourly Lasso')
coeffs = lasso(weather18, electric_2018, plot=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
print('Daily Lasso')
coeffs = lasso(daily_weather18, daily_electric_2018, plot=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# read historical electricity demand for 2017

electric_2017 = get_demand('2017', espini)
# electric_2017 = demand_ref['ENGLAND_WALES_DEMAND']
# give it the same index as 2018 so we can plot
electric_2017.index = electric_2018.index
print(electric_2017)
daily_electric_2017 = electric_2017.resample('D').sum()

# read historical electricity demand for 2009

electric_2009 = get_demand('2009', espini)
# give it the same index as 2018 so we can plot
electric_2009.index = electric_2018.index
print(electric_2009)
daily_electric_2009 = electric_2009.resample('D').sum()

# 2017 weather variables
weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2017.csv'
weather17 = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
#weather17.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')
weather17.index = weather18.index

augment(weather17)

# 2009 weather variables
weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2009.csv'
weather09 = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0) 
weather09.index = weather18.index

augment(weather09)

# train ML to predict 2018 electric from weather
# ( add more variables till it predicts 2017 well )
# use it to predict what if we'd had 2017 weather. ( needs weather vars for 2017 )
# NOTES: only using dailytemp gives R=0.78 for daily, need hour for hourly 
#variables = ['dailytemp', 'hour', 'temp_dp']
variables = ['dailytemp', 'hour']
#variables = ['dailytemp']
#variables = weather.columns
predicted17 = forecast_demand(weather18[variables], electric_2018, weather17[variables])
daily_predicted17 = predicted17.resample('D').sum()

# read resistive heat for 2018 and 2017

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather2018I-Bbdew_resistive.csv'
heat_demand2018 = readers.read_copheat(demand_filename, ['electricity', 'temperature'])
resistive_heat_2018 = heat_demand2018['electricity'] * 1e-6
temperature_2018 = heat_demand2018['temperature']
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2017/GBRef2018Weather2017I-Bbdew_resistive.csv'
resistive_heat_2017 = readers.read_demand(demand_filename) * 1e-6
resistive_heat_2017.index = resistive_heat_2018.index

# subtract 2018 heat demand from 2018 electric and add 2017 heat.
#heat_that_is_electric_2018 = 0.06
#heat_that_is_electric_2017 = 0.06
heat_that_is_electric_2018 = electric_2018.sum() * 0.14 / resistive_heat_2018.sum()
heat_that_is_electric_2017 = electric_2017.sum() * 0.14 / resistive_heat_2017.sum()
print('Heat electric 2018 {} 2017 {}'.format(heat_that_is_electric_2018, heat_that_is_electric_2017) )

# get the portion of heat the is currently electric
electric2018_heat = resistive_heat_2018 * heat_that_is_electric_2018
electric2018_no_heat = electric_2018 - electric2018_heat


# weekly plots
weekly_electric_2018 = electric_2018.resample('W').sum()
weekly_heat_demand = resistive_heat_2018.resample('W').sum()
weekly_electric2018_heat = electric2018_heat.resample('W').sum()
weekly_electric2018_no_heat = electric2018_no_heat.resample('W').sum()

weekly_electric_2018.plot(color='blue', label='Weekly Electricity demand 2018')
#weekly_heat_demand.plot(color='yellow', label='Weekly Electric heat 2018 all resistive')
weekly_electric2018_heat.plot(color='red', label='Weekly Electric heat 2018')
weekly_electric2018_no_heat.plot(color='green', label='Weekly Electricity demand 2018 minus heat')
plt.title('Weekly electricty demand series')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per week', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# plot 2018 electric, heat and difference. daily
daily_electric2018_heat = electric2018_heat.resample('D').sum()
daily_electric2018_no_heat = electric2018_no_heat.resample('D').sum()

daily_electric_2018.plot(color='blue', label='Electricity demand time series 2018')
daily_electric2018_heat.plot(color='red', label='Existing Electric heat 2018')
daily_electric2018_no_heat.plot(color='green', label='Electricity 2018 with heat removed')
plt.title('Removing existing heat from daily electricty demand series')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# historic 2018 with the existing electric heat removed and 2017 added
mod_electric_ref = electric_2018 - (resistive_heat_2018 * heat_that_is_electric_2018) + (resistive_heat_2017 * heat_that_is_electric_2017)
daily_mod_electric_ref = mod_electric_ref.resample('D').sum()

# compare it to the actual 2017 electric series (hourly and daily )
daily_electric_2018.plot(color='blue', label='Electricity demand time series 2018')
daily_electric_2017.plot(color='red', label='Electricity demand time series 2017')
daily_predicted17.plot(color='green', label='Random Forrest Prediction of 2017')
daily_mod_electric_ref.plot(color='yellow', label='Synthetic time series with 2017 weather')
#daily_electric_2009.plot(color='orange', label='Electricity demand time series 2009')
plt.title('Daily electricty demand series from weather')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Mwh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# compare hourly series
print('Hourly Comparison to 2017 electricity demand')
stats.print_stats_header()
stats.print_stats(electric_2018, electric_2017, '2018 Electric', nvars=1, plot=True, xl='Electricity demand 2018', yl='Electricity demand 2017')
stats.print_stats(predicted17, electric_2017, '2017 ML', nvars=1, plot=True, xl='Electricity demand ML', yl='Electricity demand 2017')
stats.print_stats(mod_electric_ref, electric_2017, '2017 Heat only', nvars=1, plot=True, xl='Electricity demand synthetic', yl='Electricity demand 2017')

print('Hourly Storage Metrics')
print('2017 demand {}'.format(stats.esmv(electric_2017)) )
print('2018 demand {}'.format(stats.esmv(electric_2018)) )
print('2017 ML RF  {}'.format(stats.esmv(predicted17)) )
print('2017 heat   {}'.format(stats.esmv(mod_electric_ref)) )
print('2009 demand {}'.format(stats.esmv(electric_2009)) )

# compare daily series
daily_mod_electric_ref = mod_electric_ref.resample('D').sum()
print('Daily Comparison to 2017 electricity demand')
stats.print_stats_header()
stats.print_stats(daily_electric_2018, daily_electric_2017, '2018 Electric', nvars=1, plot=False, xl='Electricity demand 2018', yl='Electricity demand 2017')
stats.print_stats(daily_predicted17, daily_electric_2017, '2017 ML', nvars=1, plot=False, xl='Electricity demand ML', yl='Electricity demand 2017')
stats.print_stats(daily_mod_electric_ref, daily_electric_2017, '2017 Heat only', nvars=1, plot=False, xl='Electricity demand synthetic', yl='Electricity demand 2017')
stats.print_stats(daily_electric_2009, daily_electric_2017, '2009 Electric', nvars=1, plot=False, xl='Electricity demand 2009', yl='Electricity demand 2017')

# plot daily series

# correlation of the various forecasting parameters with the demand
coeffs = correlation(weather18, electric2018_no_heat, plot=False)
print('Hourly Feature correlation for electric with no heat')
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

print('Hourly Lasso for electric with no heat')
coeffs = lasso(weather18, electric2018_no_heat, plot=False)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# calculate 2018 cooling TWh
#  2018 enery for cooling and ventillation is 12.4 TWh but what it the split?
#  2050 Pathway says in 2007 cooling was 9TWh
#cooling_energy_2018 = 12.4
cooling_energy_2018 = 9
cooling2018 = cooling(temperature_2018, cooling_energy_2018)
electric2018_no_heat_or_cooling = electric2018_no_heat - cooling2018

# weekly cooling
weekly_cooling2018 = cooling2018.resample('W').sum()
weekly_electric2018_no_heat_or_cooling = electric2018_no_heat_or_cooling.resample('W').sum()

weekly_electric_2018.plot(color='blue', label='Weekly Electricity demand 2018')
weekly_cooling2018.plot(color='red', label='Weekly Electric cooling 2018')
weekly_electric2018_no_heat.plot(color='green', label='Weekly Electricity demand 2018 minus heat')
weekly_electric2018_no_heat_or_cooling.plot(color='yellow', label='Weekly Electricity demand 2018 minus heat and cooling')
plt.title('Weekly electricty demand series')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per week', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# regression on hdh and cdh
c1, c2 = regression(weather18[['hdh', 'cdh']].values, electric_2018.values)
base_heating  = weather18 ['hdh'] * c1
base_cooling  = weather18 ['cdh'] * c2
base18 = electric_2018 - base_heating - base_cooling
weekly_base = base18.resample('W').sum()
weekly_electric_2018.plot(color='blue', label='Weekly Electricity demand 2018')
weekly_base.plot(color='green', label='Weekly Electricity base 2018')
plt.title('Weekly electricty demand base using HDH and CDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per week', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# heating and cooling energy ?
print('Total 2018 {} Heating {} Cooling {}'.format(electric_2018.sum(), base_heating.sum(), base_cooling.sum() ) )


# regression on hdh and cdh for 2017 and 2009
c1, c2 = regression(weather17[['hdh', 'cdh']].values, electric_2017.values)
base_heating  = weather17 ['hdh'] * c1
base_cooling  = weather17 ['cdh'] * c2
base17 = electric_2017 - base_heating - base_cooling
print('Total 2017 {} Heating {} Cooling {}'.format(electric_2017.sum(), base_heating.sum(), base_cooling.sum() ) )

c1, c2 = regression(weather09[['hdh', 'cdh']].values, electric_2009.values)
base_heating  = weather09 ['hdh'] * c1
base_cooling  = weather09 ['cdh'] * c2
base09 = electric_2009 - base_heating - base_cooling
print('Total 2009 {} Heating {} Cooling {}'.format(electric_2009.sum(), base_heating.sum(), base_cooling.sum() ) )

daily_base18 = base18.resample('D').sum()
daily_base17 = base17.resample('D').sum()
daily_base09 = base09.resample('D').sum()

daily_base18.plot(color='blue', label='Base electricity demand 2018')
daily_base17.plot(color='red', label='Base electricity demand 2017')
daily_base09.plot(color='green', label='Base electricity demand 2009')
plt.title('Daily electricty demand base using HDH and CDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# Random Forrest 3 different years of electricity demand with same weather 

rfbase18 = forecast_demand(weather18[variables], electric_2018, weather18[variables])
daily_rfbase18 = rfbase18.resample('D').sum()
rfbase17 = forecast_demand(weather17[variables], electric_2017, weather18[variables])
daily_rfbase17 = rfbase17.resample('D').sum()
rfbase09 = forecast_demand(weather09[variables], electric_2009, weather18[variables])
daily_rfbase09 = rfbase09.resample('D').sum()

daily_rfbase18.plot(color='blue', label='Random Forrest 2018 with 2018 weather')
daily_rfbase17.plot(color='green', label='Random Forrest 2017 with 2018 weather')
daily_rfbase09.plot(color='red', label='Random Forrest 2009 with 2018 weather')
plt.title('Different daily electricty demands with 2018 weather')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

stats.print_stats_header()
stats.print_stats(daily_rfbase17, daily_rfbase18, '2017 Electric', nvars=1, plot=False, xl='Electricity demand 2018', yl='Electricity demand 2017')
stats.print_stats(daily_rfbase09, daily_rfbase18, '2009 Electric', nvars=1, plot=False, xl='Electricity demand 2018', yl='Electricity demand 2009')

# subtract bases to remove the weather and see how the patterns compare

sub17 = daily_rfbase18 - daily_rfbase17
sub09 = daily_rfbase18 - daily_rfbase09

sub17.plot(color='blue', label='2017 subtracted from 2018 with 2018 weather')
sub09.plot(color='green', label='2009 subtracted frm 2018 with 2018 weather')
plt.title('Daily Time series for 2018 weather subtracted to find base pattern')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

sub17_weekly = sub17.resample('W').sum()
sub09_weekly = sub09.resample('W').sum()
sub17_weekly.plot(color='blue', label='2017 subtracted from 2018 with 2018 weather')
sub09_weekly.plot(color='green', label='2009 subtracted frm 2018 with 2018 weather')
plt.title('Weekly Time series for 2018 weather subtracted to find base pattern')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

print('Hourly Feature correlation for 2009')
coeffs = correlation(weather09, electric_2009, plot=False)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# lasso
print('Hourly Lasso for 2009')
coeffs = lasso(weather09, electric_2009, plot=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
