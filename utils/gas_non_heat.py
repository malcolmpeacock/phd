# python script to investigate idenfitying the non-heat gas algorithmically.
# IDEAS
# g(t) = cwv(t) * ( annual_gas_heat / 365.0 ) + n(t)
#  where cwt is the normalised composite weather variable
#
# then look at n(t) dependence on temperature and other variables.
# n(t) = a + b*T + c*T-sqaured
# calculate a, b, c by training a regression model.
#
# ISSUES: cwv only available for each LDZ
# so just using hdh

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

# custom code
import stats
import readers
import augment

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


def get_weather(year):
    weather_file = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(year)
    weather = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0)
    weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

    augment.augment(weather)
    daily_weather = weather.resample('D').mean()
#   daily_weather.to_csv('/home/malcolm/uclan/output/temp/augment.csv', float_format='%g');
    return daily_weather

def regression(X, y):
    estimator = LinearRegression()
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
#   print(fit.score(pf,y))
#   print(estimator.coef_)
#   print(estimator.intercept_)
#   p = fit.predict(Xp)
    # return intercept as coeff[0]
    coeffs[0] = estimator.intercept_
    return coeffs

# feature identification using lasso - not working 
def lasso(input_df, output, plot=False, normalize=False):
    print('LASSO #################')
    if normalize:
        transformer = MaxAbsScaler().fit(input_df.values)
        X = transformer.transform(input_df.values)
        transformer = MaxAbsScaler().fit(output.values.reshape(-1, 1))
        y = transformer.transform(output.values.reshape(-1, 1))
    else:
        X = input_df.values
        y = output.values.reshape(-1, 1)
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

# main program
weather_year = '2018'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'

gas = readers.read_gas(gas_filename)
# print(gas)
gas.index = pd.DatetimeIndex(gas.index).tz_localize('UTC')
# TWh
gas = gas * 1e-9

# other years
weather_year = '2017'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'
gas17 = readers.read_gas(gas_filename)
gas17.index = pd.DatetimeIndex(gas17.index).tz_localize('UTC')
# TWh
gas17 = gas17 * 1e-9

weather_year = '2019'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'
gas19 = readers.read_gas(gas_filename)
gas19.index = pd.DatetimeIndex(gas19.index).tz_localize('UTC')
# TWh
gas19 = gas19 * 1e-9

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2018.csv'
weather = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0)
weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

augment.augment(weather)
daily_weather = weather.resample('D').mean()
# heating degree days
daily_weather['hdd'] = (14.8 - daily_weather['temp']).clip(0.0)
# other years
daily_weather17 = get_weather('2017')
daily_weather17['hdd'] = (14.8 - daily_weather17['temp']).clip(0.0)
daily_weather19 = get_weather('2019')
daily_weather19['hdd'] = (14.8 - daily_weather19['temp']).clip(0.0)

coeffs = correlation(daily_weather, gas, plot=True)
print('Daily Feature correlation 2018')
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# lasso
print('Daily Lasso 2018')
coeffs = lasso(daily_weather, gas, plot=True, normalize=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))


# regression on mean hdh
#hdd = daily_weather['hdh'] / 24.0
hdd = daily_weather['hdd']
coeffs = regression(hdd.values.reshape(-1,1), gas.values)
c0 = coeffs[0]
c1 = coeffs[1]
print('2018 regression coeffs hdh')
print(coeffs)

# plot regression with hdd
#hdh = daily_weather['hdh']
y = c0 + hdd * c1
ax = plt.scatter(hdd.values, gas.values, color='blue', label='Daily gas demand 2018')
plt.plot(hdd, y, color='red')
plt.title('Relationship between daily gas consumption and HDD')
plt.xlabel('Heat Degree Days (degrees C)', fontsize=15)
plt.ylabel('Gas Demand (Twh) per day', fontsize=15)
#plt.legend(loc='upper left')
legend_elements = [Line2D([0], [0], color='red', label='Ordinary Least Squares Regression (Rd)'),
                   Line2D([0], [0], marker='o', color='blue', label='Daily gas demand 2018 (Gd)') ]
plt.legend(loc='upper left', handles=legend_elements)
plt.show()
# forecast of gas time series from hdh
#gas_hdh  = daily_weather ['hdh'] * c1
gas_hdh  = hdd * c1
base = gas - gas_hdh
# plot regression with hdh
gas.plot(color='blue', label='Daily total gas demand 2018')
base.plot(color='green', label='Daily Gas demand with heating removed')
plt.title('Daily gas demand 2018 base using HDD')
plt.xlabel('day', fontsize=15)
plt.ylabel('Gas Demand (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()


print('2018 HDH gives non heat gas {} heat gas {} total {}'.format(base.sum(), gas_hdh.sum(), gas.sum() ) )

gas_hdh  = daily_weather ['hdh'] * c1 + c0

# correlation after heat removed

coeffs = correlation(daily_weather, base, plot=True)
print('2018 Daily Feature correlation base')
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
print('2018 Daily Lasso after heat removed')
coeffs = lasso(daily_weather, base, plot=True, normalize=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

total_gas = gas.sum()
#total_heat = base_heating.sum()
#print('Total {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

# regression on mean hdh and windspeed
coeffs = regression(daily_weather[['hdh', 'wind']], gas.values)
print('hdh and wind', coeffs)
c1 = coeffs[1]
c2 = coeffs[2]
base_heating  = daily_weather ['hdh'] * c1 + daily_weather ['wind'] * c2
base2 = gas - base_heating
# plot regression with hdh
gas.plot(color='blue', label='Daily gas demand 2018')
base2.plot(color='green', label='Gas demand with heating removed using wind and hdh')
plt.title('Daily gas demand 2018 base using mean HDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()
total_gas = gas.sum()
total_heat = base_heating.sum()
print('2018 Total (hdh and wind) {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

gas_hdh_wind = base_heating + coeffs[0]

# other years hdh
print('daily_weather17 {} gas17 {}'.format(len(daily_weather17), len(gas17)))
coeffs = regression(daily_weather17[['hdh', 'wind']], gas17.values)
gas17_hdh_wind = daily_weather17['hdh'] * coeffs[1] + daily_weather17['wind'] * coeffs[2] + coeffs[0]
print('daily_weather19 {} gas19 {}'.format(len(daily_weather19), len(gas19)))
coeffs = regression(daily_weather19[['hdh', 'wind']], gas19.values)
gas19_hdh_wind = daily_weather19['hdh'] * coeffs[1] + daily_weather19['wind'] * coeffs[2] + coeffs[0]

# regression on mean hdh and windspeed and ghi
coeffs = regression(daily_weather[['hdh', 'ghi', 'wind']], gas.values)
print(coeffs)
c1 = coeffs[1]
c2 = coeffs[2]
c3 = coeffs[3]
base_heating  = daily_weather ['hdh'] * c1 + daily_weather ['ghi'] *c2 + daily_weather ['wind'] * c3
base3 = gas - base_heating
# plot regression with hdh
gas.plot(color='blue', label='Daily gas demand 2018')
base3.plot(color='green', label='Gas demand with heating removed using wind and ghi')
plt.title('Daily gas demand 2018 base using mean HDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()
total_gas = gas.sum()
total_heat = base_heating.sum()
print('Total hdh, wind and ghi {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

gas_hdh_wind_ghi = base_heating + coeffs[0]

# regression on lasso coefficients.
variables = ['hdh', 'dailytemp', 'cdh', 'thermal', 'wind']
coeffs = regression(daily_weather[variables], gas.values)
print('lasso', coeffs)
base_heating = daily_weather['hdh'] * 0.0
for i in range(len(variables)):
    base_heating = base_heating + daily_weather[variables[i]] * coeffs[i+1]
#   base_heating = daily_weather['hdh'] * coeffs[1] + daily_weather['dailytemp'] * coeffs[2] + daily_weather['cdh'] * coeffs[3]  + daily_weather['thermal'] * coeffs[4] + daily_weather['wind'] * coeffs[5]
gas_lasso = base_heating + coeffs[0]
base4 = gas - base_heating
total_gas = gas.sum()
total_heat = gas_lasso.sum()
print('Total lasso {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

# plot hdh, wind regression with lasso
gas.plot(color='blue', label='Daily gas demand 2018')
base2.plot(color='green', label='Gas demand with heating removed using wind and hdh')
base4.plot(color='red', label='Gas demand with heating removed using lasso')
plt.title('Daily gas demand 2018 with heat removed in 2 ways')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()
total_gas = gas.sum()
total_heat = base_heating.sum()

# get 2017 weather and give it the same index as 2018
weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2017.csv'
weather17 = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0)
weather17.index = weather.index

augment.augment(weather17)
daily_weather17 = weather17.resample('D').mean()

# forecast gas with RandomForrest built from weather of previous year
#variables = ['hdh', 'dailytemp', 'cdh', 'thermal', 'wind', 'zenith']
variables = ['hdh', 'dailytemp', 'cdh', 'thermal', 'wind']
gas_rf = forecast_demand(daily_weather17[variables], gas, daily_weather[variables])

# regression on mean hdh and weekend
coeffs = regression(daily_weather[['hdh', 'weekend']], gas.values)
print(coeffs)
c1 = coeffs[1]
c2 = coeffs[2]
base_heating  = daily_weather ['hdh'] * c1 + daily_weather ['weekend'] *c2
gas_hdh_weekend = base_heating + coeffs[0]

# compare models
stats.print_stats_header()
stats.print_stats(gas_hdh, gas, 'hdh only')
stats.print_stats(gas_hdh_wind, gas, 'hdh and wind')
stats.print_stats(gas_hdh_wind_ghi, gas, 'hdh, wind and ghi')
stats.print_stats(gas17_hdh_wind, gas17, 'hdh and wind 2017')
stats.print_stats(gas19_hdh_wind, gas19, 'hdh and wind 2019')
stats.print_stats(gas_lasso, gas, 'lasso')
stats.print_stats(gas_rf, gas, 'rf forecast')
stats.print_stats(gas_hdh_weekend, gas, 'hdh and weekend')

# compare by plotting
gas.plot(color='blue', label='Daily gas demand 2018')
gas_hdh_wind.plot(color='green', label='Daily gas forecast by hdh and wind')
gas_lasso.plot(color='yellow', label='Daily gas demand forecast by lasso parms')
gas_rf.plot(color='red', label='Daily gas demand forecast by rf model built on last years weather')
plt.title('Daily gas demand 2018 and 2 forecasts')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

# output more info
print('Gas: annual {:.2f} min {:.2f} max {:.2f}'.format(gas.sum(), gas.min(), gas.max()))

