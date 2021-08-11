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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures

# custom code
import stats
import readers
import augment

def regression(X, y):
    estimator = LinearRegression()
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
    print(fit.score(pf,y))
    print(estimator.coef_)
    print(estimator.intercept_)
#   p = fit.predict(Xp)
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
    print(input_df.index)
    print(output.index)
    coef = {}
    for column in input_df.columns:
        coef[column] = output.corr(input_df[column])
    return(coef)

# main program
weather_year = '2018'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'

gas = readers.read_gas(gas_filename)
print(gas)
gas.index = pd.DatetimeIndex(gas.index).tz_localize('UTC')
# TWh
gas = gas * 1e-9

weather_file = '/home/malcolm/uclan/output/wparms/weather_parms2018.csv'
weather = pd.read_csv(weather_file, header=0, parse_dates=[0], index_col=0)
weather.index = pd.DatetimeIndex(weather.index).tz_localize('UTC')

augment.augment(weather)
daily_weather = weather.resample('D').mean()

coeffs = correlation(daily_weather, gas, plot=True)
print('Daily Feature correlation')
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# lasso
print('Daily Lasso')
coeffs = lasso(daily_weather, gas, plot=True, normalize=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

# regression on mean hdh
coeffs = regression(daily_weather['hdh'].values.reshape(-1,1), gas.values)
c1 = coeffs[1]
print(coeffs)
base_heating  = daily_weather ['hdh'] * c1
base = gas - base_heating
# plot regression with hdh
gas.plot(color='blue', label='Daily gas demand 2018')
base.plot(color='green', label='Gas demand with heating removed')
plt.title('Daily gas demand 2018 base using mean HDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()

gas_hdh = base + base_heating

# correlation after heat removed

coeffs = correlation(daily_weather, base, plot=True)
print('Daily Feature correlation base')
print(coeffs)
for col, value in sorted(coeffs.items(), key=lambda item: abs(item[1]), reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))
print('Daily Lasso after heat removed')
coeffs = lasso(daily_weather, base, plot=True, normalize=True)
for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
    print('{:15}         {:.3f}'.format(col,value))

total_gas = gas.sum()
total_heat = base_heating.sum()
print('Total {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

# regression on mean hdh and windspeed
coeffs = regression(daily_weather[['hdh', 'wind']], gas.values)
print(coeffs)
c1 = coeffs[1]
c2 = coeffs[2]
base_heating  = daily_weather ['hdh'] * c1 + daily_weather ['wind'] * c2
base = gas - base_heating
# plot regression with hdh
gas.plot(color='blue', label='Daily gas demand 2018')
base.plot(color='green', label='Gas demand with heating removed using wind and hdh')
plt.title('Daily gas demand 2018 base using mean HDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()
total_gas = gas.sum()
total_heat = base_heating.sum()
print('Total (hdh and wind) {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

gas_hdh_wind = base + base_heating

# regression on mean hdh and windspeed and ghi
coeffs = regression(daily_weather[['hdh', 'ghi', 'wind']], gas.values)
print(coeffs)
c1 = coeffs[1]
c2 = coeffs[2]
c3 = coeffs[3]
base_heating  = daily_weather ['hdh'] * c1 + daily_weather ['ghi'] *c2 + daily_weather ['wind'] * c3
base = gas - base_heating
# plot regression with hdh
gas.plot(color='blue', label='Daily gas demand 2018')
base.plot(color='green', label='Gas demand with heating removed using wind and ghi')
plt.title('Daily gas demand 2018 base using mean HDH')
plt.xlabel('day', fontsize=15)
plt.ylabel('Energy (Twh) per day', fontsize=15)
plt.legend(loc='upper center')
plt.show()
total_gas = gas.sum()
total_heat = base_heating.sum()
print('Total hdh, wind and ghi {} Heating {} Percent {}'.format(total_gas, total_heat, total_heat / total_gas ) )

gas_hdh_wind_ghi = base + base_heating

# compare models
stats.print_stats_header()
stats.print_stats(gas_hdh, gas, 'hdh only')
stats.print_stats(gas_hdh_wind, gas, 'hdh and wind')
stats.print_stats(gas_hdh_wind_ghi, gas, 'hdh, wind and ghi')
