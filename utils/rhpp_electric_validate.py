# python script to validate synthetic electric demand against actual.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math
from sklearn.ensemble import IsolationForest
# custom code
import stats

# fit a regression line through the cops
def cop_plot(df_cop, hp):
    rmodel = sm.OLS(df_cop['cop'].to_numpy(), sm.add_constant(df_cop['deltat'].to_numpy()))
    residual_results = rmodel.fit()
    res_const = residual_results.params[0]
    res_grad = residual_results.params[1]
    print('COP Regression gradient {} const {}'.format(res_const, res_grad))
    x = np.array([df_cop['deltat'].min(),df_cop['deltat'].max()])
    y = res_const + res_grad * x

    # polynomial of degree 2 fit
#   coeffs = np.polynomial.polynomial.polyfit(df_cop['cop'].to_numpy(), df_cop['deltat'].to_numpy(), 2)
    coeffs = np.polynomial.polynomial.polyfit(df_cop['deltat'].to_numpy(), df_cop['cop'].to_numpy(), 2)
    print(coeffs)
    y = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x

    range_start = math.floor(df_cop['deltat'].min())
    range_end = math.ceil(df_cop['deltat'].max())
    bins = pd.cut(df_cop['deltat'], range(range_start,range_end,2) )
#   print(bins)
    means = df_cop.groupby(bins).mean()
#   print(means)

    plt.scatter(df_cop['deltat'], df_cop['cop'], s=12, label='cop vs deltat')
    # regression line
    plt.plot(x, y, color='red', label='regression line')
    # line of COPs calculated using mean cop in bins
#   plt.plot(means['deltat'], means['cop'], color='green', label='mean cop')

    plt.title('RHPP {} COP vs DELTA T'.format(hp))
    plt.xlabel('Temperature Difference (degrees C)')
    plt.ylabel('COP')
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

# cop calculation
#   df
#   hp
#   sink_time
#   source_name  = variable name for source temperature
def cop(df, hp, sink_temp,source_name):
    df_hp = df[[source_name, 'elec_'+hp, 'heat_'+hp]]
    # only include points where there is some electricity
    # put in some minimum as it may consume some when not heating?
    df_hp = df_hp[ df_hp['elec_'+hp] > 200.0]
    print('{} SMALL'.format(hp) )
    print(df_hp['elec_'+hp].nsmallest() )
    print(df_hp['elec_'+hp].nlargest() )
    df_hp = df_hp[ df_hp['heat_'+hp] > 200.0]
    df_hp['deltat'] = (df_hp[source_name] * -1.0) + sink_temp
    df_hp['cop'] = df_hp['heat_'+hp] / df_hp['elec_'+hp]
#   df_hp = df_hp.sort_values('deltat', axis=0)
    print('{} {} mean cop {}'.format(hp, source_name, df_hp['cop'].mean() ))
    return df_hp

# daily cop calculation
#   df
#   hp
#   sink_time
#   source_name  = variable name for source temperature
def daily_cop(df, hp, sink_temp,source_name):
    source_temp = df[source_name].resample('D').mean()
    elec = df['elec_'+hp].resample('D').sum()
    heat = df['heat_'+hp].resample('D').sum()
 
    df_hp = pd.concat([source_temp, elec, heat], axis=1,keys=[source_name, 'elec_'+hp, 'heat_'+hp ])
#   print(df_hp)
    # only include points where there is some electricity
    # put in some minimum as it may consume some when not heating?
    df_hp = df_hp[ df_hp['elec_'+hp] > 200.0]
    print('{} SMALL'.format(hp) )
    print(df_hp['elec_'+hp].nsmallest() )
    print(df_hp['elec_'+hp].nlargest() )
    df_hp = df_hp[ df_hp['heat_'+hp] > 200.0]
    df_hp['deltat'] = (df_hp[source_name] * -1.0) + sink_temp
    df_hp['cop'] = df_hp['heat_'+hp] / df_hp['elec_'+hp]
#   df_hp = df_hp.sort_values('deltat', axis=0)
    return df_hp

# main program

# process command line
parser = argparse.ArgumentParser(description='Validate electricity demand.')
parser.add_argument('--scale', action="store", dest="scale", help='Amount to scale by to fix the bias', default=1.0, type=float )
parser.add_argument('--cop', action="store_true", dest="cop", help='Do COP plots', default=False)
args = parser.parse_args()

# read in the data
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
filename = output_dir + 'electric.csv'
df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# convert from Wh to kWh

df['synthetic'] = df['synthetic'] * 0.001
df['real'] = df['real'] * 0.001

synthetic = df['synthetic']
real = df['real']

# scale the synthetic to correct the bias

synthetic = synthetic * args.scale

# extract a sample few days to look at hourly
synthetic = synthetic['2014-01-01 00:00:00':'2014-01-04 23:00:00']
real = real['2014-01-01 00:00:00':'2014-01-04 23:00:00']
# print('synthetic')
# print(synthetic)
# print('real')
# print(real)

synthetic.plot(label='Predicted Electricity Time Series BDEW method', color='blue')
real.plot(label='Actual Electricity Time Series (measured heat pumps)', color='red')
plt.title('Hourly synthetic electric heat vs measured data 4 days')
plt.xlabel('Hour of the day', fontsize=15)
plt.ylabel('Electricity Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# COP calculation
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
if args.cop:

# ASHP

#df_ashp = cop(df,'ASHP',40.0, 'temperature')
#df_gshp = daily_cop(df,'GSHP',40.0, 'soiltemp')
    filename = output_dir + 'ashp.csv'
    df_ashp = pd.read_csv(filename, header=0, sep=',', index_col=0, squeeze=True)
    count_raw = len(df_ashp)
    # replace NaNs by interpolation
    df_ashp = df_ashp.dropna()
    count_nan = len(df_ashp)
    # remove stuff that physically doesn't make sense
    df_ashp = df_ashp[(df_ashp['cop'] < 10.0) & (df_ashp['cop'] > 0.001)]
    count_phys = len(df_ashp)
    # remove outliers using Isolation Forrest
    print('Isolation Forrest on ASHP')
    outliers = IsolationForest(random_state=0).fit_predict(df_ashp)
    df_ashp = df_ashp.loc[outliers!=-1]
    count_isol = len(df_ashp)
    print('Original cop {} remove NaN {} remove >10 and <0.001 {} Isolation Forrest {}'.format(count_raw, count_nan, count_phys, count_isol))

    cop_plot(df_ashp, 'ASHP')

# GSHP

    #df_gshp = cop(df,'GSHP',40.0, 'soiltemp')
    #df_gshp = daily_cop(df,'GSHP',40.0, 'soiltemp')
    filename = output_dir + 'gshp.csv'
    df_gshp = pd.read_csv(filename, header=0, sep=',', index_col=0, squeeze=True)
    # replace NaNs by interpolation
    df_gshp = df_gshp.interpolate()
    # remove stuff that physically doesn't make sense
    df_gshp = df_gshp[(df_gshp['cop'] < 10.0) & (df_gshp['cop'] > 0.001)]
    # remove outliers using Isolation Forrest
    print('Isolation Forrest on GSHP')
    outliers = IsolationForest(random_state=0).fit_predict(df_gshp)
    df_gshp = df_gshp.loc[outliers!=-1]
    cop_plot(df_gshp, 'GSHP')

# convert to daily
dfd = df.resample('D').sum()
daily_synthetic = dfd['synthetic'] * args.scale
daily_real = dfd['real']

# correction factor between sereis
total_real = daily_real.sum()
total_synthetic = daily_synthetic.sum()
percent = (total_real - total_synthetic) / total_real
mean_factor =  daily_real.mean() / daily_synthetic.mean()
print("Total Real {} Synthetic {} Percentage {} Mean Factor {}".format(total_real, total_synthetic, percent, mean_factor) )
#fixed_synthetic = daily_synthetic  * mean_factor
#fixed_synthetic = (daily_synthetic * 0.4669) + 38.36
#fixed_synthetic = (daily_synthetic / 0.4669) - 38.36
#fixed_synthetic = (daily_synthetic - 38.36) / 0.4669
#fixed_synthetic = (daily_synthetic - 38.36) * 0.4669
#fixed_synthetic = (daily_synthetic + 38.36) * 0.4669
fixed_synthetic = (daily_synthetic * 1.4669) + 38.36

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(daily_synthetic, daily_real, 'Electric Daily', 1, True)
stats.print_stats(fixed_synthetic  * mean_factor, daily_real, 'Electric Fixed', 1, True)

# output plots

#print('COLUMNS')
#print(dfd.columns)
daily_synthetic.plot(label='Modelled Electricity Time Series', color='blue')
daily_real.plot(label='Actual Measured Electricity Time Series', color='red')
fixed_synthetic.plot(label='Fixed Electricity Time Series', color='green')
plt.title('Comparison of predicted electricty demand and measured from heat pumps')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Electricity Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# load duration curves

sorted_real = daily_real.sort_values(ascending=False)
sorted_synthetic = daily_synthetic.sort_values(ascending=False)

sorted_real.plot(label='Measured Electricity Demand', use_index=False, color='blue')
sorted_synthetic.plot(label='Modelled Electricity Demand', use_index=False, color='red')
plt.title('Trial group of houses heat pumps electricity demand sorted by demand')
plt.xlabel('Day sorted by demand', fontsize=15)
plt.ylabel('Daily Eletricity Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(rotation=0)
plt.show()

# hourly profile
df = df[['temperature', 'elec_ASHP', 'heat_ASHP', 'elec_GSHP', 'heat_GSHP']]
df['hour'] = df.index.hour
hours = {}
for hour in range(0,24):
    # extract the hour
    hourly = df[df['hour'] == hour]
    # split into binds by temperature ranges and average
    bins = pd.cut(hourly['temperature'], range(-15,40,5) )
    means = hourly.groupby(bins).mean()
    # split according to UK distribution of GSHP and ASHP
    means['profile'] = means['heat_GSHP'] * 0.1 + means['heat_ASHP'] * 0.9
    # replace NaNs at start and end with nearest values
    profile = means['profile'].fillna(method='bfill').fillna(method='ffill')
#   print(profile)
    hours[str(hour)] = profile

df = pd.DataFrame(data=hours)
df.index = range(-15,35,5)
# print(df)
# swap rows and columns
df = df.transpose()
df.index = pd.Index( "{:02d}:00".format(i) for i in range(0,24,1) )
# print(df)

# normalize
for column in df:
    df[column] = df[column] / df[column].sum()

# plot example

example_profile = df[10]
example_profile.plot()
plt.title('Hourly Profile')
plt.xlabel('Hour of the day', fontsize=15)
plt.ylabel('Demand (kWh)', fontsize=15)
#plt.xticks(rotation=0)
plt.show()


output_filename = "/home/malcolm/uclan/data/rhpp-heatpump/testing/hourly_factors.csv"
df.to_csv(output_filename, float_format='%g');

# sink tmperatures
# read in the data
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
filename = output_dir + 'sinks.csv'
df = pd.read_csv(filename, header=0, sep=',', index_col=0, squeeze=True)
#print(df)
print('SINKS: min {} max {} mean {}'.format(df['min'].min(), df['max'].max(), df['mean'].mean() ) )
