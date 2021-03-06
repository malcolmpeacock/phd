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
# custom code
import stats

# fit a regression line through the cops
def cop_plot(df_cop, hp):
    rmodel = sm.OLS(df_cop['cop'].to_numpy(), sm.add_constant(df_cop['deltat'].to_numpy()))
    residual_results = rmodel.fit()
    res_const = residual_results.params[0]
    res_grad = residual_results.params[1]
    x = np.array([df_cop['deltat'].min(),df_cop['deltat'].max()])
    y = res_const + res_grad * x

    range_start = math.floor(df_cop['deltat'].min())
    range_end = math.ceil(df_cop['deltat'].max())
    bins = pd.cut(df_cop['deltat'], range(range_start,range_end,2) )
    print(bins)
    means = df_cop.groupby(bins).mean()
    print(means)

    plt.scatter(df_cop['deltat'], df_cop['cop'], s=12)
    # regression line
    plt.plot(x, y, color='red')
    # line of COPs calculated using mean cop in bins
    plt.plot(means['deltat'], means['cop'], color='green')
    plt.title('RHPP {} COP vs DELTA T'.format(hp))
    plt.xlabel('Temperature Difference (degrees C)')
    plt.ylabel('COP')
    plt.show()

# cop calculation
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
    return df_hp

# main program

# read in the data
output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
filename = output_dir + 'electric.csv'
df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# convert from Wh to kWh

df['synthetic'] = df['synthetic'] * 0.001
df['real'] = df['real'] * 0.001

synthetic = df['synthetic']
real = df['real']

# extract a sample few days to look at hourly
synthetic = synthetic['2014-01-01 00:00:00':'2014-01-04 23:00:00']
real = real['2014-01-01 00:00:00':'2014-01-04 23:00:00']
print('synthetic')
print(synthetic)
print('real')
print(real)

synthetic.plot(label='Synthetic Electicity Time Series', color='blue')
real.plot(label='Real Electicity Time Series', color='red')
plt.title('Hourly synthetic electric heat vs measured data 4 days')
plt.xlabel('Hour of the day', fontsize=15)
plt.ylabel('Electricity Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# cop calculation
df_ashp = df[['temperature', 'elec_ASHP', 'heat_ASHP']]
df_ashp = df_ashp[ df_ashp['elec_ASHP'] > 0.0]
df_ashp['deltat'] = (df_ashp['temperature'] * -1.0) + 40.0
df_ashp['cop'] = df_ashp['heat_ASHP'] / df_ashp['elec_ASHP']
df_ashp = df_ashp.sort_values('deltat', axis=0)


# fit a regression line through the cops
rmodel = sm.OLS(df_ashp['cop'].to_numpy(), sm.add_constant(df_ashp['deltat'].to_numpy()))
residual_results = rmodel.fit()
res_const = residual_results.params[0]
res_grad = residual_results.params[1]
x = np.array([df_ashp['deltat'].min(),df_ashp['deltat'].max()])
y = res_const + res_grad * x

plt.scatter(df_ashp['deltat'], df_ashp['cop'], s=12)
plt.plot(x, y, color='red')
plt.title('RHPP ASHP COP vs DELTA T')
plt.xlabel('Temperature Difference (degrees C)')
plt.ylabel('COP')
plt.show()

# GSHP

df_gshp = cop(df,'GSHP',40.0, 'soiltemp')
cop_plot(df_gshp, 'GSHP')

# convert to daily
dfd = df.resample('D').sum()

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(dfd['synthetic'], dfd['real'], 'Electric Daily', 1, True)
total_real = dfd['real'].sum()
total_synthetic = dfd['synthetic'].sum()
percent = (total_real - total_synthetic) / total_real
print("Total Real {} Synthetic {} Percentage {}".format(total_real, total_synthetic, percent) )

# output plots

dfd['synthetic'].plot(label='Synthetic Electicity Time Series', color='blue')
dfd['real'].plot(label='Real Electicity Time Series', color='red')
plt.title('Comparison of daily synthetic electric and measured heat pump')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Electricity Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()


# load duration curves

sorted_real = dfd['real'].sort_values(ascending=False)
sorted_synthetic = dfd['synthetic'].sort_values(ascending=False)

sorted_real.plot(label='Electricity Demand Measured)', use_index=False, color='blue')
sorted_synthetic.plot(label='Electricity Demand Modelled)', use_index=False, color='red')
plt.title('Electricity Demand sorted by demand')
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
    print(profile)
    hours[str(hour)] = profile

df = pd.DataFrame(data=hours)
df.index = range(-15,35,5)
print(df)
# swap rows and columns
df = df.transpose()
df.index = pd.Index( "{:02d}:00".format(i) for i in range(0,24,1) )
print(df)

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
