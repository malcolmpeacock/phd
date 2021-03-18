# python script to do some plots to investigate lockdown levels

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
import statsmodels.api as sm

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Clean weather data.')
parser.add_argument('set', help='weather file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# lock down period
df_ld = df['2020-03-16' : '2020-07-01']

# daily values
daily_demand = df_ld['demand'].resample('D', axis=0).sum()
daily_temp = df_ld['tempm'].resample('D', axis=0).mean()
daily_level  = df_ld['lockdown'].resample('D',axis=0).first()

temps = daily_temp.values
demands = daily_demand.values
# Fit line through the points - the add constant bit gives us
# the intercept as well as the gradient of the fit line.
rmodel = sm.OLS(demands, sm.add_constant(temps))
residual_results = rmodel.fit()
#   print(residual_results.summary())
res_const = residual_results.params[0]
res_grad = residual_results.params[1]
print('Gradient {} intercept {}'.format(res_grad, res_const) )
if args.plot:
    # Fit of residuals line
    x = np.array([min(temps),max(temps)])
    y = res_const + res_grad * x
    fig, ax = plt.subplots()
    ax.scatter(temps, demands)
    plt.title('Temp vs demand')
    plt.xlabel('Temperature ', fontsize=15)
    plt.ylabel('Demand (MWh)', fontsize=15)
    plt.plot(x,y,color='red')
    plt.show()

adjusted_demand = daily_demand - (res_grad * daily_temp)
ax = daily_level.plot(label='Lockdown level')
plt.ylabel('Lockdown level', fontsize=15, color='red')
ax2 = ax.twinx()
#ax2.set_ylabel('Temperature (Degres C)', fontsize=15)
daily_demand.plot(label='demand')
adjusted_demand.plot(label='adjusted demand')

plt.title('Lock down period')
plt.xlabel('Hour of the day', fontsize=15)
plt.ylabel('Demand (MWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()

