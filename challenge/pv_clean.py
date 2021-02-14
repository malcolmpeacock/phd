# python script to clean up the pv data.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import argparse
import numpy as np
import math
import pvlib

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Clean pv data.')
parser.add_argument('set', help='PV file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"

# pv data
pv_filename = "{}pv_train_{}.csv".format(input_dir,dataset)
print('Cleaning {} {}'.format(dataset, pv_filename) )
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(pv)
# fix errors at particular points
# Note: replacing whole days doesn't make sense as the weather will be 
# different - so better to remove, so the join with weather df will have
# missing index values - but we can cope with this.
if dataset[0:3] == 'set':
    print('Fixing certain dodgy hours')
    # 2017-11-29 07:30:00 - no panel temperature value
    pv['panel_temp_C']['2017-11-29 07:30:00'] = pv['panel_temp_C']['2017-11-29 08:00:00']
    # 2018-02-28 11:00:00 - no power but irradiance.
    #   - irradiance is pretty low here as well
    pv['panel_temp_C']['2018-02-28 11:00:00'] = pv['panel_temp_C']['2018-02-28 10:30:00']
    # replace a suspect days with different ones.
#   utils.replace_day(pv, '2018-03-04', '2018-03-01')
    print('Dropping 2018-03-04 lot of missing values')
    pv.drop(pv.loc['2018-03-04'].index, inplace=True)
    # 2017-12-26 14:30:00 - Zero power value
    pv['pv_power_mw']['2017-12-26 14:30:00'] = pv['pv_power_mw']['2017-12-26 14:00:00']
    # replace a suspect days with different ones.
#   utils.replace_day(pv, '2018-06-15', '2018-06-14')
    # drop day completely
    print('DROPPING')
#   2018-03-02  - no power for several hours but irradiance
    pv.drop(pv.loc['2018-03-02'].index, inplace=True)
#   2018-03-03  - no power for several hours but irradiance
    pv.drop(pv.loc['2018-03-03'].index, inplace=True)
#   2018-03-05  - no power for several hours but irradiance
    pv.drop(pv.loc['2018-03-05'].index, inplace=True)
#   2018-03-08  - no power for several hours but irradiance
    pv.drop(pv.loc['2018-05-08'].index, inplace=True)
#   2018-06-15  - no power for several hours but irradiance
    pv.drop(pv.loc['2018-06-15'].index, inplace=True)

# ERROR CHECKS:

large_pv = pv['pv_power_mw'].max() * 0.8
small_pv = pv['pv_power_mw'].max() * 0.2
large_irrad = pv['irradiance_Wm-2'].max() * 0.8
small_irrad = pv['irradiance_Wm-2'].max() * 0.2
large_temp = pv['panel_temp_C'].max() * 0.8
small_temp = pv['panel_temp_C'].max() * 0.2

# pv large but irradiance small or temp small
pv_large = pv[pv['pv_power_mw']>large_pv]
suspect = pv_large[pv_large['irradiance_Wm-2']<small_irrad]
print('PV large but irradiance small {}'.format(len(suspect)) )
print(suspect)
suspect = pv_large[pv_large['panel_temp_C']<small_temp]
print('PV large but temp small {}'.format(len(suspect)) )
print(suspect)

# pv small but irradiance large or temp large
pv_small = pv[pv['pv_power_mw']<small_pv]
suspect = pv_small[pv_small['irradiance_Wm-2']>large_irrad]
print('PV small but irradiance large {}'.format(len(suspect)) )
print(suspect)
suspect = pv_small[pv_small['panel_temp_C']>large_temp]
print('PV small but temp large {}'.format(len(suspect)) )
print(suspect)


# replace NaN panel temps with zero if irradiance is zero
missing_panel_temp = pv[pv['panel_temp_C'].isnull().values]
#rint(missing_panel_temp)
missing_panel_temp_with0 = missing_panel_temp['pv_power_mw'] == 0.0
#print(missing_panel_temp_with0)
index_to_update = missing_panel_temp[missing_panel_temp_with0].index
pv['panel_temp_C'][index_to_update] = 0.0

# look for zero power during the middle of the day
zero_power = pv[pv['pv_power_mw'] == 0.0].copy()
zero_power['hour'] = zero_power.index.hour
print(zero_power)
midday_zero = zero_power[zero_power['hour'] >10]
midday_zero = midday_zero[midday_zero['hour'] <15]
print(' MID DAY ZERO')
print(midday_zero)

# check the errors were fixed
# PV
n_missing = utils.missing_times(pv, '30min')
if n_missing>0:
    print("Missing rows in pv {}".format(n_missing) )
for col in pv.columns:
    if pv[col].isna().sum() >0:
        print("ERROR nans in {}".format(col))
        print(pv[pv[col].isnull().values])
        quit()

#print(pv)
print('Smallest Power')
print(pv['pv_power_mw'].nsmallest())

# plot pv
if args.plot:
    pv['pv_power_mw'].plot(label='pv power', color='blue')
    plt.title('pv')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('PV Generation (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    pv['irradiance_Wm-2'].plot(label='iradiance', color='blue')
    plt.title('PV System Measured Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    pv['panel_temp_C'].plot(label='panel temp', color='blue')
    plt.title('Panel Temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('temperature (degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    expected_power = pv['irradiance_Wm-2'] * 0.8 * 5000.0
#   efficency = pv['pv_power_mw'] / pv['irradiance_Wm-2']
    efficency = pv['pv_power_mw'] / expected_power
    plt.scatter(pv['panel_temp_C'].values, efficency.values, s=12, color='blue')
    plt.title('Panel Temperature vs Efficiency')
    plt.xlabel('Panel Temperature (degrees C)', fontsize=15)
    plt.ylabel('Efficiency = power/expected', fontsize=15)
    plt.show()

    fewdays = pv['2018-06-01 00:00:00' : '2018-06-04 23:30:00']
    ax = fewdays['pv_power_mw'].plot(label='PV power generation', color='red')
    plt.ylabel('Power (MW)', fontsize=15, color='red')
    ax2 = ax.twinx()
    ax2.set_ylabel('Irradiance (W/M2)', fontsize=15)
    fewdays['irradiance_Wm-2'].plot(label='iradiance', color='blue')
    plt.title('PV System Measured Irradiance and power')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

pv.columns = ['pv_ghi', 'pv_power', 'pv_temp']
print(pv.columns)

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}pv_fixed_{}.csv'.format(output_dir, dataset)

pv.to_csv(output_filename, float_format='%.2f')
