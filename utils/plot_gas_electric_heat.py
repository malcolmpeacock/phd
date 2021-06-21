# python script to plot gas electric methods electric time series

import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import readers

# main program

# read 2018 historical electricity demand

scotland_factor = 1.1    # ( Fragaki et. al )
demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand18 = readers.read_electric_hourly(demand_filename)
electric = demand18['ENGLAND_WALES_DEMAND'] * scotland_factor
print(electric)

# read 2018 electric heat demand.

path = "/home/malcolm/uclan/tools/python/scripts/heat/output/2018/"
profile = 'flat'
# profile = 'bdew'

gas_filename    = '{}GBRef2018Weather2018I-G{}.csv'.format(path, profile)
gas = readers.read_copheat(gas_filename, ['electricity'])

bdew_filename   = "{}GBRef2018Weather2018I-B{}.csv".format(path, profile)
bdew = readers.read_copheat(bdew_filename, ['electricity'])

watson_filename = "{}GBRef2018Weather2018I-W{}.csv".format(path, profile)
watson = readers.read_copheat(watson_filename, ['electricity'])

hdd155_filename = "{}GBRef2018Weather2018I-S{}.csv".format(path, profile)
hdd155 = readers.read_copheat(hdd155_filename, ['electricity'])

hdd128_filename = "{}GBRef2018Weather2018I-H{}.csv".format(path, profile)
hdd128 = readers.read_copheat(hdd128_filename, ['electricity'])

# calculate peak demand and ramp rate.
gas_diff = gas.diff()
bdew_diff = bdew.diff()
watson_diff = watson.diff()
hdd155_diff = hdd155.diff()
hdd128_diff = hdd128.diff()
electric_diff = electric.diff()
print('Method Demand Max Min       Ramp Rate Max Min')
print('Gas      {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(gas.max(), gas.min(), gas_diff.max(), gas_diff.min() ) )
print('BDEW     {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(bdew.max(), bdew.min(), bdew_diff.max(), bdew_diff.min() ) )
print('Watson   {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(watson.max(), watson.min(), watson_diff.max(), watson_diff.min() ) )
print('HDD155   {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(hdd155.max(), hdd155.min(), hdd155_diff.max(), hdd155_diff.min() ) )
print('HDD128   {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(hdd128.max(), hdd128.min(), hdd128_diff.max(), hdd128_diff.min() ) )
print('historic {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(electric.max(), electric.min(), electric_diff.max(), electric_diff.min() ) )

# convert to daily for plotting

electric = electric.resample('D').sum()
gas = gas.resample('D').sum()
bdew = bdew.resample('D').sum()
watson = watson.resample('D').sum()
hdd155 = hdd155.resample('D').sum()
hdd128 = hdd128.resample('D').sum()

# output plots

gas.plot(label='Electric Heat Gas', fontsize=15)
bdew.plot(label='Electric Heat BDEW', fontsize=15)
watson.plot(label='Electric Heat Watson', fontsize=15)
hdd155.plot(label='Electric Heat HDD15.5', fontsize=15)
hdd128.plot(label='Electric Heat HDD12.8', fontsize=15)
electric.plot(label='Historic Electricity', fontsize=15)
plt.title('Electrification of heat 2018')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Daily Demand (TWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=16)
plt.show()

# add to the historic demand.
gas = gas + electric
bdew = bdew + electric
watson = watson + electric
hdd128 = hdd128 + electric
hdd155 = hdd155 + electric

# calculate peak demand and ramp rate.
gas_diff = gas.diff()
bdew_diff = bdew.diff()
watson_diff = watson.diff()
hdd155_diff = hdd155.diff()
hdd128_diff = hdd128.diff()
electric_diff = electric.diff()
print('After adding to the historic series')
print('Method Demand Max Min       Ramp Rate Max Min')
print('Gas      {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(gas.max(), gas.min(), gas_diff.max(), gas_diff.min() ) )
print('BDEW     {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(bdew.max(), bdew.min(), bdew_diff.max(), bdew_diff.min() ) )
print('Watson   {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(watson.max(), watson.min(), watson_diff.max(), watson_diff.min() ) )
print('HDD155   {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(hdd155.max(), hdd155.min(), hdd155_diff.max(), hdd155_diff.min() ) )
print('HDD128   {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(hdd128.max(), hdd128.min(), hdd128_diff.max(), hdd128_diff.min() ) )
print('historic {:.2f}    {:.2f}         {:.2f}   {:.2f}'.format(electric.max(), electric.min(), electric_diff.max(), electric_diff.min() ) )

# output plots

gas.plot(label='Electric with Heat Gas', fontsize=15)
bdew.plot(label='Electric with Heat BDEW', fontsize=15)
watson.plot(label='Electric with Heat Watson', fontsize=15)
hdd155.plot(label='Electric with Heat HDD15.5', fontsize=15)
hdd128.plot(label='Electric with Heat HDD12.8', fontsize=15)
electric.plot(label='Historic Electricity', fontsize=15)
plt.title('Electrification of heat 2018')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Daily Demand (TWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=16)
plt.show()
