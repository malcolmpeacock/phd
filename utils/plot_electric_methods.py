# python script to plot electricity time series from methods TODO
# used for plot in my IECSF20 paper presentation.

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

# read 2018 heat demand for each method

hdd155_filename = "/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather2018I-Sbdew.csv"
heat = readers.read_copheat(hdd155_filename, ['heat'])

# read historic gas demand
year = '2018'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + year + '.csv'
gas = readers.read_gas(gas_filename)

# read the distribution of heating and pumps etc
# Read in combination of ASHP GSHP etc.

heating_path = "/home/malcolm/uclan/tools/python/scripts/heat/input"
parm_heat, parm_water = read.electric_parameters(heating_path, 'GB')

# Convert gas energy from kWh to MWh
gas = gas * (10 ** -3)

# 
# scale gas by 0.8 to convert to heat
gas = gas * 0.8

# scale by the annual demand as per the paper
domestic_space = 191
domestic_water = 56
commerical_space = 56
commerical_water = 9


# create daily gas-heat series for domestic/commercial, space/water
# convert to hourly via either the flat or BDEW profile.
# multiply by the hourly COP to create 4 hourly eelctric series
# combine these electric series

# calculate peak demands of all 4 methods + gas

# plot all methods + gas vs existing electric.

# Convert to daily so the plot is less cluttered.

gas = gas.resample('D').sum()
heat = heat.resample('D').sum()
electric = electric.resample('D').sum()

# Convert to TWh
gas = gas * (10 ** -6)
heat = heat * (10 ** -6)
electric = electric * (10 ** -6)

# Show half a year
gas = gas['2018-01-01' : '2018-05-30']
heat = heat['2018-01-01' : '2018-05-30']
electric = electric['2018-01-01' : '2018-05-30']

# output plots

gas.plot(label='Gas Energy', fontsize=15)
heat.plot(label='Heat Demand', fontsize=15)
electric.plot(label='Electricity', fontsize=15)
plt.title('Electricity gas and heat 2018')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Daily Demand (TWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=16)
plt.show()

# generation
year = '2018'
generation_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'

generation = readers.read_electric(generation_filename)
pv = generation['EMBEDDED_SOLAR_GENERATION']
wind = generation['EMBEDDED_WIND_GENERATION']
pv = pv.resample('D').sum()
wind = wind.resample('D').sum()
pv = pv['2018-01-01' : '2018-05-30']
wind = wind['2018-01-01' : '2018-05-30']

gas.plot(label='Gas Energy', fontsize=15)
heat.plot(label='Heat Demand', fontsize=15)
electric.plot(label='Electricity', fontsize=15)
wind.plot(label='Wind Generation', fontsize=15)
pv.plot(label='Solar Generation', fontsize=15)
plt.title('Electricity gas and heat 2018')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Daily Demand (TWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=16)
plt.show()
