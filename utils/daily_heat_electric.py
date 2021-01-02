# python script to:
#   - create a generation (supply) daily time series from an input (PV Wind)
#     generation series.
#   - create a daily heat time series from an hourly one.
#   - create a daily mean temperatures from hourly
#   - using COP regression equation calculate daily COP
#   - generate daily eletric series
#   - look at storage impact.
#
# NOTE: - heat program now does COP and electric
#         but this was used to compare the methods.

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm

# custom code
import stats
import readers
from misc import upsample_df

# GSHP = 10.29 - 0.21 DT + 0.0012 DT2
# 0.85 to allow for real world issues
def cop_gshp(temperature):
    cop = 10.29 - (0.21*temperature) + (0.0012*temperature*temperature)
    return cop * 0.85

# ASHP = 6.08 - 0.09 DT + 0.0005 DT2
# 0.85 to allow for real world issues
def cop_ashp(temperature):
    cop = 6.08 - (0.09*temperature) + (0.0005*temperature*temperature)
    return cop * 0.85

# main program
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018HDD15.5.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018Ruhnau.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018Watson.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Wbdew.csv'
demand_filename = '/home/malcolm/uclan/tools/python/output/2018/GBRef2018Weather2018I-Sbdew.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Sbdew.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Rbdew.csv'
supply_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'

# read heat demand and temp
demand = readers.read_copheat(demand_filename,['space','water','temperature'])
# print('DEMAND')
# print(demand)
space = demand['space']
space = space.resample('D').sum()
print('space')
print(space)
water = demand['water']
water = water.resample('D').sum()
temperature = demand['temperature']
temperature = temperature.resample('D').mean()
print('temp')
print(temperature)

# sink = 40-T, minimum 15
# NOTE: this is wierd because if sink=40-T, then DeltaT = sink - T = 40-2*T
sink_radiator = ((temperature - 40.0) * -1.0).clip(15)
print('sink_radiator')
print(sink_radiator)
# sink = 30-0.5*T, minimum 15
sink_floor = ((temperature*0.5 - 30.0) * -1.0).clip(15)
# sink = 50
sink_water = temperature*0.0 + 50.0
dt_radiator = sink_radiator - temperature
dt_floor = sink_floor - temperature
dt_water = sink_water - temperature
# (90%) ASHP = 6.08 - 0.09 DT + 0.0005 DT2
# space - radiator (90%)
ashp_space_radiator = space * 0.9 * 0.9 / dt_radiator.apply(cop_ashp)
# space - floor    (10%)
ashp_space_floor = space * 0.9 * 0.1 / dt_floor.apply(cop_ashp)
# water - water    (100%)
ashp_water = water * 0.9 / dt_water.apply(cop_ashp)

# (10%) GSHP = 10.29 - 0.21 DT + 0.0012 DT2
# space - radiator (90%)
gshp_space_radiator = space * 0.1 * 0.9 / dt_radiator.apply(cop_gshp)
# space - floor    (10%)
gshp_space_floor = space * 0.1 * 0.1 / dt_floor.apply(cop_gshp)
# water - water    (10%)
gshp_water = water * 0.1 / dt_water.apply(cop_gshp)

electric_heat = ashp_space_radiator + ashp_space_floor + ashp_water + gshp_space_radiator + gshp_space_floor + gshp_water
print(electric_heat)
# read electric demand

electric = readers.read_electric(supply_filename)
# factor the generation to match the demand
# (imagine we added just enough solar and wind annually and then we can
#  look at net to see how much storage we might need )
supply = electric['EMBEDDED_WIND_GENERATION'] + electric['EMBEDDED_SOLAR_GENERATION']
# print('Annual supply before up sample {}'.format(supply.sum()) )
# supply = supply.resample('60min').pad()
# supply = upsample_df(supply.resample,'60min')
# print('Annual supply after up sample {}'.format(supply.sum()) )
# supply = supply.resample('60min')
print('SUPPLY')
print(supply.index)
print(supply)
supply = supply * ( electric_heat.sum() / supply.sum() )

# convert all to net demand
net_demand = electric_heat - supply
# convert to GWh
net_demand = net_demand / 1000.0
supply = supply / 1000.0
electric_heat = electric_heat / 1000.0
print(net_demand.head())

# calculate peak and annual demands
print('Annual demand {:.2f} TWh supply {:.2f} TWh'.format(electric_heat.sum()/1000.0, supply.sum()/1000.0) )
peak_added = electric_heat.max()
print('Peak Electric Heat Demand Daily {:.2f} Gw Hourly {:.2f} Gw '.format(peak_added, peak_added/24.0) )
peak_net = net_demand.max()
print('Peak Net demand {:.2f} Gw Hourly {:.2f} Gw '.format(peak_net, peak_net/24.0) )

# output plots

electric_heat.plot(label='Demand')
plt.title('Demand from electric heat')
plt.xlabel('Day of the year')
plt.ylabel('Demand (GWh)')
# plt.legend(loc='upper right')
plt.show()

net_demand.plot(label='Net Demand')
plt.title('Net Demand from electric heat')
plt.xlabel('Day of the year')
plt.ylabel('Net Demand (GWh)')
# plt.legend(loc='upper right')
plt.show()

# calculate storage
store = 0.0
eta_charge = 0.75
eta_discharge = 0.75
# loop through each hourly value ...
for index, value in net_demand.items():
    # if net demand is positive, subtract from the store
    if value > 0.0:
        store = store - value * eta_discharge
    # if net demand is negative, add to the store
    # note: value is negative, so we are adding here!
    else:
        store = store - value * eta_charge
print('Storage {:.2f} TWh'.format(store/1000.0))
