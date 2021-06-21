# python script to create a net demand time series
# from an input (PV Wind) generation series and an electric (heat)
# daily time series generated from gas. Then look at storage impact.

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

# main program
weather_year = '2018'
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'
supply_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/Ref2018Weather2018Rbdew.csv'

# read the cop from the heatandcop ouput
demand = readers.read_copheat(demand_filename, ['ASHP_floor','ASHP_radiator','ASHP_water','GSHP_floor','GSHP_radiator','GSHP_water'])

# convert to daily
demand = demand.resample('D').mean()
print(demand)

# read the gas
gas = readers.read_gas(gas_filename)

# convert to TWh
gas = gas * 1e-9
print(gas)

# scale gas by 0.8 to convert to heat
gas = gas * 0.8

# scale by the annual demand as per the paper
annual_space = 191 + 56
annual_water = 56 + 9

# space and water split?
#space_proportion = 
# how can we split the gas time series into space and water to apply the COP?

# multiply by cop in proportion for ashp etc to get electric
sink_radiator = ((temperature - 40.0) * -1.0).clip(15)
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


# read the electricity demand
demand = readers.read_demand(demand_filename)
print('DEMAND')
print(demand.index)
print(demand)

electric = readers.read_electric_hourly(supply_filename)
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
supply = supply * ( demand.sum() / supply.sum() )

# convert all to net demand
net_demand = demand - supply
print(net_demand.head())

# calculate peak and annual demands
print('Annual demand {} supply {}'.format(demand.sum(), supply.sum()) )
print('Peak Electric Heat Demand {} '.format(demand.max()) )
print('Peak Net demand {} '.format(net_demand.max()) )

# output plots

net_demand.plot(label='Net Demand')
plt.title('Net Demand from electric heat')
plt.xlabel('Day of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# calculate storage
store = 0.0
min_store = 0.0
eta_charge = 0.75
eta_discharge = 0.75
for index, value in net_demand.items():
    if value > 0.0:
        store = store - value * eta_discharge
    else:
        store = store + value * eta_charge
    if store < min_store:
        min_store = store
print('Storage at end {} TWh max {} TWh'.format(-store/1000000.0, -min_store/1000000.0))
