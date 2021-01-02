# python script to create a net demand time series
# from an input (PV Wind) generation series and an input electric (heat)
# hourly time series. Then look at storage impact.

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
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018HDD15.5.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018Ruhnau.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018Watson.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Sbdew.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Sbdew.csv'
demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Rbdew.csv'
supply_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'

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
plt.title('Net Demand from eletric heat')
plt.xlabel('Day of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# calculate storage
store = 0.0
eta_charge = 0.75
eta_discharge = 0.75
for index, value in net_demand.items():
    if value > 0.0:
        store = store - value * eta_discharge
    else:
        store = store + value * eta_charge
print('Storage {} TWh'.format(store))
