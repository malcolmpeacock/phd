# Compare modelling of energy storage using an hourly time series
# vs a daily time series.
# NOTE: this could also be done half hourly ?

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

# calculate storage

def storage(net_demand):
    history = net_demand.copy()
    store = 0.0
    min_store = 0.0
    eta_charge = 0.75
    eta_discharge = 0.75
    count=0
    for index, value in net_demand.items():
        history.iat[count] = store / 1000.0
        count += 1
        # NOte: both subtract because value is negative in the 2nd one!
        if value > 0.0:
            store = store - value * eta_discharge
        else:
            store = store - value * eta_charge
        if store < min_store:
            min_store = store
    print('Storage {:.2f} GWh'.format(min_store/1000.0))
    return history

# main program

#  read 2018 historical demand for England and Waltes

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
electric = readers.read_electric_hourly(demand_filename)
historic = electric['ENGLAND_WALES_DEMAND']

#  create a supply based on the existing PV and Solar

supply = electric['EMBEDDED_WIND_GENERATION'] + electric['EMBEDDED_SOLAR_GENERATION']
supply = supply * ( historic.sum() / supply.sum() )

#  calculate how much storage we need

print('Hourly')
net_hourly = historic - supply
stored_hourly = storage(net_hourly)

#  convert them to daily
#  calculate how much storage we need again
# TODO - PLOT BOTH hourly and daily.
# record the size of the store and plot that also

print('Daily')
net_daily = historic.resample('D').sum() - supply.resample('D').sum()
# convert to same units as hourly - ie MWh
stored_daily = storage(net_daily)
net_daily = net_daily / 24.0

print('net_hourly {} stored_hourly {} net_daily {} stored_daily {}'.format(len(net_hourly), len(stored_hourly), len(net_daily), len(stored_daily)))

print('hourly largest')
print(net_hourly.nlargest())
print('daily largest')
print(net_daily.nlargest())

historic.plot(label='historic')
supply.plot(label='supply')
net_hourly.plot(label='net_hourly')
plt.title('Demand, scaled renewables and net')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()


net_hourly.plot(label='net_hourly')
stored_hourly.plot(label='hourly_store')
plt.title('Net demand and store')
plt.xlabel('Hour of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

net_daily.plot(label='net_daily')
stored_daily.plot(label='daily_store')
plt.title('Net demand and store')
plt.xlabel('Day of the year')
plt.ylabel('Demand (MWh)')
plt.legend(loc='upper right')
plt.show()
