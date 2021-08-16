# How much storage is required for different amounts of PV and Wind ? using the capacity factors
# Reproduce Katerina graph for 2018 only.

# library stuff
import sys
import pandas as pd
from datetime import datetime
# import pytz
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import numpy as np
import scipy.interpolate

# custom code
import stats
import readers
from misc import upsample_df

# calculate storage

def storage(net_demand, eta=0.75):
    history = net_demand.copy()
    store = 0.0
    min_store = 0.0
    eta_charge = eta
    eta_discharge = eta
    count=0
    for index, value in net_demand.items():
        history.iat[count] = store
        count += 1
        # Note: both subtract because value is negative in the 2nd one!
        if value > 0.0:
            store = store - value * eta_discharge
        else:
            store = store - value * eta_charge
        if store < min_store:
            min_store = store
#   print('Storage {:.2f} GWh'.format(min_store))
    return min_store, history

# constant storage line

def storage_line(df,storage_value):
    x=[]
    y=[]
    # for each pv value ...
    for i_wind in range(0,14):
        f_wind = i_wind * 0.5
        # extract those values with a wind=xs
        is_xs = df['f_wind'] == f_wind
        df_xs = df[is_xs]
        # check storage in range
        if storage_value < df_xs['storage'].max() and storage_value > df_xs['storage'].min():
            # sort them by storage
            df_xs = df_xs.sort_values('storage',ascending=False)
            # interpolate a pv value for the storage
            y_interp = scipy.interpolate.interp1d(df_xs['storage'], df_xs['f_pv'])
            f_pv = y_interp(storage_value)
            # store the points
            x.append(f_wind)
            y.append(f_pv.item())

    sline = { 'Pw' : x, 'Ps' :y }
    df = pd.DataFrame(data=sline)
    return df

def storage_grid(demand, wind, pv, Lw, Ls, eta):
    print('storage_grid: demand max {} min {} mean {}'.format(demand.max(), demand.min(), demand.mean()) )
    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [] }
    ndays = len(demand)

    # For each percent of PV/Wind
    for i_pv in range(0,14):
        for i_wind in range(0,14):
            f_pv = i_pv * 0.5
            f_wind = i_wind * 0.5
            # energy supply is calculated using the capacity factors
            supply = (wind * f_wind * Lw) + (pv * f_pv * Ls)
            net = demand - supply

            #  calculate how much storage we need

            store_size, store_hist = storage(net, eta)
            results['f_pv'].append(f_pv)
            results['f_wind'].append(f_wind)
            results['storage'].append(store_size * ndays)

    df = pd.DataFrame(data=results)
    return df

# main program

Lw = 0.28
Ls = 0.116

# minimum generation

Ps = []
Pw = []

# y intercept
Pw.append(0.0)
Ps.append( 1 / Ls )
# x intercept
Ps.append(0.0)
Pw.append( 1 / Lw )

min_energy_line = { 'Pw' : Pw, 'Ps' : Ps }
df_min = pd.DataFrame(data=min_energy_line)
df_min.plot(x='Pw', y='Ps', label='minimum generation')
plt.title('Minimum Generation')
plt.xlabel('Wind')
plt.ylabel('Solar PV')
plt.show()

# years = ['2018']
years = ['2015', '2016', '2017', '2018', '2019']

#  read historic demand and generation for England and Wales
demand_years = {}
wind_years = {}
pv_years = {}
max_demand = 0.0
max_wind = 0.0
max_pv = 0.0
for year in years:
    electric_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'
    electric_data = readers.read_electric_hourly(electric_filename)
    # convert from hourly to daily
    electric_data = electric_data.resample('D').sum()
    # don't scale by 1.1 for including Scotland because only got generation for
    # England and Wales
    demand = electric_data['ENGLAND_WALES_DEMAND'] * 1.1
    if max_demand < demand.sum():
        max_demand = demand.sum()
    demand_years[year] = demand
    # extract wind
    wind = electric_data['EMBEDDED_WIND_GENERATION']
    if max_wind < wind.sum():
        max_wind = wind.sum()
    wind_years[year] = wind
    # extract solar
    pv = electric_data['EMBEDDED_SOLAR_GENERATION']
    if max_pv < pv.sum():
        max_pv = pv.sum()
    pv_years[year] = pv

# concatonate the years of demand.
demand_original = pd.concat(demand_years[year] for year in years)
ndays = len(demand_original)

# calculate average power
total_demand = demand_original.sum()
average_power = total_demand / (ndays * 24 * 1000.0 )
print("Average Power: {} Gw ".format(average_power))

# Normalise the original demand data
demand_original = demand_original / demand_original.sum()

print(demand_original)
ndays = len(demand_original)
print("Read data for {} years {} days".format(len(years), ndays) )

# shift each years demand so that they have the same annual demand
for year in years:
    dy = demand_years[year]
    print("Year {} sum {} max_demand {} days {}".format(year, dy.sum(), max_demand, len(dy) ) )
    dy = dy + ( (dy.sum() - max_demand) / len(dy) )
    demand_years[year] = dy

# concatonate the shifted years of demand.
demand = pd.concat(demand_years[year] for year in years)
# Normalise the shifted demand data
demand = demand / demand.sum()

print('demand_original')
print(demand_original)
print('demand')
print(demand)

# plot the original and shifted demand
demand_original.plot(label='original load')
demand.plot(label='adjusted load')
plt.title('True and adjusted load profiles')
plt.xlabel('Year')
plt.ylabel('Normalised load')
plt.legend(loc='upper right')
plt.show()

# Normalise each years generation (so they have the same annual generation)
# and calculate annual generation ratios
pv_load = []
wind_load = []
for year in years:
    wind_sum = wind_years[year].sum()
    wind_load.append(wind_sum)
    wind_years[year] = wind_years[year] / wind_sum
    pv_sum = pv_years[year].sum()
    pv_load.append(pv_sum)
    pv_years[year] = pv_years[year] / pv_sum



results = { 'wind' : wind_load, 'pv' : pv_load }
df_load = pd.DataFrame(data=results,index=years)
print(df_load)
df_load['wind'].plot(label='wind')
df_load['pv'].plot(label='pv')
plt.title('Average annual generation')
plt.xlabel('Year')
plt.ylabel('Average annual generation')
plt.legend(loc='lower right')
plt.show()

#  concat the PV and Solar years together.
wind = pd.concat(wind_years[year] for year in years)
pv = pd.concat(pv_years[year] for year in years)

# Normalise again
# (becaue otherwise the total will add up to the number of years and not 1! )
wind = wind / wind.sum()
pv = pv / pv.sum()

# plot wind and solar generation of the 5 years
wind.plot(label='wind')
pv.plot(label='pv')
plt.title('Normalised generation')
plt.xlabel('Day of the year')
plt.ylabel('Normalised generation')
plt.legend(loc='upper right')
plt.show()

# energy deficit plots for 3 examples

supply = wind * 0.57 + pv * 0.43
net = demand - supply
store_size, hist_mix = storage(net)
supply = wind
net = demand - supply
store_size, hist_wind = storage(net)
supply = pv * 0.43
net = demand - supply
store_size, hist_pv = storage(net)

# multiply by ndays to convert to days storage.

plt.plot(hist_mix * ndays, label='wind 57% PV 43%')
plt.plot(hist_wind * ndays, label='wind only')
plt.plot(hist_pv * ndays, label='pv only')
plt.title('Energy deficit')
plt.xlabel('Day of the year')
plt.ylabel('Energy deficit (days)')
plt.legend(loc='lower left')
plt.show()

# calculate storage at grid of Pv and Wind capacities for unit efficiency
df= storage_grid(demand, wind, pv, Lw, Ls, 1.0)
print(df)
print("Max storage {} Min Storage {}".format(df['storage'].max(), df['storage'].min()) )

ax2 = df.plot.scatter(x='f_wind',
                      y='f_pv',
                      c='storage',
                      colormap='viridis')
plt.show()

# plot minimum storage line
ax = df_min.plot(x='Pw', y='Ps',label='minimum generation')

# calcuate constant storage line for 40 days and plot
storage_40 = storage_line(df,-40.0)
storage_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days')

# calcuate constant storage line for 25 days and plot
storage_25 = storage_line(df,-25.0)
storage_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days')

plt.title('Storage unit charge efficiency')
plt.xlabel('Wind')
plt.ylabel('Solar PV')
plt.show()

print('Thing    mean     total')
print('demand   {}       {}   '.format(demand.mean(), demand.sum()))
print('pv   {}       {}   '.format(pv.mean(), pv.sum()))
print('wind   {}       {}   '.format(wind.mean(), wind.sum()))


# plot minimum storage line
ax = df_min.plot(x='Pw', y='Ps',label='minimum generation')
# calculate storage at grid of Pv and Wind capacities for 75% efficiency
df75= storage_grid(demand, wind, pv, Lw, Ls, 0.75)
# calculate storage at grid of Pv and Wind capacities for 85% efficiency
df85= storage_grid(demand, wind, pv, Lw, Ls, 0.85)

# calcuate constant storage line for 40 days and plot
storage_40_75 = storage_line(df75,-40.0)
storage_40_75.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days at 75%')

# calcuate constant storage line for 25 days and plot
storage_25_75 = storage_line(df75,-25.0)
storage_25_75.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days at 75%')

# calcuate constant storage line for 40 days and plot
storage_40_85 = storage_line(df85,-40.0)
storage_40_85.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days at 85%')

# calcuate constant storage line for 25 days and plot
storage_25_85 = storage_line(df85,-25.0)
storage_25_85.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days at 85%')

plt.title('Storage 75% and 85% charge efficiency')
plt.xlabel('Wind')
plt.ylabel('Solar PV')
plt.show()
