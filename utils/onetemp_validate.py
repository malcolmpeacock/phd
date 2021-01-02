# python script to validate the when2heat stuff.
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import stats

def read_mycsv(filename, year):
#   when2heat = pd.read_csv(filename, header=0, sep=';', decimal=',', parse_dates=[0], index_col=0, squeeze=True, usecols=['utc_timestamp','GB_heat_demand_space','GB_heat_demand_water'] )
    when2heat = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True, usecols=['time','space','water'] )
#   when2heat.columns = ['space','water']
    print(when2heat.head(7))
    print(when2heat.tail(7))
#   when2heat_year = when2heat.loc[year+'-01-01 00:00:00':year+'-12-31 23:00:00']
#   print(when2heat_year.head(7))
#   print(when2heat_year.tail(7))
#   return when2heat_year
    return when2heat

def read_gas(filename):
    gas = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y').replace(tzinfo=pytz.timezone('Europe/London')), index_col=0, squeeze=True, usecols=[1,3] )
    gas = gas.astype('float')
    # reverse it (december was first! )
    gas = gas.iloc[::-1]
    # get rid of time so we just have a date
    gas.index = pd.DatetimeIndex(pd.to_datetime(gas.index).date)
    # take the average of multiple values at same date. really should take
    # the recently amended one.
    # gas = gas.resample('D').mean()
    gas = gas.resample('D').mean()
    # get smallest values to check for missing data
    print('GAS: SMALLEST')
    print(gas.nsmallest())
    # replace zeros with NaN
    gas = gas.replace(0.0, float("NaN"))
    # replace missing values (NaN) by interpolation
    gas = gas.interpolate()
    # print(gas['2016-07-28'])
    return gas

def get_heat(filename, year):
    # read when2heat demand for GB space total
    demand = read_mycsv(filename, year)

    space_that_is_gas = {"2018" : 0.72, "2017" : 0.72, "2016" : 0.72}
    water_that_is_gas = {"2018" : 0.81, "2017" : 0.81, "2016" : 0.81}
    # 72% of the space heat demand comes from gas
    demand['space'] = demand['space'] * space_that_is_gas[year]
    # 81% of the water heat demand comes from gas
    demand['water'] = demand['water'] * water_that_is_gas[year]

    # sample the hourly for an example winter and summer day
    # winter_day = demand.loc[winter_date+' 00:00:00':winter_date+' 23:00:00']
    # print(winter_day)
    # winter_day.plot()
    # plt.show()
    # summer_day = demand.loc[summer_date+' 00:00:00':summer_date+' 23:00:00']
    # summer_day.plot()
    # plt.show()

    # aggregate to daily.
    space_daily = demand.resample('D')['space'].sum()
    # remove the time so we only have date
    space_daily.index = pd.DatetimeIndex(pd.to_datetime(space_daily.index).date)
    # convert to TWh
    space_daily = space_daily / 1000000.0

    # do the same for water.
    water_daily = demand.resample('D')['water'].sum()
    water_daily.index = pd.DatetimeIndex(pd.to_datetime(water_daily.index).date)
    water_daily = water_daily / 1000000.0

    # remove time from the date
    #space_daily.index = space_daily.index.dt.date
    total_space = space_daily.sum()
    print('total_space ' + str(total_space))
    print(space_daily.head(7))

    total_water = water_daily.sum()
    print('total_water ' + str(total_water))
    print(water_daily.head(7))
    space_and_water = space_daily + water_daily
    return space_and_water

# main program
hdd_filename15 = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018HDD15.5.csv'
onetemp_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018HDD15.5OneTemp.csv'
onetemp1d_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018HDD15.5OneTemp1day.csv'
year = '2018'
# gas energy in GWh
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + year + '.csv'
# winter_date = year + '-02-02'
# summer_date = year + '-06-02'
space_and_water_gas = {"2018" : 388.7, "2017" : 365.36, "2016" : 377.7}

# get aggregated daily space and water from when2heat.
space_and_water = get_heat(hdd_filename15, year)
space_and_waterS1 = get_heat(onetemp_filename, year)
space_and_waterS2 = get_heat(onetemp1d_filename, year)

# read historic gas demand
gas = read_gas(gas_filename)

# scale gas to convert from mcm to twh
# 11 GWh is 1 mscm (millions of standard cubic metres)
# gas_energy = gas * 11.0 * 0.001

# Convert gas energy from kWh to TWh
gas_energy = gas * (10 ** -9)
total_gas = gas_energy.sum()
non_heat_gas = total_gas - space_and_water_gas[year]
nvalues = len(gas_energy.index)
print('total_gas: {0:.2f} number of values: {1:6d} non heat gas: {2:.2f} '. format(total_gas, nvalues, non_heat_gas))

# subtracting the non heat gas and divide evenly amongst the days.
# assuming that the non-gas energy is constant.
# (if its not, this could go negative)
gas_energy = gas_energy - (non_heat_gas / 365.0)
# print(gas_energy.head(7))
print('GAS ENERGY SMALLEST')
print(gas_energy.nsmallest())

# account for boiler efficiency of 90%
# (this is inconsistent as 80% is used for the EU buildings figures)
gas_energy = gas_energy * 0.8

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(space_and_water, gas_energy, 'HDD 15.5  ')
stats.print_stats(space_and_waterS1, gas_energy, 'HDD 15.5 S1 ')
stats.print_stats(space_and_waterS2, gas_energy, 'HDD 15.5 S2 ')

# output plots

gas_energy.plot(label='Gas Energy')
space_and_water.plot(label='Heat Demand HDD 15.5')
space_and_waterS1.plot(label='Heat Demand HDD 15.5 S1')
space_and_waterS2.plot(label='Heat Demand HDD 15.5 S2')
plt.title('Comparison of Heat Demand Methods')
plt.xlabel('Day of the year')
plt.ylabel('Heat Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# load duration curves

gas_energy.sort_values(ascending=False, inplace=True)
space_and_water.sort_values(ascending=False, inplace=True)
space_and_waterS1.sort_values(ascending=False, inplace=True)
space_and_waterS2.sort_values(ascending=False, inplace=True)

gas_energy.plot(label='Gas Energy', use_index=False)
space_and_water.plot(label='Heat Demand HDD 15.5', use_index=False)
space_and_waterS1.plot(label='Heat Demand HDD 15.5 S1', use_index=False)
space_and_waterS2.plot(label='Heat Demand HDD 15.5 S2', use_index=False)
plt.title('Comparison of Heat Demand Methods')
plt.xlabel('Time sorted (d)')
plt.ylabel('Heat Demand')
plt.legend(loc='upper right')
plt.show()

print('Load duration curves')

stats.print_stats_header()
stats.print_stats(space_and_water, gas_energy, 'HDD 15.5')
stats.print_stats(space_and_waterS1, gas_energy, 'HDD 15.5 S1')
stats.print_stats(space_and_waterS2, gas_energy, 'HDD 15.5 S2')
