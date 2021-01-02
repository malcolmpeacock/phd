# compare 2018 equest model with gas and my crude spreadsheet of all UK
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import stats

# read equest file

def dt_parse(y,m,d,h):
    h = str(int(h) - 1)
    dt = '{}-{}-{}T{}:00:00Z'.format(y, m.zfill(2), d.zfill(2), h.zfill(2))
    return dt

def read_eqeust(filename):

    equest = pd.read_csv(filename, header=0, sep=',', parse_dates={'datetime': [0,1,2,3]}, date_parser=dt_parse, index_col='datetime', squeeze=True)
    # create a datetime index so we can plot
    equest.index = pd.DatetimeIndex(pd.to_datetime(equest.index).date)
    # resample from hourly to daily.
    total_before = equest['space'].sum()
    equest = equest.resample('D').sum()
    # output raw total as a test
    total_after = equest['space'].sum()
    print('EQuest total read {} after {}'.format(total_before, total_after))
    # Convert BTU to KWh
    equest = equest * 0.000293071
    # Scale up for total UK number of dwellings
    equest = equest * 23950.0 * 1000.0
    # convert to TWh
    equest = equest * (10 ** -9)
    print(equest.head(7))
    return equest


def read_thermal(filename):
    thermal = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    # create a datetime index so we can plot
    thermal.index = pd.DatetimeIndex(pd.to_datetime(thermal.index,utc=True).date)
#   print(thermal.head(7))
    # convert to TWh
    thermal = thermal * (10 ** -6)
    return thermal

def read_gas(filename):
    gas = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y').replace(tzinfo=pytz.timezone('Europe/London')), index_col=0, squeeze=True, usecols=[1,3] )
    gas = gas.astype('float')
    # reverse it (december was first! )
    gas = gas.iloc[::-1]
    # create a datetime index so we can plot
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

# main program
year = '2018'

# gas energy in GWh
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + year + '.csv'
# eq_filename = '/home/malcolm/uclan/data/EQuestMPTest2BaselineDesignHourlyResults2.csv'
eq_filename = '/home/malcolm/uclan/data/EquestMPTest2BaselineDesign2018HourlyResults2.csv'
thermal_filename = '/home/malcolm/uclan/tools/python/output/thermal/test2018.csv'
space_and_water_gas = {"2018" : 388.7, "2017" : 365.36, "2016" : 377.7}
space_gas = {"2018" : 191.4, "2017" : 244.1, "2016" : 249.9}

thermal = read_thermal(thermal_filename)
total_thermal = thermal.sum()

eq = read_eqeust(eq_filename)
# eq_space = eq['space'] + eq['water']
eq_space = eq['space']
total_eq = eq_space.sum()

# read historic gas demand
gas = read_gas(gas_filename)

# scale gas to convert from mcm to twh
# 11 GWh is 1 mscm (millions of standard cubic metres)
# gas_energy = gas * 11.0 * 0.001

# Convert gas energy from kWh to TWh
gas_energy = gas * (10 ** -9)
total_gas = gas_energy.sum()
non_heat_gas = total_gas - space_gas[year]
nvalues = len(gas_energy.index)
print('total_gas: {0:.2f} number of values: {1:6d} non heat gas: {2:.2f} '. format(total_gas, nvalues, non_heat_gas))

# eq_space = eq_space * total_gas / total_eq

# subtracting the non heat gas and divide evenly amongst the days.
# assuming that the non-gas energy is constant.
# (if its not, this could go negative)
gas_energy = gas_energy - (non_heat_gas / 365.0)
# print(gas_energy.head(7))
# print('GAS ENERGY SMALLEST')
# print(gas_energy.nsmallest())

# account for boiler efficiency of 90%
# (this is inconsistent as 80% is used for the EU buildings figures)
gas_energy = gas_energy * 0.8

total_gas_energy = gas_energy.sum()
print('Total Thermal {} Eq {} Gas {}'.format(total_thermal,total_eq,total_gas_energy))
# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(eq_space, gas_energy, 'EQuest')
stats.print_stats(thermal, gas_energy, 'Thermal XL')

# output plots

gas_energy.plot(label='Gas Energy')
eq_space.plot(label='Equest')
thermal.plot(label='Thermal XL')
plt.title('Comparison of Heat Demand Methods')
plt.xlabel('Day of the year')
plt.ylabel('Heat Demand (MWh)')
plt.legend(loc='upper right')
plt.show()

# load duration curves

gas_energy.sort_values(ascending=False, inplace=True)
eq_sorted = eq_space.sort_values(ascending=False)
thermal.sort_values(ascending=False, inplace=True)

gas_energy.plot(label='Gas Energy', use_index=False)
eq_sorted.plot(label='Equest', use_index=False)
thermal.plot(label='Thermal XL', use_index=False)
plt.title('Comparison of Heat Demand Methods')
plt.xlabel('Time sorted (d)')
plt.ylabel('Heat Demand')
plt.legend(loc='upper right')
plt.show()

print('Load duration curves')

stats.print_stats_header()
stats.print_stats(eq_space, gas_energy, 'Equest')
stats.print_stats(thermal, gas_energy, 'Thermal XL')
