# python script to validate the updated heat and cop with ERA5 etc
# as used in R squared investigations and residuals plots.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
# custom code
import stats

def read_mycsv(filename):
    when2heat = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True, usecols=['time','space','water'] )
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
    print(filename, year)
    # read when2heat demand for GB space total
    demand = read_mycsv(filename)

    print('Total space {} water {}'.format(demand['space'].sum(), demand['water'].sum() ) )
# wrongly based on fuel energy not heat demand but in the IECSF20 paper
#   space_that_is_gas = {"2018" : 0.72, "2017" : 0.72, "2016" : 0.72}
#   water_that_is_gas = {"2018" : 0.81, "2017" : 0.81, "2016" : 0.81}
# new corrected values
    # 
    space_that_is_gas = {"2018" : 0.70, "2017" : 0.70, "2016" : 0.70}
    water_that_is_gas = {"2018" : 0.79, "2017" : 0.78, "2016" : 0.79}
    #
    # 70% of the space heat demand comes from gas
    demand['space'] = demand['space'] * space_that_is_gas[year]
    # 79% of the water heat demand comes from gas
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
    total_space = space_daily.sum()
    total_water = water_daily.sum()
    space_and_water = space_daily + water_daily
    print('Space {} Water {} Total {}'.format(total_space, total_water, space_and_water.sum()))
    return space_and_water

# scale a time series to a desired total
def scale_to(series, total):
    new_series = series * ( total / series.sum() )
    return new_series

# main program

# process command line

parser = argparse.ArgumentParser(description='Compare heat time series methods.')
parser.add_argument('--gas', action="store", dest="gas", help='Method of gas heat demand a=all temp, n=no temp, c=compromise', default='c' )
parser.add_argument('--year', action="store", dest="year", help='Year', default='2018' )
args = parser.parse_args()
gas_method = args.gas
weather_year = args.year
reference_year = args.year
# 

hdd155_filename = "/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef{}Weather{}I-Sbdew.csv".format(weather_year,reference_year,weather_year);
bdew_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef{}Weather{}I-Bbdew.csv'.format(weather_year,reference_year,weather_year);
hdd128_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef{}Weather{}I-Hbdew.csv'.format(weather_year,reference_year,weather_year);
watson_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef{}Weather{}I-Wbdew.csv'.format(weather_year,reference_year,weather_year);
# ERA5
#hdd155_filename = "/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather20185-Sbdew.csv"
#bdew_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather20185-Rbdew.csv'
#hdd128_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather20185-Hbdew.csv'
#watson_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/2018/GBRef2018Weather20185-Wbdew.csv'
# gas energy in GWh
gas_filename = '/home/malcolm/uclan/data/GasLDZOfftakeEnergy' + weather_year + '.csv'
# winter_date = year + '-02-02'
# summer_date = year + '-06-02'
space_and_water_gas = {"2018" : 388.7, "2017" : 365.36, "2016" : 377.7}

# get aggregated daily space and water from when2heat.
space_and_waterR = get_heat(bdew_filename, reference_year)
space_and_waterW = get_heat(watson_filename, reference_year)
space_and_waterH = get_heat(hdd128_filename, reference_year)
space_and_waterS = get_heat(hdd155_filename, reference_year)

print('Annual heat BDEW {} Watson {} HDD128 {} HDD155 {}'.format(space_and_waterR.sum(), space_and_waterW.sum(), space_and_waterH.sum(), space_and_waterS.sum()) )

# fix watson from regression
# wtotal = space_and_waterW.sum()
# g = -0.1288
# c = 0.09
# space_and_waterW = space_and_waterW * ( 1 - g ) - c
# g2 = 0.0314
# c2 = 0.01
# space_and_waterW = space_and_waterW * ( 1 - g2 ) - c2
# space_and_waterW = space_and_waterW * (wtotal / space_and_waterW.sum() )

# fix hdd15.5 from regression
# htotal = space_and_waterS.sum()
# g = 0.0865
# c = -0.05
# space_and_waterS = space_and_waterS * ( 1 - g ) - c
# g2 = -0.0536
# c2 = 0.01
# space_and_waterS = space_and_waterS * ( 1 - g2 ) - c2
# space_and_waterS = space_and_waterS * (htotal / space_and_waterS.sum() )

# read historic gas demand
gas = read_gas(gas_filename)

# Convert gas energy from kWh to TWh
gas_energy = gas * (10 ** -9)
total_gas = gas_energy.sum()
unknown_gas = total_gas - space_and_water_gas[reference_year]
# avoid it going negative
gas_energy_smallest = gas_energy.nsmallest(1)[0]
non_heat_gas_per_day = min(gas_energy_smallest, unknown_gas/365.0)
nvalues = len(gas_energy.index)
print('total_gas: {0:.2f} number of values: {1:6d} unknown gas: {2:.2f} non_heat_gas_per_day {3:.2f} space and water gas {4:.2f} '. format(total_gas, nvalues, unknown_gas, non_heat_gas_per_day, space_and_water_gas[reference_year]))

# gas energy by scaling assuming all the unknown is temperature dependent
gas_energy_temp = scale_to(gas_energy, space_and_water_gas[reference_year])
# gas energy by assuming all the unknown is split evenly amongst the days.
gas_energy_const = scale_to(gas_energy - non_heat_gas_per_day, space_and_water_gas[reference_year] )
# gas energy assuming a compromise
gas_energy_comp = scale_to((gas_energy - ( non_heat_gas_per_day * 0.5 ) ), space_and_water_gas[reference_year] )
# scale so we still have the same total
# print(gas_energy.head(7))
# print('GAS ENERGY SMALLEST')
# print(gas_energy.nsmallest())
# print('NEW GAS TOTAL')
# print(gas_energy.sum())

# account for boiler efficiency of 80% consistent with calc for annual heat demand gas
gas_energy_comp = gas_energy_comp * 0.8
gas_energy_temp = gas_energy_temp * 0.8
gas_energy_const = gas_energy_const * 0.8

# set according to the command line arg
if gas_method == 'c':
    gas_energy = gas_energy_comp
else:
    if gas_method == 'a':
        gas_energy = gas_energy_temp
    else:
        # (n) method in the IECSF20 paper 
        gas_energy = gas_energy_const

print('Gas method {} heat energy total {}'.format(gas_method, gas_energy.sum() ) )

# compute R2 and correlation

stats.print_stats_header()
stats.print_stats(space_and_waterR, gas_energy, 'BDEW', 2, True)
stats.print_stats(space_and_waterW, gas_energy, 'Watson', 1, True)
stats.print_stats(space_and_waterH, gas_energy, 'HDD 12.8', 1, True)
stats.print_stats(space_and_waterS, gas_energy, 'HDD 15.5  ', 1, True)

# output plots

gas_energy_comp.plot(label='Heat Demand Gas (compromise)', color='blue')
gas_energy_const.plot(label='Heat Demand Gas (not temp)', color='blue', style='--')
gas_energy_temp.plot(label='Heat Demand Gas (all temp)', color='blue', style='.')
#gas_energy_temp.plot(label='Heat Demand Gas', color='blue')
space_and_waterR.plot(label='Heat Demand BDEW', color='red')
space_and_waterW.plot(label='Heat Demand Watson', color='green')
space_and_waterH.plot(label='Heat Demand HDD 12.8', color='purple')
space_and_waterS.plot(label='Heat Demand HDD 15.5', color='orange')
plt.title('Comparison of Heat Demand Methods')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Heat Demand (TWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

# monthly nRMSE

stats.monthly_stats_header()
m_b = stats.monthly_stats(space_and_waterR, gas_energy, 'BDEW')
m_w = stats.monthly_stats(space_and_waterW, gas_energy, 'Watson')
m_h = stats.monthly_stats(space_and_waterH, gas_energy, 'HDD 12.8')
m_s = stats.monthly_stats(space_and_waterS, gas_energy, 'HDD 15.5')

plt.plot(range(12), m_b, label='BDEW')
plt.plot(range(12), m_w, label='Watson')
plt.plot(range(12), m_h, label='HDD 12.8')
plt.plot(range(12), m_s, label='HDD 15.5')
plt.xlabel('Month of the year', fontsize=15)
plt.ylabel('nRMSE', fontsize=15)
plt.title('Monthly variation in nRMSE')
plt.legend(loc='upper right', fontsize=15)
plt.show()




# load duration curves

gas_energy_const.sort_values(ascending=False, inplace=True)
gas_energy_temp.sort_values(ascending=False, inplace=True)
gas_energy_comp.sort_values(ascending=False, inplace=True)
space_and_waterR.sort_values(ascending=False, inplace=True)
space_and_waterW.sort_values(ascending=False, inplace=True)
space_and_waterH.sort_values(ascending=False, inplace=True)
space_and_waterS.sort_values(ascending=False, inplace=True)

# gas_energy.plot(label='Gas Energy', use_index=False, style='--', fontsize=18)
# fontsize here only affects the numbers
gas_energy_comp.plot(label='Heat Demand Gas (compromise)', use_index=False, color='blue')
gas_energy_const.plot(label='Heat Demand Gas (not temp)', use_index=False, style='--', color='blue')
gas_energy_temp.plot(label='Heat Demand Gas (all temp)', use_index=False, style='.', color='blue')
#gas_energy_temp.plot(label='Heat Demand Gas', use_index=False, color='blue')
space_and_waterR.plot(label='Heat Demand BDEW', use_index=False, color='red')
space_and_waterW.plot(label='Heat Demand Watson', use_index=False, color='green')
space_and_waterH.plot(label='Heat Demand HDD 12.8', use_index=False, color='purple')
space_and_waterS.plot(label='Heat Demand HDD 15.5', use_index=False, color='orange')
plt.title('Comparison of Heat Demand Methods')
plt.xlabel('Day sorted by demand', fontsize=15)
plt.ylabel('Daily Heat Demand (TWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.xticks(rotation=0)
plt.show()

print('Load duration curves')

stats.print_stats_header()
stats.print_stats(space_and_waterR, gas_energy, 'BDEW')
stats.print_stats(space_and_waterW, gas_energy, 'Watson')
stats.print_stats(space_and_waterH, gas_energy, 'HDD 12.8')
stats.print_stats(space_and_waterS, gas_energy, 'HDD 15.5')

