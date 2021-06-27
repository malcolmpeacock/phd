# Create a 2050 electric time series from:
#  reference year heat demand
#  weather year heat demand
#  reference year electricity time series
# And assumption from various sources, such as FES 2019

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse

# custom code
import stats
import readers
import storage

def hydrogen_boiler(heat, efficiency):
    hydrogen = heat / efficiency
    return hydrogen
#
#  mod_electric_ref  - reference electricity with heat removed
#              wind  - wind capacity factor time series
#                pv  - pv capacity factor time series
# percent_heat_pumps - percentage of heating from heat pumps
#              years - list of years to do the analysis for
def supply_and_storage(mod_electric_ref, wind, pv, percent_heat_pumps, years, plot, hourly):
   
    annual_demand = mod_electric_ref.sum()
    # TODO what about leap years? need to adjust number of days
    if hourly:
        annual_demand = annual_demand / 8760
    else:
        annual_demand = annual_demand / 365
    annual_demand = annual_demand * len(years)

    print('annual_demand {}'.format(annual_demand) )
    demand_years=[]
    # for each weather year ...
    for year in years:
        print('Creating demand for {}'.format(year))
        # read weather year electric heat for ref year
        demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-Sbdew.csv'.format(year, args.reference)
        demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
        if not hourly:
            demand = demand.resample('D').sum()
        heat_weather = demand['heat']
        # set the index on the electricity series to the weather year
        # TODO - account for leap years.
        #  Need to create a new df with the index same as heat_weather
        #  then create a 29th of Feb by interpolation between 28th and
        #  1st March and then set the data values into empty DF
        electric_ref = mod_electric_ref.copy()
        electric_ref.index = heat_weather.index
        if percent_heat_pumps > 0:
            # electric heat series
            electric_heat = demand['electricity'] * percent_heat_pumps
            electric_ref = electric_ref + electric_heat

        # normalise and add to the list
        demand_years.append( electric_ref / annual_demand)
    # concantonate the demand series
    all_demand = pd.concat(demand_years[year] for year in range(len(years)) )
    if plot:
        all_demand.plot(color='blue', label='Electric with {} heating'.format(percent_heat_pumps))
        pv.plot(color='red', label='Solar PV')
        wind.plot(color='green', label='Wind')
        plt.title('Daily Electricity demand and generation')
        plt.xlabel('Month', fontsize=15)
        plt.ylabel('Electricity (Mwh) per day', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()
    #
    print('Thing    mean     total')
    print('demand   {}       {}   '.format(all_demand.mean(), all_demand.sum()))
    print('pv   {}       {}   '.format(pv.mean(), pv.sum()))
    print('wind   {}       {}   '.format(wind.mean(), wind.sum()))
    # calculate storage at grid of Pv and Wind capacities for
    # hydrogen efficiency TODO need a reference
#   df= storage.storage_grid(all_demand, wind, pv, 0.8)
    df= storage.storage_grid(all_demand/365.0, wind/365.0, pv/365.0, 0.8)
    return df

# main program

# process command line
parser = argparse.ArgumentParser(description='Show the impact of heat pumps or hydrogen on different shares of wind and solar')
parser.add_argument('--start', action="store", dest="start", help='Start Year', type=int, default=2017 )
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--hourly', action="store_true", dest="hourly", help='Use hourly time series', default=False)
args = parser.parse_args()

# print arguments
print('Start year {} Reference year {} plot {} hourly {}'.format(args.start, args.reference, args.plot, args.hourly) )

# input assumptions for reference year
heat_that_is_electric = 0.06     # my spreadsheet from DUKES
heat_that_is_heat_pumps = 0.01   # greenmatch.co.uk, renewableenergyhub.co.uk

scotland_factor = 1.1    # ( Fragaki et. al )

# read historical electricity demand for reference year
# TODO - could we include the actual Scottish demand here?

demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + args.reference + '.csv'
demand_ref = readers.read_electric_hourly(demand_filename)
electric_ref = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor

# read reference year electric heat series based on purely resistive heating
# so that it can be removed from the reference year series. 
# TODO - recaluate this using BDEW method?

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Sbdew_resistive.csv'.format(args.reference) 
ref_resistive_heat = readers.read_demand(demand_filename)

#  To remove existing space and water heating from the electricity demand time 
#  series for the reference year - subtract the resistive heat series

mod_electric_ref = electric_ref - (ref_resistive_heat * heat_that_is_electric)
daily_electric_ref = mod_electric_ref.resample('D').sum()

# plot reference year electricity

if args.plot:
    daily_electric_ref.plot(color='blue', label='Historic Electric without heating {}'.format(args.reference))
    plt.title('Reference year Daily Electricity with heat removed')
    plt.xlabel('Month', fontsize=15)
    plt.ylabel('Electricity (Mwh) per day', fontsize=15)
#plt.legend(loc='upper right')
    plt.show()

years = range(args.start, 2019)
print(years)
# read ninja
ninja_start = str(years[0]) + '-01-01 00:00:00'
ninja_end = str(years[-1]) + '-12-31 23:00:00'
print(ninja_start, ninja_end)
# Ninja capacity factors for pv
ninja_filename_pv = '/home/malcolm/uclan/data/ninja/ninja_pv_country_GB_merra-2_corrected.csv'
# Ninja capacity factors for wind
ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_near-termfuture-merra-2_corrected.csv'

print('Loading ninja ...')
ninja_pv = readers.read_ninja_country(ninja_filename_pv)
ninja_wind = readers.read_ninja_country(ninja_filename_wind)

print('Extracting PV ...')
ninja_pv = ninja_pv[ninja_start : ninja_end]
pv = ninja_pv['national']

print('Extracting Wind ...')
ninja_wind = ninja_wind[ninja_start : ninja_end]
wind = ninja_wind['national']

print('Read PV {} Wind {} '.format(len(pv), len(wind) ) )

if args.plot:
    print(wind)
    print(pv)
    wind_daily = wind.resample('D').sum()
    pv_daily = pv.resample('D').sum()
    wind_daily.plot(color='blue', label='Ninja wind generation')
    pv_daily.plot(color='red', label='Ninja pv generation')
    plt.title('Wind and solar generation from ninja')
    plt.xlabel('Month', fontsize=15)
    plt.ylabel('Electricity (Mwh) per day', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

if args.hourly:
    print('Using hourly time series')
else:
    print('Using daily time series')
    mod_electric_ref = mod_electric_ref.resample('D').sum()
    wind = wind.resample('D').mean()
    pv = pv.resample('D').mean()

df = supply_and_storage(mod_electric_ref, wind, pv, 0.0, years, args.plot, args.hourly)
print("Zero heat pumps: Max storage {} Min Storage {}".format(df['storage'].max(), df['storage'].min()) )

df5 = supply_and_storage(mod_electric_ref, wind, pv, 0.5, years, args.plot, args.hourly)
print("50% heat pumps: Max storage {} Min Storage {}".format(df5['storage'].max(), df5['storage'].min()) )

if args.plot:
    ax2 = df.plot.scatter(x='f_wind', y='f_pv', c='storage', colormap='viridis')
    plt.show()

    Lw = 0.28
    Ls = 0.116
#   Lw = 1 / wind.mean()
#   Ls = 1 / pv.mean()

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

    # plot minimum generation line TODO
#   ax = df_min.plot(x='Pw', y='Ps',label='minimum generation')

    # calcuate constant storage line for 40 days and plot
    storage_40 = storage.storage_line(df,-40.0)
    print(storage_40)
    ax = storage_40.plot(x='Pw',y='Ps',label='storage 40 days. 2018 system')

    # calcuate constant storage line for 25 days and plot
    storage_25 = storage.storage_line(df,-25.0)
    storage_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days. 2018 system')

    # calcuate constant storage line for 40 days and plot
    storage5_40 = storage.storage_line(df5,-40.0)
    storage5_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days. 2018 system with 50% heating by heat pumps')

    # calcuate constant storage line for 25 days and plot
    storage5_25 = storage.storage_line(df5,-25.0)
    storage5_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days, 2018 system with 50% heating by heat pumps')

    plt.title('Constant storage lines with and without heat pumps')
    plt.xlabel('Wind ( capacity in proportion to nomalised generation)')
    plt.ylabel('Solar PV ( capacity in proportion to nomalised generation)')
    plt.show()