# Look at the impact of 50% heat pumps on the electricity demand
# using 40 years weather
# and justify using the 2018 electricity series 

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy.stats import wasserstein_distance
from icecream import ic 

# custom code
import stats
import readers
import storage

# create regression line for plotting

def list_regression_line(x,y):
    nx = np.array(list(x))
    ny = np.array(list(y))
    return regression_line(nx,ny)

def regression_line(nx,ny):
    rmodel = sm.OLS(ny, sm.add_constant(nx))
    residual_results = rmodel.fit()
#   print(residual_results.summary())
    reg_const = residual_results.params[0]
    reg_grad = residual_results.params[1]
    x_reg = np.array([nx.min(),nx.max()])
    y_reg = reg_grad * x_reg + reg_const
    print('Regression grad {} const {}'.format(reg_grad, reg_const) )
    return x_reg, y_reg

def hydrogen_boiler(heat, efficiency):
    hydrogen = heat / efficiency
    return hydrogen
#
#  mod_electric_ref  - reference electricity with heat removed
#              wind  - wind capacity factor time series
#                pv  - pv capacity factor time series
# percent_heat_pumps - percentage of heating from heat pumps
#              years - list of years to do the analysis for
def supply_and_storage(mod_electric_ref, wind, pv, percent_heat_pumps, years, plot, hourly, ref_temp):
   
    # factor to normalise by.
    # If doing daily is the max daily demand so that storage is relative to 
    # peak daily demand energy.
    normalise_factor = mod_electric_ref.max()
    total_demand = 0
    # ordinary year
    ordinary_year = mod_electric_ref.values
    # leap year
    # find doy for feb 28th ( 31 days in jan )
    feb28 = 31 + 28
    if hourly:
        feb28 = mod_electric_ref['2018-02-28'].values
        mar1 = mod_electric_ref['2018-03-01'].values
        feb29 = np.add(feb28, mar1) * 0.5
        leap_year = np.concatenate([mod_electric_ref['2018-01-01 00:00' : '2018-02-28 23:00'].values, feb29,  mod_electric_ref['2018-03-01 00:00' : '2018-12-31 23:00'].values])
    else:
        feb29 = (ordinary_year[feb28-1] + ordinary_year[feb28]) * 0.5
        feb29a = np.array([feb29])
        leap_year = np.concatenate([ordinary_year[0:feb28], feb29a, ordinary_year[feb28:] ] )


    demand_years=[]
    total_heat_demand_years={}
    mean_temp_years={}
    monthly_temp_years={}
    monthly_sd_years={}
    yearly_wd={}
    # for each weather year ...
    for year in years:
        print('Creating demand for {}'.format(year))

        # read weather year electric heat for ref year
#       demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-Bbdew.csv'.format(year, args.reference)
        # rhpp heat pump profile
        demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-Brhpp.csv'.format(year, args.reference)
        demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
        # store total heat demand
        total_heat_demand_years[year] = demand['heat'].sum() * 1e-6
        mean_temp_years[year] = demand['temperature'].mean()
        monthly_temp_years[year] = demand['temperature'].resample('M').mean().values
        monthly_sd_years[year] = demand['temperature'].resample('M').std().values
        # wasserstein distance between temperatures
        yearly_wd[year] = wasserstein_distance(ref_temp, demand['temperature'])
        #  account for leap years.
        #  Need to create a new df with the index same as heat_weather
        #  then create a 29th of Feb by interpolation between 28th and
        #  1st March and then set the data values into empty DF
        if not hourly:
            demand = demand.resample('D').sum()
        heat_weather = demand['heat']
        # use leap year data or ordinary year data
        if calendar.isleap(year):
            electric_ref = pd.Series(leap_year, index=heat_weather.index)
        else:
            electric_ref = pd.Series(ordinary_year, index=heat_weather.index)
        if percent_heat_pumps > 0:
            # electric heat series
            electric_heat = demand['electricity'] * percent_heat_pumps
            electric_ref = electric_ref + electric_heat

        # normalise and add to the list
        demand_years.append( electric_ref / normalise_factor)
    # concantonate the demand series
    all_demand = pd.concat(demand_years[year] for year in range(len(years)) )
    if plot:
        all_demand.plot(color='blue', label='Electric with {} heating'.format(percent_heat_pumps))
        pv.plot(color='red', label='Solar PV')
        wind.plot(color='green', label='Wind')
        plt.title('Daily Electricity demand and generation')
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Normalised daily Electricity', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()
    #
    print('Thing    mean     total')
    print('demand   {}       {}   '.format(all_demand.mean(), all_demand.sum()))
    print('pv   {}       {}   '.format(pv.mean(), pv.sum()))
    print('wind   {}       {}   '.format(wind.mean(), wind.sum()))
#   print(total_heat_demand_years)
#   print(mean_temp_years)
   
    if plot:
        x=total_heat_demand_years.keys()
        y=total_heat_demand_years.values()
        plt.scatter(x,y)
        plt.title('Heat demand 2018 would have had for different years weather')
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Heat Demand (Twh) per year', fontsize=15)
        x_reg, y_reg = list_regression_line(x,y)
        print('Heat demand regression start {} end {}'.format(y_reg[0], y_reg[1]) )
        plt.plot(x_reg, y_reg, color='red')
        plt.show()

        x=mean_temp_years.keys()
        y=mean_temp_years.values()
        plt.scatter(x,y)
        plt.title('Mean population weighted temperature against year')
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Mean population weighted temperature (degrees C)', fontsize=15)
        x_reg, y_reg = list_regression_line(x,y)
        print('Temerature regression start {} end {}'.format(y_reg[0], y_reg[1]) )
        plt.plot(x_reg, y_reg, color='red')
        plt.show()

        # monthly mean temperature over the years
        year_var=monthly_temp_years.keys()
        for month in range(12):
            month_values=[]
            for year in years:
                month_values.append(monthly_temp_years[year][month])
#           print('Month {} len {}'.format(month, len(month_values) ) )
            x_reg, y_reg = list_regression_line(year_var,month_values)
            plt.plot(x_reg, y_reg, label=calendar.month_abbr[month+1])
        plt.legend(loc='upper right')
        plt.title('Regression lines of mean monthly temperature for different years')
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Mean population weighted temperature (degrees C)', fontsize=15)
        plt.show()

        # monthly standard deviation of temperature over the years
        for month in range(12):
            std_values=[]
            for year in years:
                std_values.append(monthly_sd_years[year][month])
#           print('Month {} len {}'.format(month, len(month_values) ) )
            x_reg, y_reg = list_regression_line(year_var,std_values)
            plt.plot(x_reg, y_reg, label=calendar.month_abbr[month+1])
        plt.legend(loc='upper right')
        plt.title('Regression lines of standard deviation of monthly temperature for different years')
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Mean population weighted temperature (degrees C)', fontsize=15)
        plt.show()

    # look at the storage history for 2 wind and 1 pv which is what we have now
    # look at the storage history for 3 wind and 1 pv
    f_wind = 1.8
    f_pv = 1.0
    supply = wind * f_wind  +  pv * f_pv
    net = all_demand - supply
#   print('NET')
#   print(net)

    #  calculate how much storage we need
    store_hist = storage.storage(net, 0.8)
#   print(store_hist)
    store_size = store_hist.min()
    storage_days = round(store_size * -1.0)
    # if minimum value was the last one then store is just emptying more and
    # more so we didn't find out the size
    last_one = store_hist.iat[-1]
    if store_size == store_hist.iat[-1] or storage_days>200:
        storage_days = 200
    if storage_days <0.0:
        storage_days = 0.0
    print('Storage for {} wind {} pv size {} is {} days. last_one {}'.format(f_wind, f_pv, store_size, storage_days, last_one) )
    period_hist = store_hist + storage_days
    period_hist = period_hist.clip(0.0, storage_days)
#   print(period_hist)

    if plot:
        period_hist.plot()
        plt.title('Daily store size: wind {} pv {} days {} '.format(f_wind, f_pv, storage_days))
        plt.xlabel('day', fontsize=15)
        plt.ylabel('Store size in days', fontsize=15)
        plt.show()

        yearly_start = store_hist.resample('Y').first()
        print(yearly_start)
        yearly_max = store_hist.resample('Y').max()
        print(yearly_max)
        yearly_min = store_hist.resample('Y').min()
        print(yearly_min)
#       yearly_diff = yearly_max - yearly_min
        yearly_diff = yearly_start - yearly_min
        print(yearly_diff)
        yearly_diff.plot(color='green', label='yearly store size')
        plt.legend(loc='upper right')
        plt.title('Store size at the end of each year: {} wind to {} solar'.format(f_wind, f_pv) )
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Store size in days', fontsize=15)
        plt.show()

        # plot the wasserstein distances
        plt.scatter(list(year_var), yearly_wd.values())
        plt.title('Wasserstien distance of temperature distributions from 2018' )
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Wasserstien distance', fontsize=15)
        plt.show()
        

    # calculate storage at grid of Pv and Wind capacities for
    # hydrogen efficiency TODO need a reference
    df= storage.storage_grid(all_demand, wind, pv, 0.8, hourly)
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

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew_resistive.csv'.format(args.reference) 
ref_resistive = readers.read_copheat(demand_filename, ['electricity', 'temperature'])
ref_resistive_heat = ref_resistive['electricity']
ref_temperature = ref_resistive['temperature']

#  To remove existing space and water heating from the electricity demand time 
#  series for the reference year - subtract the resistive heat series

mod_electric_ref = electric_ref - (ref_resistive_heat * heat_that_is_electric)
daily_electric_ref = mod_electric_ref.resample('D').sum()

# plot reference year electricity

if args.plot:
    daily_electric_ref.plot(color='blue', label='Historic Electric without heating {}'.format(args.reference))
    plt.title('Reference year Daily Electricity with heat removed')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Electricity (Mwh) per day', fontsize=15)
#plt.legend(loc='upper right')
    plt.show()

# weather years from the start to 2019
# ( need to download more ninja to get up to 2020 )
last_weather_year = 2019
years = range(args.start, last_weather_year+1)
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
#   print(wind)
#   print(pv)

    # daily plot
    wind_daily = wind.resample('D').mean()
    pv_daily = pv.resample('D').mean()
    wind_daily.plot(color='blue', label='Ninja wind generation')
    pv_daily.plot(color='red', label='Ninja pv generation')
    plt.title('Wind and solar generation from ninja')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Electricity generation capacity factor per day', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

    # yearly plot
    wind_yearly = wind.resample('Y').mean()
    pv_yearly = pv.resample('Y').mean()
    wind_yearly.plot(color='blue', label='Ninja wind generation')
    pv_yearly.plot(color='red', label='Ninja pv generation')
    year_var=wind_yearly.index.year.values.astype(np.float32)
    values_var = wind_yearly.values
    x_reg, y_reg = regression_line(year_var,values_var)
    print('Wind regression start {} end {}'.format(y_reg[0], y_reg[1]) )
    wind_reg = pd.Series(y_reg, index=[wind_yearly.index.min(),wind_yearly.index.max()])
    wind_reg.plot(color='green', label='Ninja wind regression')
    values_var = pv_yearly.values
    x_reg, y_reg = regression_line(year_var,values_var)
    print('PV regression start {} end {}'.format(y_reg[0], y_reg[1]) )
    pv_reg = pd.Series(y_reg, index=[wind_yearly.index.min(),wind_yearly.index.max()])
    pv_reg.plot(color='orange', label='Ninja pv regression')
    plt.title('Wind and solar generation from ninja')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Electricity generation capacity factor per year', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

    # yearly plot PV only
    pv_yearly.plot(color='red', label='Ninja pv generation')
    plt.title('Solar generation from ninja')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Electricity generation capacity factor per year', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

if args.hourly:
    print('Using hourly time series')
else:
    print('Using daily time series')
    mod_electric_ref = mod_electric_ref.resample('D').sum()
    wind = wind.resample('D').mean()
    pv = pv.resample('D').mean()

df = supply_and_storage(mod_electric_ref, wind, pv, 0.0, years, args.plot, args.hourly, ref_temperature)
print("Zero heat pumps: Max storage {} Min Storage {}".format(df['storage'].max(), df['storage'].min()) )

df5 = supply_and_storage(mod_electric_ref, wind, pv, 0.5, years, args.plot, args.hourly, ref_temperature)
print("50% heat pumps: Max storage {} Min Storage {}".format(df5['storage'].max(), df5['storage'].min()) )

if args.plot:
    df.plot.scatter(x='f_wind', y='f_pv', c='storage', colormap='viridis')
#   plt.colorbar(label='Storage (days)')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('Storage in days for different proportions of wind and solar (no heat pumps).')
    plt.show()

    df5.plot.scatter(x='f_wind', y='f_pv', c='storage', colormap='viridis')
#   plt.colorbar(label='Storage (days)')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('Storage in days for different proportions of wind and solar (50% heat pumps).')
    plt.show()

#   Lw = 0.28
#   Ls = 0.116
#   Lw = 1 / wind.mean()
#   Ls = 1 / pv.mean()
    Lw = wind.mean()
    Ls = pv.mean()
#   ic(Lw)
#   ic(Ls)

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
    storage_40 = storage.storage_line(df,40.0)
#   storage_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days. 2018 system')
    ax = storage_40.plot(x='Pw',y='Ps',label='storage 40 days. 2018 system')

    # calcuate constant storage line for 25 days and plot
    storage_25 = storage.storage_line(df,25.0)
    storage_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days. 2018 system')

    # calcuate constant storage line for 40 days and plot
    storage5_40 = storage.storage_line(df5,40.0)
    storage5_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days. 2018 system with 50% heating by heat pumps')

    # calcuate constant storage line for 25 days and plot
    storage5_25 = storage.storage_line(df5,25.0)
    storage5_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days, 2018 system with 50% heating by heat pumps')

    # calcuate constant storage line for 2 days and plot
    storage5_2 = storage.storage_line(df5,2.0)
    storage5_2.plot(x='Pw',y='Ps',ax=ax,label='storage 2 days, 2018 system with 50% heating by heat pumps')

    plt.title('Constant storage lines with and without heat pumps {} to {}'.format(args.start, last_weather_year) )
    plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
    plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
    plt.show()

## TODO wasserstein distance of heat demand to see if it changes over the
##      years.
## and for wind speed and solar irradiance.


