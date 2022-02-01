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

# electric vehicle time series
ev_annual_energy = 95.83  # annual ev energy TWh fes Net Zero 2050
ev_annual_energy = 93.38  # percentage adjusted for 2018 with all ev

def ev_series(temperature, annual_energy):
    daily_energy = annual_energy * 1000000 / 365.0
    # at 14.4 kWh/100 km, gives
    daily_range_km = daily_energy * 100.0 / 14.4
    # calculate ev profile
    daytime_temp = 19.0
    new_daily_temp = 0.0
    ev = temperature * 0.0
    for i in range(0,len(ev)):
        hour = i%24
        # night charging at 1am, 2am, 3am
        if hour==1 or hour==2 or hour==3:
            ev[i] = 0.2
        # peak daily charging
        if hour==14 or hour==15 or hour==16 or hour==17:
            ev[i] = 0.1
        # factor increased energy for exterme low or high temperatures
        if hour==23:
            daytime_temp = new_daily_temp / 12.0
            new_daily_temp = 0.0
        # calculate temperature during the day
        if hour > 7 or hour < 21:
            new_daily_temp += temperature.values[i]
        # calculate energy increase if below 10 or above 28
        increase = 0.0
        if daytime_temp < 10.0:
            increase = ( (2.4 * daily_range_km)/100.0 ) * (10.0 - daytime_temp) / 5
        if daytime_temp > 28.0:
            increase = ( (2.3 * daily_range_km)/100.0 ) * (daytime_temp - 28.0) / 5
#       print('Temp {} energy {} increase {}'.format(daytime_temp, daily_energy, increase) )
        energy = daily_energy + increase
        ev[i] = ev[i] * energy
#   print(ev)
    return ev


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
#   print('Regression grad {} const {}'.format(reg_grad, reg_const) )
    return x_reg, y_reg

# Hydrogen boiler time series

def hydrogen_boiler(heat, efficiency):
    hydrogen = heat / efficiency
    return hydrogen

# Hybrid Heat Pump time series
def hybrid_heat_pump(heat, efficiency, threshold):
    demand = heat.copy()
    total_heat = demand['heat'].sum()
    # sort by temperature
    demand = demand.sort_values('temperature')
    hydrogen = demand['electricity'] * 0.0
    # starting with coldest hour, allocate to gas and zero electric
    # until the % energy is reached is reached.
    # (this is based on FES 2019 assumption of % energy used so we find
    #  out what threshold temperature this would need )
    #
    irow=0
    for index, row in demand.iterrows():
        # divide be efficiency because we use more primary gas energy than heat.
        hydrogen.iloc[irow] = row['heat'] / efficiency
        demand.iloc[irow,demand.columns.get_loc('electricity')] = 0.0
        threshold_temperature = row['temperature']
        if row['temperature'] > threshold:
            break
        irow+=1

    # gas = heat * 0.85
    # electric = 0
    demand = demand.sort_index()
    hydrogen = hydrogen.sort_index()
#   print(demand)
#   print(hydrogen)
    # output the temperature - this is the threshold temperature.

    return demand['electricity'], hydrogen

#
#  mod_electric_ref  - reference electricity with heat removed
#              wind  - wind capacity factor time series
#                pv  - pv capacity factor time series
#           scenario - percentage of heating from heat pumps
#              years - list of years to do the analysis for
#   normalise_factor - factor to normalise by.
#                      If doing daily is the max daily demand so that storage
#                      is relative to peak daily demand energy.
def supply_and_storage(mod_electric_ref, wind, pv, scenario, years, plot, hourly, ref_temp, climate, historic, heat_that_is_electric, normalise_factor, base=False):
    total_demand = 0

    # create the synthetic years
    if not historic:
        # ordinary year
        ordinary_year = mod_electric_ref.values
        # leap year
        # create a feb 29th by interpolating between feb 28th and Mar 1st
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

    if args.adverse:
        # make up full file name from the abreviation
        etypes = { 'd' : 'duration', 's': 'severity' }
        warmings = { 'a' : '12-3', 'b' : '12-4', 'c': '4' }
        warming = args.adverse[0:1]
        period = args.adverse[1:2]
        etype = args.adverse[2:3]
        eno = args.adverse[3:4]
        demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/adv/winter_wind_drought_uk_return_period_1_in_{}_years_{}_gwl{}degC_event{}.csv'.format(period, etypes[etype], warmings[warming], eno)
        adv_demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
        # get list of years in the file
        years = pd.Series(adv_demand.index.year).unique()

    demand_years=[]
    hydrogen_years=[]
    total_heat_demand_years={}
    mean_temp_years={}
    monthly_temp_years={}
    monthly_sd_years={}
    yearly_wd={}
    # for each weather year ...
    for year in years:
        print('Year {}'.format(year) )

        if args.adverse:
            demand = adv_demand[str(year) + '-01-01' : str(year) + '-12-31']
        else:
            # file name contructed from:
            # B    - the BDEW method
            # rhpp - the rhpp hourly (heat pump) profile
            # C    - climate change adjusted for temperature increase.
            file_base = 'Brhpp'
            if climate:
                file_base = 'BrhppC'
            demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(year, args.reference, file_base)
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
#       print(heat_weather)

        if historic:
            electric_ref = mod_electric_ref[str(year) + '-01-01' : str(year) + '-12-31']
            electric_ref.index = heat_weather.index
        else:
            # use leap year data or ordinary year data
            if calendar.isleap(year):
                electric_ref = pd.Series(leap_year, index=heat_weather.index)
            else:
                electric_ref = pd.Series(ordinary_year, index=heat_weather.index)
        heat_added = 0.0
        hydrogen = demand['electricity'] * 0.0
        hydrogen_efficiency = 0.85  # hydrogen boiler efficiency

        heat_pump_share = 0.0
        if scenario == 'H':
            heat_pump_share = 0.5
        if scenario == 'P':
            heat_pump_share = 1.0
        if scenario == 'F':
            heat_pump_share = 0.28  # fes 2019 Net Zero 2050

        # Half Heat pumps of all heat pumps
        if scenario == 'H' or scenario == 'P' or scenario == 'F':
            electric_heat = demand['electricity'] * heat_pump_share
            electric_ref = electric_ref + electric_heat
            heat_added = electric_heat.sum()
        # Existing heat only
        if scenario == 'E' :
            electric_heat = heat_weather * heat_that_is_electric
            electric_ref = electric_ref + electric_heat
            heat_added = electric_heat.sum()
        # Hydrogen boilers or half heat pumps
        if scenario == 'B' or scenario == 'H' or scenario == 'F':
            percent_boilers = 1.0
            if scenario == 'H':
                percent_boilers = 0.5
            if scenario == 'F':
                percent_boilers = 0.45
            hydrogen = hydrogen_boiler(heat_weather, hydrogen_efficiency) * percent_boilers
        # Hybdrid heat pumps
        if scenario == 'F' :
            percent_hybdrid_heat_pump = 0.13 # fes 2019 Net Zero 2050
            # temperature below which hybrid heat pumps switch completely to
            # hydrogen to balance the grid
            hybdrid_threshold = 5.0         
            electric_hp, hydrogen_hp = hybrid_heat_pump(demand, hydrogen_efficiency, hybdrid_threshold)
            electric_ref = electric_ref + electric_hp * percent_hybdrid_heat_pump
            hydrogen = hydrogen + hydrogen_hp * percent_hybdrid_heat_pump

        print('Demand for {} total {} heat {}'.format(year, electric_ref.sum(), heat_added ))

        # electric transport charging
        if args.ev:
            ev = ev_series(demand['temperature'], ev_annual_energy)
            electric_ref = electric_ref + ev

        # normalise and add to the list
        demand_years.append( electric_ref / normalise_factor)
        hydrogen_years.append( hydrogen / normalise_factor )
    # concantonate the demand series
    all_demand = pd.concat(demand_years[year] for year in range(len(years)) )
    all_hydrogen = pd.concat(hydrogen_years[year] for year in range(len(years)) )
    if plot:
        all_demand.plot(color='blue', label='Electric with {} heating'.format(heat_pump_share))
        pv.plot(color='red', label='Solar PV')
        wind.plot(color='green', label='Wind')
        all_hydrogen.plot(color='yellow', label='Hydrogen heating')
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
   
    if args.cplot:
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

    # look at the storage history for 2 wind and 1 pv
    # which is what we have now
    f_wind = 3.0
    f_pv = 2.0
    supply = wind * f_wind  +  pv * f_pv
    net = all_demand - supply
#   print('NET')
#   print(net)

    #  calculate how much storage we need
    store_hist = storage.storage(net, args.eta)
#   print(store_hist)
    store_size = store_hist.min()
    storage_days = round(store_size * -1.0)
    # if minimum value was the last one then store is just emptying more and
    # more so we didn't find out the size
    last_one = store_hist.iat[-1]
#   if store_size == store_hist.iat[-1] or storage_days>200:
#       storage_days = 200
    if storage_days <0.0:
        storage_days = 0.0
    print('Storage for {} wind {} pv size {} is {} days. last_one {}'.format(f_wind, f_pv, store_size, storage_days, last_one) )
    period_hist = store_hist + storage_days
    period_hist = period_hist.clip(0.0, storage_days)

    if plot:
        ax = period_hist.plot(label='store', color='green')
        plt.xlabel('day', fontsize=15)
        plt.ylabel('Store size in days', fontsize=15)
        ax2 = ax.twinx()
        ax2.set_ylabel('Energy ',color='red', fontsize=15)
        all_demand.plot(color='red', label='demand')
        supply.plot(color='blue', label='supply')
        net.plot(color='yellow', label='net')
        plt.title('Daily store size: wind {} pv {} days {} '.format(f_wind, f_pv, storage_days))
        plt.legend(loc='upper right')
        plt.show()

#   Calculate storage for different years for plotting
    yearly_start = store_hist.resample('Y').first()
#   print(yearly_start)
    yearly_max = store_hist.resample('Y').max()
#   print(yearly_max)
    yearly_min = store_hist.resample('Y').min()
#   print(yearly_min)
#   yearly_diff = yearly_max - yearly_min
    yearly_diff = yearly_start - yearly_min
#   print('YEARLY_DIFF')
#   print(yearly_diff)
    if plot:
        yearly_diff.plot(color='green', label='yearly store size')
        plt.legend(loc='upper right')
        plt.title('Store size at the end of each year: {} wind to {} solar'.format(f_wind, f_pv) )
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Store size in days', fontsize=15)
        plt.show()

    if args.cplot:
        # plot the wasserstein distances
        plt.scatter(list(year_var), yearly_wd.values())
        plt.title('Wasserstien distance of temperature distributions from 2018' )
        plt.xlabel('Year', fontsize=15)
        plt.ylabel('Wasserstien distance', fontsize=15)
        plt.show()
        

    # 
    if args.genh:
        h_input = all_hydrogen
    else:
        h_input = None
    # calculate storage at grid of Pv and Wind capacities for
    # hydrogen efficiency TODO need a reference
    grid=args.grid    # number of points (60)
    step=args.step    # step size (0.1)
    if base:
        df_list=[]
        for i_base in range(0,grid):
            base_load = i_base * 0.05
            print('Base load {}'.format(base_load))
            df= storage.storage_grid(all_demand, wind, pv, args.eta, hourly, grid, step, base_load, h_input)
            df['base'] = df['storage'] * 0.0 + base_load
            df_list.append(df)
        df = pd.concat(df_list)
#       print(df)
    else:
        print('Base load Zero')
        df= storage.storage_grid(all_demand, wind, pv, args.eta, hourly, grid, step, 0.0, h_input)

    # store actual capacity in GW
    df['gw_wind'] = df['f_wind'] * normalise_factor / ( 24 * 1000.0 )
    df['gw_pv'] = df['f_pv'] * normalise_factor / ( 24 * 1000.0 )

    # store yearly values
    yearly_data = { 'year'   : total_heat_demand_years.keys(),
                    'heat'   : total_heat_demand_years.values(),
                    'temp'   : mean_temp_years.values(),
                    'wd'     : yearly_wd.values(),
                    'storage'     : yearly_diff }
    yd = pd.DataFrame(yearly_data).set_index('year')
    return df, yd, all_demand, all_hydrogen

# main program

# Scenarios
scenarios = { 'P' : 'All Heat Pumps', 
              'F' : 'FES Net Zero Hybrid Heat Pumps',
              'H' : 'Half Heat Pumps',
              'B' : 'All Hydrgoen Boilers',
              'N' : 'No heating',
              'E' : 'Existing Heating based on weather' }

# process command line
parser = argparse.ArgumentParser(description='Show the impact of heat pumps or hydrogen on different shares of wind and solar')
parser.add_argument('--start', action="store", dest="start", help='Start Year', type=int, default=2017 )
parser.add_argument('--end', action="store", dest="end", help='End Year', type=int, default=2019 )
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
parser.add_argument('--adverse', action="store", dest="adverse", help='Use specified Adverse scenario file of the form a5s1 where a=warming, 5=return period, s=severity or d=duration, 1=event. The possible warmings are: a=12-3, b=12-4, c=4', default=None )
parser.add_argument('--scenario', action="store", dest="scenario", help=str(scenarios), default='H' )
parser.add_argument('--dir', action="store", dest="dir", help='Output directory', default='40years' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--cplot', action="store_true", dest="cplot", help='Show climate related plots', default=False)
parser.add_argument('--historic', action="store_true", dest="historic", help='Use historic time series instead of synthetic', default=False)
parser.add_argument('--hourly', action="store_true", dest="hourly", help='Use hourly time series', default=False)
parser.add_argument('--climate', action="store_true", dest="climate", help='Use climate change adjusted time series', default=False)
parser.add_argument('--base', action="store_true", dest="base", help='Use baseload shares', default=False)
parser.add_argument('--ev', action="store_true", dest="ev", help='Include Electric Vehicles', default=False)
parser.add_argument('--genh', action="store_true", dest="genh", help='Assume hydrogen made from electricity and stored in the same store', default=False)
parser.add_argument('--normalise', action="store", dest="normalise", help='Method of normalise by: annual, peak, kf.', default='annual')
parser.add_argument('--scale', action="store", dest="scale", help='How to scale : average (energy over the period), reference (year) or kf.', default="reference")
parser.add_argument('--eta', action="store", dest="eta", help='Efficiency of charge and discharge.', type=float, default=0.80)
parser.add_argument('--grid', action="store", dest="grid", help='Number of pionts in grid.', type=int, default=60)
parser.add_argument('--step', action="store", dest="step", help='Step size.', type=float, default=0.1)
parser.add_argument('--kf', action="store_true", dest="kf", help='Scale the generation data to KF Capacity factors', default=False)
parser.add_argument('--onshore', action="store_true", dest="onshore", help='Use only onshore wind', default=False)

args = parser.parse_args()

last_weather_year = args.end
hourly = args.hourly
if args.historic:
    hourly = False
# print arguments
print('Start year {} End Year {} Reference year {} plot {} hourly {} climate {} historic {} base {} ev {} genh {}'.format(args.start, args.end, args.reference, args.plot, args.hourly, args.climate, args.historic, args.base, args.ev, args.genh) )

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
#  ( even if using historic series we calculate it so can scale to it )

mod_electric_ref = electric_ref - (ref_resistive_heat * heat_that_is_electric)
total_energy = electric_ref.sum()

# Normalise by the unmodified reference year time series
# so comparisons are possible
daily_original_electric_with_heat = electric_ref.resample('D').sum()
if args.normalise == 'annual':
    normalise_factor = daily_original_electric_with_heat.mean()
else:
    if args.normalise == 'peak':
        normalise_factor = daily_original_electric_with_heat.max()
    else:
        normalise_factor = 835616.0
print('PEAK DEMAND {} Annual Demand {} Normalise Factor {}'.format(daily_original_electric_with_heat.max(), daily_original_electric_with_heat.mean(), normalise_factor))

if not args.historic:

    daily_electric_ref = mod_electric_ref.resample('D').sum()
else:
    # use KFs historic series instead
    demand_filename = '/home/malcolm/uclan/data/kf/UKDailyELD19832014.csv'
    last_weather_year = 2013
    if args.start > last_weather_year:
        print("ERROR: start year > last")
        quit()
    kf = pd.read_csv(demand_filename, header=None, squeeze=True)
    d = pd.date_range(start = '1983-01-01', end = '2013-12-31', freq='D' )
    mod_electric_ref = pd.Series(kf.values[0:len(d)], d, dtype='float64', name='ENGLAND_WALES_DEMAND')
    # scale england and wales to scotland
    mod_electric_ref = mod_electric_ref * scotland_factor
    # use the KF method of scaling to the average annual energy of the 30 years
    #  instead of # of the reference year.
    if args.scale == 'average':
        total_energy = mod_electric_ref.sum() / 30
    if args.scale == 'kf':
        total_energy = 305000000.0

    # scale by adding or subtracting a constant as per KF method
    new_values = np.empty(0)
    for year in range(args.start, last_weather_year+1):
        year_electric = mod_electric_ref[str(year)]
#       print(year)
#       print(year_electric)
        adjustment = (total_energy - year_electric.sum()) / year_electric.size
        print("Year {} len {} adjustment {} total {}".format(year, year_electric.size, adjustment, total_energy) )
        year_electric = year_electric + adjustment
        new_values = np.concatenate((new_values, year_electric.values))
    d = pd.date_range(start = str(args.start) + '-01-01', end = str(last_weather_year) + '-12-31', freq='D' )
    mod_electric_ref = pd.Series(new_values, d, dtype='float64', name='ENGLAND_WALES_DEMAND')
    daily_electric_ref = mod_electric_ref


# plot reference year electricity

if args.plot:

    daily_electric_ref.plot(color='blue', label='Historic Electric without heating {}'.format(args.reference))
    plt.title('Reference year Daily Electricity with heat removed')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Electricity (Mwh) per day', fontsize=15)
#plt.legend(loc='upper right')
    if args.ev:
        ev = ev_series(ref_temperature, ev_annual_energy)
        daily_ev = ev.resample('D').sum()
        daily_ev.plot(color='red', label='Electric vehicles {}'.format(args.reference))
        plt.legend(loc='upper right')
    plt.show()

if args.adverse:
    print('Loading adverse PV')
    pv_filename = '/home/malcolm/uclan/output/pv/adv{}.csv'.format(args.adverse)
    pv_adv = pd.read_csv(pv_filename, header=0, parse_dates=[0], index_col=0 )

    annual_pv = {}
#   TODO there seems to be a problem with the c location for pv? Fixed ?!
    locations =  ['a', 'c', 'e', 'u']
#   locations =  ['a', 'e', 'u']
    for location in locations:
        annual_pv[location] = pv_adv['power_' + location].sum()
        print(location, annual_pv[location])
    max_pv = max(annual_pv.values())

    # PV rated power in kW ????
    rated_power = 1000.0

    for location in locations:
        # scale to max energy
        pv_adv['power_' + location] = pv_adv['power_' + location] * (max_pv / annual_pv[location])

    for location in locations:
        # convert to capacity factor
        pv_adv['power_' + location] = pv_adv['power_' + location] / (rated_power * 24)

    pv_cf = pd.concat([pv_adv['power_' + location] for location in locations], axis=1)

    # create mean to represent whole country
    pv = pv_cf.sum(axis=1) / len(locations)
    # adjust the capacity factor inline with the assumed load factor of 0.116
    # from KF TODO - use Ninja CF ?
    pv = pv * ( 0.116 / pv.mean() )
    print(pv)

    print('Loading adverse Wind')
    wind_filename = '/home/malcolm/uclan/output/wind/adv{}.csv'.format(args.adverse)
    wind_adv = pd.read_csv(wind_filename, header=0, parse_dates=[0], index_col=0 )
    locations =  ['a', 'b', 'c', 'l', 's', 'w']
    annual_wind={}
    for location in locations:
        annual_wind[location] = wind_adv['power_' + location].sum()
        print(location, annual_wind[location])
    max_wind = max(annual_wind.values())

    rated_power = 2500

    for location in locations:
        # scale to max energy
        wind_adv['power_' + location] = wind_adv['power_' + location] * (max_wind / annual_wind[location])

    for location in locations:
        # convert to capacity factor
        wind_adv['power_' + location] = wind_adv['power_' + location] / (rated_power * 24)

    wind_cf = pd.concat([wind_adv['power_' + location] for location in locations], axis=1)

    # create mean to represent whole country
    wind = wind_cf.sum(axis=1) / len(locations)

    # adjust the capacity factor inline with the assumed load factor of 0.28
    # from KF TODO - use Ninja CF ?
    wind = wind * ( 0.28 / wind.mean() )
    print(wind)

    years = pd.Series(wind.index.year).unique()
    print(years)
else:

    # weather years from the start to 2019
    # ( need to download more ninja to get up to 2020 )
    #last_weather_year = 2019
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
    if args.onshore:
        wind = ninja_wind['onshore']
    else:
        wind = ninja_wind['national']

    if args.kf:
        wind_mean = wind.mean()
        pv_mean = pv.mean()
        print('Converting to KF capacity factors from wind {} pv {} ...'.format(wind_mean, pv_mean))
        pv = pv * 0.116 / pv_mean
        wind = wind * 0.28 / wind_mean

print('Generation PV: Number of value {} mean CF {} ,  Wind: number of values {} meaqn CF {} '.format(len(pv), pv.mean(), len(wind), wind.mean() ) )

if args.plot:
#   print(wind)
#   print(pv)
    weather_source = 'ninja'
    if args.adverse:
        weather_source = 'adv ' + args.adverse
  
    # daily plot
    wind_daily = wind.resample('D').mean()
    pv_daily = pv.resample('D').mean()
    wind_daily.plot(color='blue', label='{} wind generation'.format(weather_source))
    pv_daily.plot(color='red', label='{} pv generation'.format(weather_source))
    plt.title('Wind and solar generation from {}'.format(weather_source))
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

if hourly:
    print('Using hourly time series')
else:
    print('Using daily time series')
    mod_electric_ref = mod_electric_ref.resample('D').sum()
    if not args.adverse:
        wind = wind.resample('D').mean()
        pv = pv.resample('D').mean()

df, yd, all_demand, all_hydrogen = supply_and_storage(mod_electric_ref, wind, pv, args.scenario, years, args.plot, hourly, ref_temperature, args.climate, args.historic, heat_that_is_electric, normalise_factor, args.base)
print("Max storage {} Min Storage {}".format(df['storage'].max(), df['storage'].min()) )

output_dir = "/home/malcolm/uclan/output/" + args.dir
electricChar = 'S'
if args.historic:
    electricChar = 'H'
climateChar = 'N'
if args.climate:
    climateChar = 'C'
scenarioChar = args.scenario
if args.adverse:
    scenarioChar = args.adverse + args.scenario

yd.to_csv('{}/yearly{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar))
df.to_csv('{}/shares{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar))
all_demand.to_csv('{}/demand{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar))
all_hydrogen.to_csv('{}/hydrogen{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar))

## TODO wasserstein distance of heat demand to see if it changes over the
##      years.
## and for wind speed and solar irradiance.


