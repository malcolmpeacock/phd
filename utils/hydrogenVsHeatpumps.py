# Look at the impact of 50% heat pumps on the electricity demand
# using 40 years weather
# and justify using the 2018 electricity series 

# library stuff
import sys
import os
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy.stats import wasserstein_distance

# custom code
import stats
import readers
import storage
import math

start_time = datetime.now()

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
def supply_and_storage(mod_electric_ref, wind, pv, scenario, years, plot, hourly, climate, use_baseline, heat_that_is_electric, normalise_factor, base, baseload, variable):
    total_demand = 0

    # create the synthetic years
    if use_baseline:
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
        p_end = len(args.adverse)-2
        period = args.adverse[1:p_end]
        etype = args.adverse[p_end:p_end+1]
        eno = args.adverse[p_end+1:p_end+2]
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
    dec31_wind={}
    dec31_pv={}
    dec31_demand={}

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
        #  account for leap years.
        #  Need to create a new df with the index same as heat_weather
        #  then create a 29th of Feb by interpolation between 28th and
        #  1st March and then set the data values into empty DF
        if not hourly:
            demand = demand.resample('D').sum()
        heat_weather = demand['heat']
#       print(heat_weather)

        if not use_baseline:
            electric_ref = mod_electric_ref[str(year) + '-01-01' : str(year) + '-12-31']
            electric_ref.index = heat_weather.index
        else:
            # use leap year data or ordinary year data
            if calendar.isleap(year):
                year_values = leap_year
            else:
                year_values = ordinary_year

            # Shift for days of the week here to match weather year
            if args.shift:
                year_values = storage.shiftdays(year_values, mod_electric_ref.index, heat_weather.index)

            # Convert to a series
            electric_ref = pd.Series(year_values, index=heat_weather.index)
            

        heat_added = 0.0
        hydrogen = demand['electricity'] * 0.0
        # From 0.85 hydrogen to heat and 0.95 hydrogen transmission.
        hydrogen_efficiency = 0.80  # hydrogen boiler efficiency

        heat_pump_share = 0.0
        if scenario == 'H':
            heat_pump_share = 0.5
        if scenario == 'R':
            heat_pump_share = heat_that_is_electric
        if scenario == 'P':
            heat_pump_share = 1.0
        if scenario == 'F':
            heat_pump_share = 0.28  # fes 2019 Net Zero 2050
        if scenario == 'G':
            heat_pump_share = 0.41  # fes 2019 Net Zero 2050 but lumping 
                                    # the hybrid heat pumps in with others.

        # Half Heat pumps of all heat pumps
        if scenario == 'H' or scenario == 'P' or scenario == 'F' or scenario == 'G' or scenario == 'R':
            electric_heat = demand['electricity'] * heat_pump_share
            electric_ref = electric_ref + electric_heat
            heat_added = electric_heat.sum()
        # Existing heat only
        if scenario == 'E' :
            if not use_baseline:
                heat_that_is_electric = 0
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


        # electric transport charging
        if args.ev:
            ev = ev_series(demand['temperature'], ev_annual_energy)
            electric_ref = electric_ref + ev

        print('Demand for {} total {} heat {}'.format(year, electric_ref.sum(), heat_added ))

        # normalise and add to the list
        demand_years.append( electric_ref / normalise_factor)
        hydrogen_years.append( hydrogen / normalise_factor )

        # December 31st values
        dec31_date = '{}-12-31'.format(year)
        dec31_wind[year] = wind[dec31_date]
        dec31_pv[year] = pv[dec31_date]
        dec31_demand[year] = electric_ref[dec31_date] / normalise_factor
        

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

    # energy from hydrogen
    if args.genh:
        h_input = all_hydrogen
    else:
        h_input = None
    nwind=args.nwind    # number of points (60)
    npv=args.npv    # number of points (60)
    step=args.step    # step size (0.1)
    if base:
        df_list=[]
        for i_base in range(0,10):
            base_load = i_base * 0.05
            print('Base load {}'.format(base_load))
            df= storage.storage_grid(all_demand, wind, pv, eta, hourly, npv, nwind, step, base_load, variable, h_input, args.storage)
            df['base'] = df['storage'] * 0.0 + base_load
            df_list.append(df)
        df = pd.concat(df_list)
#       print(df)
    else:
        print('Base load Zero')
        if args.storage == 'new':
            df, sample_hist, sample_durations = storage.storage_grid_new(all_demand, wind, pv, eta, hourly, npv, nwind, step, baseload, h_input, args.constraints, args.wind, args.pv, args.days, args.threshold, variable, args.contours, args.debug)
        else:
            df, sample_hist, sample_durations = storage.storage_grid(all_demand, wind, pv, eta, hourly, npv, nwind, step, baseload, variable, h_input, args.storage, args.wind, args.pv, args.threshold, args.constraints, args.debug, args.store_max)
        df['base'] = df['storage'] * 0.0 + baseload
        df['variable'] = df['storage'] * 0.0 + variable

    # store actual capacity in GW
    df['gw_wind'] = df['f_wind'] * normalise_factor / ( 24 * 1000.0 )
    df['gw_pv'] = df['f_pv'] * normalise_factor / ( 24 * 1000.0 )

    # calculate storage for different years.
    yearly_start = sample_hist.resample('Y').first()
    yearly_max = sample_hist.resample('Y').max()
    yearly_min = sample_hist.resample('Y').min()
#   yearly_diff = yearly_max - yearly_min
    yearly_diff = yearly_start - yearly_min

    # store yearly values
    yearly_data = { 'year'   : total_heat_demand_years.keys(),
                    'heat'   : total_heat_demand_years.values(),
                    'temp'   : mean_temp_years.values(),
                    'storage'      : yearly_diff,
                    'dec31_wind'   : dec31_wind.values(),
                    'dec31_pv'     :    dec31_pv.values(),
                    'dec31_demand' :    dec31_demand.values()  }
    yd = pd.DataFrame(yearly_data).set_index('year')
    return df, yd, all_demand, all_hydrogen, sample_hist, sample_durations

# main program

# Scenarios
scenarios = { 'P' : 'All Heat Pumps', 
              'F' : 'FES Net Zero Hybrid Heat Pumps',
              'G' : '41 percent Heat Pumps',
              'H' : 'Half Heat Pumps',
              'R' : 'Existing heating provided by Heat Pumps',
              'B' : 'All Hydrgoen Boilers',
              'N' : 'No heating',
              'E' : 'Existing Heating based on weather' }

# process command line
parser = argparse.ArgumentParser(description='Show the impact of heat pumps or hydrogen on different shares of wind and solar')
parser.add_argument('--start', action="store", dest="start", help='Start Year', type=int, default=2017 )
parser.add_argument('--end', action="store", dest="end", help='End Year', type=int, default=2019 )
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
parser.add_argument('--adverse', action="store", dest="adverse", help='Use specified Adverse scenario file of the form a5s1 where a=warming, 5=return period, s=severity or d=duration, 1=event. The possible warmings are: a=12-3, b=12-4, c=4', default=None )
parser.add_argument('--scenario', action="store", dest="scenario", help=str(scenarios), default='H', choices=scenarios.keys() )
parser.add_argument('--dir', action="store", dest="dir", help='Output directory', default='40years' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--debug', action="store_true", dest="debug", help='Output debug info', default=False)
parser.add_argument('--cplot', action="store_true", dest="cplot", help='Show climate related plots', default=False)
parser.add_argument('--dmethod', action="store", dest="dmethod", help='Method of creating a multi year demand series: add=add a varying amount as per KF, multiply=multiply by a varying amount as per the cost paper, baseline=my method (synthetic time series)', choices=['baseline', 'add', 'multiply'], default='baseline')
parser.add_argument('--hourly', action="store_true", dest="hourly", help='Use hourly time series', default=False)
parser.add_argument('--climate', action="store_true", dest="climate", help='Use climate change adjusted time series', default=False)
parser.add_argument('--base', action="store_true", dest="base", help='Use range of  baseload shares', default=False)
parser.add_argument('--ev', action="store_true", dest="ev", help='Include Electric Vehicles', default=False)
parser.add_argument('--genh', action="store_true", dest="genh", help='Assume hydrogen made from electricity and stored in the same store', default=False)
parser.add_argument('--normalise', action="store", dest="normalise", help='Method of normalise by (ie converting to days): annual, peak, kf.', default='annual', choices=['annual', 'peak', 'kf', 'scale'])
parser.add_argument('--scale', action="store", dest="scale", help='How to scale : average (energy over the period), reference (by the reference year) or a value passed in.', default="reference")
parser.add_argument('--storage', action="store", dest="storage", help='Storage model kf , mp, new or all', default="kf")
parser.add_argument('--constraints', action="store", dest="constraints", help='Constraints on new storage model: new or old', default="new")
parser.add_argument('--eta', action="store", dest="eta", help='Round Trip Efficiency.', type=int, default=85)
parser.add_argument('--npv', action="store", dest="npv", help='Number of points in pv grid.', type=int, default=60)
parser.add_argument('--nwind', action="store", dest="nwind", help='Number of points in wind grid.', type=int, default=60)
parser.add_argument('--baseload', action="store", dest="baseload", help='Base load capacity.', type=float, default=0.0)
parser.add_argument('--step', action="store", dest="step", help='Step size.', type=float, default=0.1)
parser.add_argument('--kf', action="store_true", dest="kf", help='Scale the generation data to KF Capacity factors', default=False)
parser.add_argument('--kf2', action="store_true", dest="kf2", help='Scale the generation data to KF Capacity factors ', default=False)
parser.add_argument('--cfpv', action="store", dest="cfpv", help='PV capacity factor to scale to, default is to leave unchanged', type=float, default=0)
parser.add_argument('--cfwind', action="store", dest="cfwind", help='Wind capacity factor to scale to, default is to leave unchanged', type=float, default=0)
parser.add_argument('--shore', action="store", dest="shore", default="all", help='on=Use only onshore wind off=only offshore, all=all' )
parser.add_argument('--ninja', action="store", dest="ninja", default="near", help='Which ninja to use: near, current, future', choices=['near', 'current', 'future'] )
parser.add_argument('--kfpv', action="store_true", dest="kfpv", help='Use KF PV generation from matlab', default=False)
parser.add_argument('--kfwind', action="store_true", dest="kfwind", help='Use KF wind generation from matlab', default=False)
parser.add_argument('--demand', action="store", dest="demand", help='Electricity demand source', choices=['espini', 'kf', 'ngrid'], default='espini')
parser.add_argument('--shift', action="store_true", dest="shift", help='Shift the days to match weather calender', default=False)
parser.add_argument('--wind', action="store", dest="wind", help='Wind value of store history to output', type=float, default=0)
parser.add_argument('--pv', action="store", dest="pv", help='Pv value of store history to output', type=float, default=0)
parser.add_argument('--days', action="store", dest="days", help='Example store size to find for store hist plotting', type=float, default=0)
parser.add_argument('--threshold', action="store", dest="threshold", help='Threshold for considering 2 wind values the same in new storage model', type=float, default=0.01)
parser.add_argument('--variable', action="store", dest="variable", help='Amount of variable generation, default-0.0', type=float, default=0.0)
parser.add_argument('--store_max', action="store", dest="store_max", help='Maximum value of storage in days, default=80.0', type=float, default=80.0)
parser.add_argument('--contours', action="store", dest="contours", help='Set of values to use for contour lines', default='med')

args = parser.parse_args()

if args.shift and args.reference != '2018':
    print('Error Can not have shift if reference year not 2018')
    quit()

output_dir = "/home/malcolm/uclan/output/" + args.dir
if not os.path.isdir(output_dir):
    print('Error output dir {} does not exist'.format(output_dir))
    quit()
    
if args.dmethod == 'baseline':
    if args.demand == 'kf' : 
        if args.reference < '1984' or args.reference > '2013':
            print('Error reference year {} out of range'.format(args.reference))
            quit()
    if args.demand == 'espini' : 
        if args.reference < '2009' or args.reference > '2020':
            print('Error reference year {} out of range'.format(args.reference))
            quit()
else:
    if args.demand == 'espini' : 
        if args.start < 2009 or args.end > 2020:
            print('Error years {} to {} out of range'.format(args.start, args.end))
            quit()
if args.demand == 'kf' : 
    if args.start < 1984 or args.end > 2013:
        print('Error years {} to {} out of range'.format(args.start, args.end))
        quit()

last_weather_year = args.end
hourly = args.hourly
if args.demand == 'kf' and hourly:
    print('Error hourly time series not possible with kf demand')
    quit()

# print arguments
print('Start year {} End Year {} Reference year {} plot {} hourly {} climate {} demand {} dmethod {} base {} ev {} genh {}'.format(args.start, args.end, args.reference, args.plot, args.hourly, args.climate, args.demand, args.dmethod, args.base, args.ev, args.genh) )

# calculate charge and discharge efficiency from round trip efficiency
eta = math.sqrt(args.eta / 100)
print('Round trip efficiency {} Charge/Discharge {} '.format(args.eta / 100, eta) )

scotland_factor = 1.1    # ( Fragaki et. al )

if args.dmethod == 'baseline':
    if args.demand == 'kf':
        print('ERROR: baseline not implemented with kf demand')
        quit()

# read historical electricity demand for reference year
# NOTE: espini includes actual Scottish demand.
#  ( even if using historic series we calculate it so can scale to it )


if args.demand == 'espini':
    demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
    demand_ref = readers.read_espeni(demand_filename, args.reference)
    electric_ref = demand_ref
else:
    demand_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + args.reference + '.csv'
    demand_ref = readers.read_electric_hourly(demand_filename)
    electric_ref = demand_ref['ENGLAND_WALES_DEMAND'] * scotland_factor

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
if args.hourly:
    normalise_factor = normalise_factor / 24.0
print('PEAK DEMAND {} Annual Demand {} Mean Daily Demand {} Normalise Factor {}'.format(daily_original_electric_with_heat.max(), daily_original_electric_with_heat.sum(), daily_original_electric_with_heat.mean(), normalise_factor))

# input assumptions for reference year
heat_that_is_electric = 0.06     # my spreadsheet from DUKES
#   heat_that_is_heat_pumps = 0.01   # greenmatch.co.uk, renewableenergyhub.co.uk

total_energy = electric_ref.sum()
# baseline demand
if args.dmethod == 'baseline':

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

    # plot the electricity time series with heating removed
    if args.plot:
        daily_original_electric_with_heat.plot(color='green', label='Historic 2018 electricity demand')
        daily_electric_ref.plot(color='blue', label='Baseline electricity demand with heat removed')
        plt.title('Removing existing heating from reference year electricity')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Electricity demand (TWh)', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()

    # plot the 2018 series with all heat pumps vs the historic 2018
    if args.plot:
        demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew.csv'.format(args.reference) 
        ref_electric_hp = readers.read_copheat(demand_filename, ['electricity'])
        electric2018_withhp = mod_electric_ref + ref_electric_hp
        fes_withhp = mod_electric_ref + ref_electric_hp * 0.41
        daily_new_2018 = electric2018_withhp.resample('D').sum()
        daily_fes = fes_withhp.resample('D').sum()
        daily_original_electric_with_heat.plot(color='blue', label='Historic 2018 electricity demand')
        daily_new_2018.plot(color='red', label='2018 electricity demand with all heating as heat pumps')
        daily_fes.plot(color='green', label='2018 electricity demand with 41% heating as heat pumps')
        if args.scenario == 'R':
            existing_withhp = mod_electric_ref + ref_electric_hp * heat_that_is_electric
            daily_existing_withhp = existing_withhp.resample('D').sum()
            daily_existing_withhp.plot(color='purple', label='2018 electricity demand with existing heating as heat pumps')
            print('Annual Demand for Existing heat with heat pumps {} Historic {} 41% {} All {}'.format(daily_existing_withhp.sum(), daily_original_electric_with_heat.sum(), daily_fes.sum(), daily_new_2018.sum()) )
            print('Peak Demand for Existing heat with heat pumps {} Historic {} 41% {} All {}'.format(existing_withhp.max(), electric_ref.max(), fes_withhp.max(), electric2018_withhp.max()) )
        plt.title('Impact of heat pumps on 2018 daily electricity demand')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Electricity demand (TWh)', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()
    
        print('Historic Electric {} With all Heat Pumps added {} with 41% heat pumps {}'.format(daily_original_electric_with_heat.sum(), daily_new_2018.sum(), daily_fes.sum() ) )

# output time series for KF
#   timeseries_dir = '/home/malcolm/uclan/output/timeseries/'
#   daily_electric_ref.to_csv(timeseries_dir + 'baseline_daily_2018.csv', float_format='%g')
#   daily_original_electric_with_heat.to_csv(timeseries_dir + 'historic_daily_2018.csv', float_format='%g')
#   daily_new_2018.to_csv(timeseries_dir + 'heatpumps_all_daily_2018.csv', float_format='%g')
#   daily_fes.to_csv(timeseries_dir + 'heatpumps_41_daily_2018.csv', float_format='%g')

# not baseline demand, using scaled historic series
    daily_electric_ref = mod_electric_ref.resample('D').sum()
else:
    if args.demand == 'espini':
        demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
        demand_espini = readers.read_espeni(demand_filename, None)
        print(demand_espini)
        espini_start = str(args.start) + '-01-01 00:00:00+00:00'
        espini_end = str(args.end) + '-12-31 23:00:00+00:00'
        mod_electric_ref = demand_espini[espini_start : espini_end]
        d = pd.date_range(start = espini_start, end = espini_end, freq='H' )

    else:
        # use KFs historic series instead
        demand_filename = '/home/malcolm/uclan/data/kf/UKDailyELD19832014.csv'
        last_weather_year = 2013
        if args.start > last_weather_year:
            print("ERROR: start year > last")
            quit()
        kf = pd.read_csv(demand_filename, header=None, squeeze=True)
        d = pd.date_range(start = '1984-01-01', end = '2013-12-31', freq='D' )
        mod_electric_ref = pd.Series(kf.values[0:len(d)], d, dtype='float64', name='ENGLAND_WALES_DEMAND')
        # scale england and wales to scotland
        mod_electric_ref = mod_electric_ref * scotland_factor

    # Scale by reference year
    if args.scale == 'average':
        total_energy = mod_electric_ref.sum() / (1 + args.end - args.start)
    # or a value passed in ( eg KF 305.0 ) in TWh
    else:
        if args.scale != 'reference':
            total_energy = float(args.scale) * 1e6

    new_values = np.empty(0)
    if args.dmethod == 'add':
        # scale by adding or subtracting a constant as per KF method
        for year in range(args.start, last_weather_year+1):
            year_electric = mod_electric_ref[str(year)]
            adjustment = (total_energy - year_electric.sum()) / year_electric.size
            print("Year {} len {} adjustment {} total {}".format(year, year_electric.size, adjustment, total_energy) )
            year_electric = year_electric + adjustment
            new_values = np.concatenate((new_values, year_electric.values))

    else:
        # scale by multiplying as per the cost paper
        for year in range(args.start, last_weather_year+1):
            year_electric = mod_electric_ref[str(year)]
            adjustment = total_energy / year_electric.sum()
            print("Year {} len {} adjustment {} total {}".format(year, year_electric.size, adjustment, total_energy) )
            year_electric = year_electric * adjustment
            new_values = np.concatenate((new_values, year_electric.values))


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
    locations =  ['a', 'c', 'e', 'u']
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
#   print(pv)

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
#   print(wind)

    years = pd.Series(wind.index.year).unique()
    print(years)
else:
    years = range(args.start, last_weather_year+1)
    ninja_start = str(years[0]) + '-01-01 00:00:00'
    ninja_end = str(years[-1]) + '-12-31 23:00:00'
    print(ninja_start, ninja_end)
    if args.kfpv or args.kfwind:
        # create index
        kf_start = '1984-01-01'
        kf_end = '2013-12-31'
        kf_index = pd.date_range(start = kf_start, end = kf_end, freq='D', tz='UTC' )
        years = range(1984, 2014)
        energy_per_day = daily_electric_ref.mean() * 1e6
        print('KF GEN Energy {}'.format(energy_per_day))
        energy_per_day = 836757995855.537

    if args.kfpv:
        print('Loading KF PV generation')
        pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
        pv = pd.read_csv(pv_filename, header=None, squeeze=True)
        pv.index = kf_index
        kf_pcf = 0.1156
        pv = pv * kf_pcf / energy_per_day
    else:
        # Ninja capacity factors for pv
        ninja_filename_pv = '/home/malcolm/uclan/data/ninja/ninja_pv_country_GB_merra-2_corrected.csv'
        print('Loading ninja ...')
        ninja_pv = readers.read_ninja_country(ninja_filename_pv)
        print('Extracting PV ...')
        ninja_pv = ninja_pv[ninja_start : ninja_end]
        pv = ninja_pv['national']

    if args.kfwind:
        wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
        wind = pd.read_csv(wind_filename, header=None, squeeze=True)
        wind.index = kf_index
        kf_wcf = 0.28
        wind = wind * kf_wcf / energy_per_day

    else:

        # Ninja capacity factors for wind
        if args.ninja == 'near' :
            ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_near-termfuture-merra-2_corrected.csv'
        else:
            if args.ninja == 'future' :
                ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_long-termfuture-merra-2_corrected.csv'
            else:
                ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_current-merra-2_corrected.csv'

        ninja_wind = readers.read_ninja_country(ninja_filename_wind)

        print('Extracting Wind ninja {} ...'.format(args.ninja))
        ninja_wind = ninja_wind[ninja_start : ninja_end]
        if args.shore == 'on':
            wind = ninja_wind['onshore']
        else:
            if args.shore == 'off':
                wind = ninja_wind['offshore']
            else:
                wind = ninja_wind['national']

    if args.kf:
        pcf = 0.1156
        wcf = 0.28
        wind_mean = wind.mean()
        pv_mean = pv.mean()
        print('Converting to KF capacity factors from wind {} pv {} to wind {} pv {} ...'.format(wind_mean, pv_mean, wcf, pcf))
        pv = pv * pcf / pv_mean
        wind = wind * wcf / wind_mean
    if args.kf2:
        pcf = 0.11517
        wcf = 0.40746
        wind_mean = wind.mean()
        pv_mean = pv.mean()
        print('Converting to KF capacity factors from wind {} pv {} to wind {} pv {} ...'.format(wind_mean, pv_mean, wcf, pcf))
        pv = pv * pcf / pv_mean
        wind = wind * wcf / wind_mean
    if args.cfpv > 0:
        print('Converting to PV capacity factors from {} to {} ...'.format(pv.mean(), args.cfpv))
        pv = pv * args.cfpv / pv.mean()
    if args.cfwind > 0:
        print('Converting to Wind capacity factors from {} to {} ...'.format(wind.mean(), args.cfwind))
        wind = wind * args.cfwind / wind.mean()


if args.plot:
#   print(wind)
#   print(pv)
    wind_source = 'ninja'
    pv_source = 'ninja'
    if args.adverse:
        wind_source = 'adv ' + args.adverse
        pv_source = 'adv ' + args.adverse
    if args.kfwind:
        wind_source = 'kf'
    if args.kfpv:
        pv_source = 'kf'
  
    # daily plot
    wind_daily = wind.resample('D').mean()
    pv_daily = pv.resample('D').mean()
    wind_daily.plot(color='blue', label='{} wind generation'.format(wind_source))
    pv_daily.plot(color='red', label='{} pv generation'.format(pv_source))
    plt.title('Wind and solar generation')
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

print('Generation PV: Number of values {} mean CF {} Total {} ,  Wind: number of values {} mean CF {} Total {} '.format(len(pv), pv.mean(), pv.sum(), len(wind), wind.mean(), wind.sum() ) )

df, yd, all_demand, all_hydrogen, sample_hist, sample_durations = supply_and_storage(mod_electric_ref, wind, pv, args.scenario, years, args.plot, hourly, args.climate, args.dmethod == 'baseline', heat_that_is_electric, normalise_factor, args.base, args.baseload, args.variable)
print("Max storage {} Min Storage {}".format(df['storage'].max(), df['storage'].min()) )

electricChar = 'S'
if args.dmethod == 'add':
    electricChar = 'H'
if args.dmethod == 'multiply':
    electricChar = 'M'
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
sample_hist.to_csv('{}/store{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar))
sample_durations.to_csv('{}/duration{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar))

# output settings
settings = {
    'start'     : args.start,
    'end'       : args.end,
    'reference' : args.reference,
    'ev'        : args.ev,
    'storage'   : args.storage,
    'variable'  : args.variable,
    'baseload'  : args.baseload,
    'hist_pv'   : args.pv,
    'hist_wind' : args.wind,
    'eta'       : args.eta,
    'cfpv'      : args.cfpv,
    'cfwind'    : args.cfwind,
    'demand'    : args.demand,
    'dmethod'   : args.dmethod,
    'hourly'    : args.hourly,
    'kfpv'      : args.kfpv,
    'kfwind'    : args.kfwind,
    'shore'     : args.shore,
    'threshold' : args.threshold,
    'normalise' : normalise_factor,
    'max_storage' : df['storage'].max(),
    'min_storage' : df['storage'].min(),
    'run_time'  : math.floor(datetime.timestamp(datetime.now()) - datetime.timestamp(start_time))
}
settings_df = pd.DataFrame.from_dict(data=settings, orient='index')
settings_df.to_csv('{}/settings{}{}{}.csv'.format(output_dir, scenarioChar, climateChar, electricChar), header=False)
