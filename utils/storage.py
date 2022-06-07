# How much storage is required for different amounts of PV and Wind ? using the capacity factors
# Reproduce Katerina graph for 2018 only.

# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import calendar
import math

# custom code
import stats
import readers
#from misc import upsample_df

# check that the store returns to zero at least once per year

def check_zero(store_hist):
    last_zero = store_hist.index[0].year
    non_zero = 0
    for index, value in store_hist.items():
        if value == 0:
            year = index.year
            if year - last_zero > 1:
                non_zero +=1
            last_zero = year
    return last_zero, non_zero

# calculate storage ( old kf method )

def storage(net_demand, eta=0.75, hydrogen=None):
    history = pd.Series(np.zeros(len(net_demand)), index=net_demand.index, name='store')
    store = 0.0
    eta_charge = eta
    eta_discharge = eta
    count=0
    for index, value in net_demand.items():
        # Note: both subtract because value is negative in the 2nd one!
        if value > 0.0:        # demand exceeds supply 
                               # discharge, so divide by eta - take more out
            store = store - value / eta_discharge
        else:                  # supply exceeds demand
                               # charge, so multiply by eta - put less in
            store = store - value * eta_charge
        # take hydrogen out of the store
        if hydrogen is not None:
            store = store - hydrogen.iat[count] / eta_discharge
#           print('Hydrogen: {} '.format(store))
        # The store starts full and it can never be over filled
        if store>0:
            store=0
        # record the size of the store
        history.iat[count] = store
        count += 1
    return history

# calculate storage ( new mp method )

def storage_mp(net_demand, eta=0.75, hydrogen=None):
    history = pd.Series(np.zeros(len(net_demand)), index=net_demand.index, name='store')
    store = 0.0
    eta_charge = eta
    eta_discharge = eta
    count=0
    for index, value in net_demand.items():
        # Note: both subtract because value is negative in the 2nd one!
        if value > 0.0:        # demand exceeds supply 
                               # discharge, so divide by eta - take more out
            store = store - value / eta_discharge
        else:                  # supply exceeds demand
                               # charge, so multiply by eta - put less in
            store = store - value * eta_charge
        # take hydrogen out of the store
        if hydrogen is not None:
            store = store - hydrogen.iat[count] / eta_discharge
#           print('Hydrogen: {} '.format(store))
        # TODO the store starts full and it can never be over filled
#       if store>0:
#           store=0
        # record the size of the store
        history.iat[count] = store
        count += 1
    return history

# constant storage line

def storage_line(df, storage_value, method='interp1', wind_parm='f_wind', pv_parm='f_pv'):
    # just take data points with storage values in a range
    if method=='threshold':
        threshold = storage_value * 0.05
        s_df = df[(df['storage'] < storage_value + threshold) & (df['storage'] > storage_value - threshold)]
        if len(s_df.index)<2:
            print('WARNING: storage line {} days had only {} points'.format(storage_value, len(s_df.index) ) )
        sline = { 'Pw' : s_df[wind_parm].values, 'Ps' :s_df[pv_parm].values }
        df = pd.DataFrame(data=sline)
    # linearly interpolate along wind and then pv
    else:
#       print('storage_line for {} days '.format(storage_value) )
        x=[]
        y=[]
        last=[]
        # for each wind value ...
        wind_values = df[wind_parm].unique()
#   for i_wind in range(0,14):
#       f_wind = i_wind * 0.5
        for f_wind in wind_values:
#       print('Fwind {} '.format(f_wind) )
        # extract those values with a wind=xs
            is_xs = df[wind_parm] == f_wind
            df_xs = df[is_xs]
        # check storage in range
            if storage_value < df_xs['storage'].max() and storage_value > df_xs['storage'].min():
                # sort them by storage
                df_xs = df_xs.sort_values('storage',ascending=False)
                # interpolate a pv value for the storage
                y_interp = scipy.interpolate.interp1d(df_xs['storage'], df_xs[pv_parm])
                f_pv = y_interp(storage_value)
                # store the points
                x.append(f_wind)
                y.append(f_pv.item())
                # interpolate a 'last' value for the storage
                l_interp = scipy.interpolate.interp1d(df_xs['storage'], df_xs['last'])
                f_last = l_interp(storage_value)

                last.append(f_last.item())
#               print('Point: f_wind {} f_pv {}'.format(f_wind, f_pv.item()) )

#       sline = { 'Pw' : x, 'Ps' :y }
        sline = { 'Pw' : x, 'Ps' :y, 'last' :last }
        df = pd.DataFrame(data=sline)
        df = df.sort_values(['Ps', 'Pw'], ascending=[True, True])
#   print('Line: Pw max {} min {} '.format(df['Pw'].max(), df['Pw'].min() ) )
#   print(df)
    return df

def storage_grid(demand, wind, pv, eta, hourly=False, grid=14, step=0.5, base=0.0, hydrogen=None, method='kf', hist_wind=1.0, hist_pv=1.0):
    print('storage_grid: demand max {} min {} mean {}'.format(demand.max(), demand.min(), demand.mean()) )

    # do one example for a store history
    supply = (wind * hist_wind) + (pv * hist_pv)
    net = demand - supply - base
    sample_hist = storage(net, eta, hydrogen)
    # turn into positive values
    sample_hist = sample_hist - sample_hist.min()
    # get durations for sample store history
    sample_durations = storage_duration(sample_hist)

    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [], 'charge' : [], 'discharge' : [], 'last' : [], 'wind_energy' : [], 'pv_energy' : [] }
    # For hourly the storage will be in hours, so divide by 24 to convert to 
    # days
    if hourly:
        store_factor = 1 / 24
    else:
        store_factor = 1

    # For each percent of PV/Wind
    for i_pv in range(0,grid):
        for i_wind in range(0,grid):
            f_pv = i_pv * step
            f_wind = i_wind * step
            sys.stdout.write('\rCalculating f_pv {:.2f} f_wind {:.2f} '.format(f_pv, f_wind) )

            # energy supply is calculated using the capacity factors
            wind_energy = wind * f_wind
            pv_energy = pv * f_pv
#           supply = (wind * f_wind) + (pv * f_pv)
            supply = wind_energy + pv_energy
            net = demand - supply - base

            #  calculate how much storage we need.
            #  ( only difference is that mp can overfill )
            if method == 'kf':
                store_hist = storage(net, eta, hydrogen)
            else:
                store_hist = storage_mp(net, eta, hydrogen)

            #  store starts off zero and gets more nagative
            #  hence amount of storage needed is minimum value it had.
            if method == 'kf':
                store_size = store_hist.min()
            #  store starts off zero and gets more nagative
            #  but can overfill so its the sum of max and min.
            else:
                store_size = store_hist.min() - store_hist.max()
            storage_days = store_size * store_factor * -1.0
                
            # for mp model not viable unless more energy at the end.
            if method == 'kf' or store_hist.iat[-1] > 0:
                # store last is the same in both cases its just that for
                # mp model it didn't start off full
                store_last = store_hist.iat[-1] * store_factor
                store_remaining = storage_days - store_last * -1.0

                #  rate of charge or discharge in a period
                charge=0.0
                #  TODO discharge needs splitting into hydrogen for boilers and
                #  hydrogen for electricity to estimate generation capacity
                discharge=0.0
                rate = store_hist.diff()
                plus_rates = rate[rate>0]
                if len(plus_rates)>0:
                    charge = plus_rates.max()
                minus_rates = rate[rate<0]
                if len(minus_rates)>0:
                    discharge = minus_rates.min() * -1.0
                # store the results
                results['f_pv'].append(f_pv)
                results['f_wind'].append(f_wind)
                results['storage'].append(storage_days)
                results['last'].append(store_last)
                results['charge'].append(charge)
                results['discharge'].append(discharge)
                results['wind_energy'].append(wind_energy.sum() / demand.sum())
                results['pv_energy'].append(pv_energy.sum() / demand.sum())

    df = pd.DataFrame(data=results)
    return df, sample_hist, sample_durations

# Shift days so that days of the week pattern is continued
#   values         - original baseline demand.
#   baseline_index - index of the baseline ie 2018
#   weather_index  - index of the year to shift to

def shiftdays(values, baseline_index, weather_index):
    # these days we don't want to shift so over write
    #   1 New years day  1st January
    #  89 Good Friday   30th March
    #  92 Easter Monday  2nd April
    # 127 Bank holiday   7th May
    # 148 Bank holiday  28th May
    # 239 Bank holiday  27th August
    # 359 Christmas     25th December
    # 360 Boxing Day    26th December
    special_days = [1, 89, 92, 127, 148, 239, 359, 360]
    # before shifting need set the sepcial days to the nearest
    # ordinary day
    special_values = {}
    for day in special_days:
        special_values[day] = values[day]
        new_day = day + 7
        if day>358:
            new_day = day - 7
        values[day] = values[new_day]
    # Jan 1st 2018 is a Monday
    extra=[]
    day_of_week_weather = weather_index[0].weekday()
    shift = day_of_week_weather
    print('SHIFT {}'.format(shift))
    while day_of_week_weather <7:
        extra.append(values[day_of_week_weather])
        day_of_week_weather +=1
    if shift > 0:
        extra_values = np.array(extra)
        new_values = np.concatenate([extra_values, values[:-len(extra_values)] ])
    else:
        new_values = values
#       
    print('Original {} new {} extra {}'.format(len(values), len(new_values), len(extra) ) )

    # put the special day values back
    for day in special_days:
        new_values[day] = special_values[day]
    
    return new_values

# Compare to contour lines of equal storage based on wind and pv.
def compare_lines(line1, line2):
    line2_copy = line2.copy()
    # put NaNs in then call interpolate
    line2_copy['Pw'] = float("NaN")
#   pv_min = max(line1['Ps'].min(), line2['Ps'].min() )
#   pv_max = min(line1['Ps'].max(), line2['Ps'].max() )
    merged = pd.concat([line1, line2_copy])
    # sort by Ps
    merged = merged.sort_values('Ps',ascending=True)
#   print(merged)
    # calculate the wind values for the pv values from line2 which are 
    # by linear interpolation from the wind value in line1
    merged = merged.interpolate()
#   print(merged)
    merged.dropna(subset=['Pw'], inplace=True)
#   print(merged)
    # get only the values in the original line 2
    merged = pd.merge(merged, line2, how='inner', on=['Ps'])
    # calculate mean values
    wind_diff = (merged['Pw_x'] - merged['Pw_y']).mean()
    ratio1  = (merged['Ps'] / merged['Pw_x']).mean()
    ratio2  = (merged['Ps'] / merged['Pw_y']).mean()

    return wind_diff, ratio1, ratio2

# new storage model which finds pv and wind combinations matching a set list
# of storage values

def storage_grid_new(demand, wind, pv, eta, hourly=False, grid=14, step=0.5, base=0.0, hydrogen=None, constraints='new', hist_wind=1.0, hist_pv=1.0, hist_days=30, threshold=0.01, variable=0.0, contours='med', debug=False):
    print('storage_grid new: constraints {}  demand max {} min {} mean {}'.format(constraints, demand.max(), demand.min(), demand.mean()) )

    # try one example and get the store history
    balanced, sample_hist, variable_total = storage_balance(demand, wind, pv, eta, base, hydrogen, hist_pv, hist_wind, hist_days, constraints, variable, debug)
    print('Sample store history wind {} pv {} days {} balance {}'.format(hist_wind,hist_pv,hist_days,balanced))
    # get durations for sample store history
    sample_durations = storage_duration(sample_hist)

    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [], 'charge' : [], 'discharge' : [], 'last' : [], 'wind_energy' : [], 'pv_energy' : [] }
    # For hourly the storage will be in hours, so divide by 24 to convert to 
    # days
    if hourly:
        store_factor = 1 / 24
    else:
        store_factor = 1

    # threshold for minimum distance between pv values.
#   threshold = 0.01
    # For each contour ...
#   days = [3, 10, 25, 30]
    if contours == 'low':
        days = [0.01, 0.25, 0.5, 1, 3, 10]
    else:
        if contours == 'med':
            days = [3, 10, 25, 30, 40, 60]
        else:
            days = [40, 60, 100]
    for store_size in days:
        # For each percent of PV
        for i_pv in range(0,grid):
            f_pv = i_pv * step
      
            # try different amounts of wind 
            wind_max = grid * step
#           wind_min = max( 1 - f_pv, 0.0 )
            wind_min = 0.0
            f_wind = wind_max
            balanced, store_hist, variable_total = storage_balance(demand, wind, pv, eta, base, hydrogen, f_pv, f_wind, store_size, constraints, variable)
            if not balanced:
                print(' No solution found for {} days, pv {} '.format(store_size, f_pv) )
            else:
                while abs(wind_max - wind_min) > threshold:
                    f_wind = ( wind_max + wind_min ) / 2.0
                    sys.stdout.write('\rCalculating {} days f_pv {:.2f} f_wind {:.2f} '.format(store_size, f_pv, f_wind) )
                    balanced, store_hist, variable_total = storage_balance(demand, wind, pv, eta, base, hydrogen, f_pv, f_wind, store_size, constraints, variable)
                    # wind_max is the always the last one that balanced
                    # wind_min is the always the last one that didn't
                    if balanced:
                        wind_max = f_wind
                    else:
                        wind_min = f_wind

                print(" ")
                print('Got balance at f_wind {} variable {}'.format(wind_max, variable_total) )
                store_last = store_hist.iat[-1] * store_factor
                store_remaining = store_size - store_last * -1.0

                #  rate of charge or discharge in a period
                charge=0.0
                #  TODO discharge needs splitting into hydrogen for boilers and
                #  hydrogen for electricity to estimate generation capacity
                discharge=0.0
                rate = store_hist.diff()
                plus_rates = rate[rate>0]
                if len(plus_rates)>0:
                    charge = plus_rates.max()
                minus_rates = rate[rate<0]
                if len(minus_rates)>0:
                    discharge = minus_rates.min() * -1.0
                # store the results
                results['f_pv'].append(f_pv)
                results['f_wind'].append(wind_max)
                results['storage'].append(store_size)
                results['last'].append(store_last)
                results['charge'].append(charge)
                results['discharge'].append(discharge)
                wind_energy = wind * wind_max
                pv_energy = pv * f_pv
                results['wind_energy'].append(wind_energy.sum() / demand.sum())
                results['pv_energy'].append(pv_energy.sum() / demand.sum())

    df = pd.DataFrame(data=results)
    return df, sample_hist, sample_durations

def storage_balance(demand, wind, pv, eta, base, hydrogen, f_pv, f_wind, capacity, constraints='new', variable=0.0, debug=False):

    # settings
    if constraints == 'new':
        # store starts 70% full and ends greater than 70% full
        store_start = capacity * 0.7
        store_end = capacity * 0.7
    else:
        if constraints == 'old':
            # store starts full and ends 97% full
            store_start = capacity
            store_end = capacity * 0.97
        else:
            fraction = float(constraints)
            store_start = capacity * fraction
            store_end = capacity * fraction


    # energy supply is calculated using the capacity factors
    wind_energy = wind * f_wind
    pv_energy = pv * f_pv
    supply = wind_energy + pv_energy
    net_demand = demand - supply - base
    history = pd.Series(np.zeros(len(net_demand)), index=net_demand.index, name='store')
    store = store_start
    eta_charge = eta
    eta_discharge = eta
    variable_total = 0.0
    count=0
    if debug:
        print('DEBUG f_wind {} f_pv {} base {} variable {} store_end {}'.format(f_wind, f_pv, base, variable, store_end))

    for index, net_value in net_demand.items():
        value = net_value
        # variable capacity generation eg biomass, gas:
        #  if demand exceeds supply add up to a maximum of 'variable'
        if value > 0.0:
            variable_supplied = variable
            value -= variable
            if value < 0.0: 
                variable_supplied = variable + value
                value = 0.0
            variable_total += variable_supplied

        if debug:
            print('Count {} net_value {} variable_supplied {} store {}'.format(count, net_value, variable_supplied, store))
        # Note: both subtract because value is negative in the 2nd one!
        if value > 0.0:        # demand exceeds supply : take from store
                               # discharge, so divide by eta - take more out
            store = store - value / eta_discharge
            # stop because we don't have enough storage
            if store<0:
                balanced=False
                return balanced, history, variable_total
        else:                  # supply exceeds demand : add to store
                               # charge, so multiply by eta - put less in
            store = store - value * eta_charge
            # don't overfill the store
            if store>capacity:
                store=capacity
        # take hydrogen out of the store
        if hydrogen is not None:
            store = store - hydrogen.iat[count] / eta_discharge
            if store<0:
                balanced=False
                return balanced, history, variable_total
        history.iat[count] = store
        count += 1
    # ensure store meets the end constraint within tolerence
    balanced = store>=store_end - (store_end * 0.001)
    return balanced, history, variable_total

# Given the 2018 baseline, create a baseline for the given year accounting
# for leap years and shifting weekly pattern if requested
#
# baseline   - pd.Series - baseline electricity demand
# year       - 4 digit year 
# year_index - index for the year
# shift      - if the weekly pattern should be shifted 
# hourly     - if the time series is hourly

def ref_baseline(baseline, year, year_index, shift=False, hourly=True):

    if calendar.isleap(year):
        # leap year
        # create a feb 29th by interpolating between feb 28th and Mar 1st
        # find doy for feb 28th ( 31 days in jan )
        ordinary_year = baseline.values
        feb28 = 31 + 28
        if hourly:
            print(baseline)
            feb28 = baseline['2018-02-28 00:00:00+00:00' : '2018-02-28 23:00:00+00:00'].values
            mar1 = baseline['2018-03-01 00:00:00+00:00' : '2018-03-01 23:00:00+00:00'].values
            feb29 = np.add(feb28, mar1) * 0.5
            year_values = np.concatenate([baseline['2018-01-01 00:00:00+00:00' : '2018-02-28 23:00:00+00:00'].values, feb29,  baseline['2018-03-01 00:00:00+00:00' : '2018-12-31 23:00:00+00:00'].values])
            print(year_values)
        else:
            feb29 = (ordinary_year[feb28-1] + ordinary_year[feb28]) * 0.5
            feb29a = np.array([feb29])
            year_values = np.concatenate([ordinary_year[0:feb28], feb29a, ordinary_year[feb28:] ] )

    else:
    # ordinary year
        year_values = baseline.values

    # Shift for days of the week here to match weather year
    if shift:
        year_values = shiftdays(year_values, baseline.index, year_index)

    # Convert to a series
    year_baseline = pd.Series(year_values, index=year_index)

    return year_baseline

# Given a years baseline, convert it into a 2018 baseline accounting
# for leap years and shifting weekly pattern if requested
#
# baseline   - pd.Series - baseline electricity demand
# year       - 4 digit year 
# year_index - index for the year
# shift      - if the weekly pattern should be shifted 
# hourly     - if the time series is hourly

def year_baseline(baseline, year, ref_index, shift=False, hourly=True):

    if calendar.isleap(year):
        # leap year
        # remove feb 29th
        # find doy for feb 28th ( 31 days in jan )
        ordinary_year = baseline.values
        feb28 = 31 + 28
        if hourly:
            year_values = np.concatenate([baseline['{}-01-01 00:00'.format(year) : '{}-02-28 23:00'.format(year)].values, baseline['{}-03-01 00:00'.format(year) : '{}-12-31 23:00'.format(year)].values])
        else:
            year_values = np.concatenate([ordinary_year[0:feb28], ordinary_year[feb28:] ] )

    else:
    # ordinary year
        year_values = baseline.values

    # Shift for days of the week here to match weather year
    if shift:
        year_values = shiftdays_match(year_values, ref_index, baseline.index)

    # Convert to a series
    year_baseline = pd.Series(year_values, index=ref_index)

    return year_baseline

def remove_feb29(series, year, hourly=True):

    if calendar.isleap(year):
        # leap year
        # remove feb 29th
        # find doy for feb 28th ( 31 days in jan )
        ordinary_year = series.values
        feb28 = 31 + 28
        if hourly:
            year_values = np.concatenate([series['{}-01-01 00:00'.format(year) : '{}-02-28 23:00'.format(year)].values, series['{}-03-01 00:00'.format(year) : '{}-12-31 23:00'.format(year)].values])
        else:
            year_values = np.concatenate([ordinary_year[0:feb28], ordinary_year[feb28:] ] )

    else:
    # ordinary year
        year_values = series.values

    return year_values

# Shift days so that days of the week pattern is matched. 
#   values         - original baseline demand.
#   baseline_index - index of the baseline ie 2018
#   weather_index  - index of the year to shift to

def shiftdays_match(values, baseline_index, weather_index, hourly=True):
    # these days we don't want to shift so over write
    #   1 New years day  1st January
    #  89 Good Friday   30th March
    #  92 Easter Monday  2nd April
    # 127 Bank holiday   7th May
    # 148 Bank holiday  28th May
    # 239 Bank holiday  27th August
    # 359 Christmas     25th December
    # 360 Boxing Day    26th December
    special_days = [1, 89, 92, 127, 148, 239, 359, 360]
    # before shifting need set the sepcial days to the nearest
    # ordinary day  TODO day-1 index is 0! AND hourly !!
    special_values = {}
    for day in special_days:
        special_values[day] = values[day]
        new_day = day + 7
        if day>358:
            new_day = day - 7
        values[day] = values[new_day]
    # Jan 1st 2018 is a Monday
    extra=[]
    day_of_week_weather = weather_index[0].weekday()
    shift = day_of_week_weather
#   print('SHIFT MATCH {}'.format(shift))
    if shift > 0:
        day_copy = 7 - shift
        while day_copy <7:
            if hourly:
                hour_start = day_copy*24
                for i in range(24):
                    extra.append(values[hour_start+i])
            else:
                extra.append(values[day_copy])
#           print('Appending day {} '.format(day_copy))
            day_copy +=1
        extra_values = np.array(extra)
#       print(extra)
        new_values = np.concatenate([extra_values, values[:-len(extra_values)] ])
    else:
        new_values = values
#       
#   print('Original {} new {} extra {}'.format(len(values), len(new_values), len(extra) ) )

    # put the special day values back
    for day in special_days:
        new_values[day] = special_values[day]
    
    return new_values

def s_bucket(s, increment):
    b = math.floor(s / increment )
    return b

def storage_duration(store_hist):
    increment=0.5       # bucket size
    durations=[]        # time that store is above this size
    starts=[]           # time that store last reached above this size
    nb=0
    b_size=0
    maxs = store_hist.max()
    # initialise durations for each bucket to zero
    while b_size<maxs:
        d = len(store_hist[(store_hist>b_size) & (store_hist<=b_size+increment)])
        durations.append(d)
        nb+=1
        b_size += increment
    nb+=1
    durations.append(0)

    d_series = pd.Series(durations, name='duration', index=np.arange(0.0,nb*increment,increment))
    d_series.index.name = 'size'
#   print(d_series)
#   d_series = d_series.diff()
#   d_series = d_series.iloc[1:]
#   d_series = d_series * -1.0
    print(d_series)
    return d_series
