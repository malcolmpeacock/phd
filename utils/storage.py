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
    history = net_demand.copy()
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
    history = net_demand.copy()
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
#               if storage_value==25:
#                   print(df_xs[['f_pv', 'f_wind', 'storage']])
                # interpolate a pv value for the storage
                y_interp = scipy.interpolate.interp1d(df_xs['storage'], df_xs[pv_parm])
                f_pv = y_interp(storage_value)
                # store the points
                x.append(f_wind)
                y.append(f_pv.item())
#               print('Point: f_wind {} f_pv {}'.format(f_wind, f_pv.item()) )

        sline = { 'Pw' : x, 'Ps' :y }
        df = pd.DataFrame(data=sline)
        df = df.sort_values(['Ps', 'Pw'], ascending=[True, True])
#   print('Line: Pw max {} min {} '.format(df['Pw'].max(), df['Pw'].min() ) )
#   print(df)
    return df

def storage_grid(demand, wind, pv, eta, hourly=False, grid=14, step=0.5, base=0.0, hydrogen=None, method='kf'):
    print('storage_grid: demand max {} min {} mean {}'.format(demand.max(), demand.min(), demand.mean()) )
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
    return df

# Shift days so that days of the week pattern is continued
#   baseline    - original baseline demand.
#   synthesised - modified baseline including heat and with index
#                 from weather year.

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
    pv_min = max(line1['Ps'].min(), line2['Ps'].min() )
    pv_max = min(line1['Ps'].max(), line2['Ps'].max() )
    merged = pd.concat([line1, line2_copy])
    # sort by Ps
    merged = merged.sort_values('Ps',ascending=True)
    print(merged)
    merged = merged.interpolate()
    # get only the values in the original line 2
    merged = pd.merge(merged, line2, how='inner', on=['Ps'])
    # calculate mean values
    wind_diff = (merged['Pw_x'] - merged['Pw_y']).mean()
    ratio1  = (merged['Ps'] / merged['Pw_x']).mean()
    ratio2  = (merged['Ps'] / merged['Pw_y']).mean()

    return wind_diff, ratio1, ratio2
