# Functions related to how much storage is required for different amounts of
# PV and Wind. Includes calculating lines of constant value

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

def storage_line(df, storage_value, method='interp1', wind_parm='f_wind', pv_parm='f_pv', variable='storage'):
    # just take data points with storage values in a range
    if method=='threshold':
        threshold = storage_value * 0.05
        s_df = df[(df[variable] < storage_value + threshold) & (df[variable] > storage_value - threshold)]
        if len(s_df.index)<2:
            print('WARNING: {} line {} days had only {} points'.format(variable, storage_value, len(s_df.index) ) )
        sline = { 'f_wind' : s_df[wind_parm].values, 'f_pv' :s_df[pv_parm].values }
        df = pd.DataFrame(data=sline)
    # linearly interpolate along wind and then pv
    else:
#       print('storage_line for {} days '.format(storage_value) )
        variables = ['f_wind', 'f_pv', 'storage', 'last', 'wind_energy', 'pv_energy', 'discharge', 'base', 'cost', 'charge_rate', 'discharge_rate', 'charge', 'variable_energy', 'variable', 'gw_wind', 'gw_pv', 'fraction', 'energy', 'yearly_store_min', 'yearly_store_max', 'lost', 'slost']
        if variable not in variables or wind_parm not in variables or pv_parm not in variables:
            print('ERROR variable : {} not in variables list'.format(variable) )
            quit()
        value_lists={}
        for var in variables:
            value_lists[var] = []

#       value_lists[wind_parm] = value_lists.pop(wind_parm)
#       value_lists[pv_parm] = value_lists.pop(pv_parm)
        if method == 'both' or method== 'sboth':
            storage_interpolate(value_lists, pv_parm, variable, df, storage_value, variables)
        storage_interpolate(value_lists, wind_parm, variable, df, storage_value, variables)
        df = pd.DataFrame(data=value_lists)
        if len(df)>1:
            df = df.sort_values([pv_parm, wind_parm], ascending=[True, True])
            if method == 'smooth' or method == 'sboth':
                df = sort_df_contour(df, pv_parm, wind_parm)
    return df

def sort_df_contour(df, pv_parm, wind_parm):
    # convert data frame to list of dicts
    LD = df.to_dict('records')

    # define a start point
    start = LD[0]

    # order the points to the closest one is adjacent
    LD = optimized_path(LD, start, pv_parm, wind_parm)

    # convert list of dicts to dict of lists
    v = {k: [dic[k] for dic in LD] for k in LD[0]}

    # convert back to a DataFrame
    dfs = pd.DataFrame(data=v)

    return dfs

# order the points in terms of the closest point to the next one
# so that the line is smooth
def optimized_path(coords, start, pv_parm, wind_parm):
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: storage_grid_distance(path[-1], x, pv_parm, wind_parm))
        path.append(nearest)
        pass_by.remove(nearest)
    return path

# calculate the 'distance' between 2 points
def storage_grid_distance(P1, P2, pv_parm, wind_parm):
    return ((P1[pv_parm] - P2[pv_parm])**2 + (P1[wind_parm] - P2[wind_parm])**2) ** 0.5

def storage_interpolate(value_lists, wind_parm, variable, df, storage_value, variables):
    # for each wind value ...
    wind_values = df[wind_parm].unique()
    for f_wind in wind_values:
    # extract those values with a wind=xs
        is_xs = df[wind_parm] == f_wind
        df_xs = df[is_xs]
        # check storage in range
        if storage_value < df_xs[variable].max() and storage_value > df_xs[variable].min():
            # sort them by storage
            df_xs = df_xs.sort_values(variable,ascending=False)

            # add the wind and storage values
            value_lists[wind_parm].append(f_wind)
            value_lists[variable].append(storage_value)

            # add the values of the other variables
            for var in variables:
                if var not in [wind_parm, variable]:
                    # interpolate a value for the storage
                    y_interp = scipy.interpolate.interp1d(df_xs[variable], df_xs[var])
                    var_value = y_interp(storage_value)
                    value_lists[var].append(var_value.item())

def storage_grid_config(demand, wind, pv, eta, hourly,base, variable, hydrogen, method, f_wind, f_pv, threshold, constraints, store_max, debug, results):

    # For hourly the storage will be in hours, so divide by 24 to convert to 
    # days
    if hourly:
        store_factor = 1 / 24
    else:
        store_factor = 1

    # energy supply is calculated using the capacity factors multiplied
    # by the capacity
    wind_energy = wind * f_wind
    pv_energy = pv * f_pv
    supply = wind_energy + pv_energy
    net = demand - supply - base

    variable_total = 0
    #  calculate how much storage we need.
    #  ( only difference is that mp can overfill )
    if method == 'kf':
        store_hist = storage(net, eta, hydrogen) * -1.0
        store_size = store_hist.max()
    else:
        if method == 'mp':
            store_hist = storage_mp(net, eta, hydrogen) * -1.0
            store_size = store_hist.max() - store_hist.min()
        else:
            store_size, store_hist, variable_total = storage_all(demand, wind, pv, base, variable, eta, hydrogen, f_wind, f_pv, threshold, constraints, store_max / store_factor, debug)

                
    # kf  model is always viable.
    # mp  model not viable unless more energy at the end.
    # all model returns None if didn't find valid balance
    if method == 'kf' or (method == 'mp' and store_hist.iat[-1] > 0) or (method == 'all' and store_size):
        # storage size in days
        storage_days = store_size * store_factor
        # amount remaining in store at the end in days
        store_last = store_hist.iat[-1] * store_factor
        # percentage remaining
        store_remaining = store_last / storage_days

        #  rate of charge or discharge in a period
        charge_rate=0.0
        charge=0.0
        #  TODO discharge needs splitting into hydrogen for boilers and
        #  hydrogen for electricity to estimate generation capacity
        discharge_rate=0.0
        discharge=0.0
        rate = store_hist.diff()
        plus_rates = rate[rate>0]
        if len(plus_rates)>0:
            charge_rate = plus_rates.max()
            charge = plus_rates.sum()
        minus_rates = rate[rate<0]
        if len(minus_rates)>0:
            discharge_rate = minus_rates.min() * -1.0
            discharge = minus_rates.sum() * -1.0

        # store at the start of the year
        yearly_store = store_hist.resample('Y').first()
        year_store_min = yearly_store.min() / store_size
        year_store_max = yearly_store.max() / store_size

        # store the results
        results['f_pv'].append(f_pv)
        results['f_wind'].append(f_wind)
        results['storage'].append(storage_days)
        results['last'].append(store_remaining)
        results['charge_rate'].append(charge_rate)
        results['discharge_rate'].append(discharge_rate)
        results['charge'].append(charge)
        results['discharge'].append(discharge)
        results['wind_energy'].append(wind_energy.sum() / demand.sum())
        results['pv_energy'].append(pv_energy.sum() / demand.sum())
        results['variable_energy'].append(variable_total)
        results['yearly_store_min'].append(year_store_min)
        results['yearly_store_max'].append(year_store_max)

    return store_hist

def storage_grid(demand, wind, pv, eta, hourly=False, npv=14, nwind=14, step=0.5, base=0.0, variable=0.0, hydrogen=None, method='kf', hist_wind=1.0, hist_pv=1.0, threshold=0.01, constraints='new', debug=False, store_max=60 ):
    print('storage_grid: demand max {} min {} mean {}'.format(demand.max(), demand.min(), demand.mean()) )
    # For hourly the storage will be in hours, so divide by 24 to convert to 
    # days
    if hourly:
        store_factor = 1 / 24
    else:
        store_factor = 1

    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [], 'charge_rate' : [], 'discharge_rate' : [], 'charge' : [], 'discharge' : [], 'last' : [], 'wind_energy' : [], 'pv_energy' : [], 'variable_energy' : [], 'yearly_store_min' : [], 'yearly_store_max' : [] }

    # do one example for a store history
    if hist_wind>0 or hist_pv>0:
        sample_hist = storage_grid_config(demand, wind, pv, eta, hourly, base, variable, hydrogen, method, hist_wind, hist_pv, threshold, constraints, store_max, debug, results)
        if len(results['storage']) == 0:
            print('Sample wind {} pv {} no solution found'.format(hist_wind, hist_pv) )
        else:
            print('Sample wind {} pv {} needs {} days'.format(hist_wind, hist_pv, results['storage'][0] ) )

    else:

        # For each percent of PV/Wind
        for i_pv in range(0,npv):
            for i_wind in range(0,nwind):
                f_pv = i_pv * step
                f_wind = i_wind * step
                sys.stdout.write('\rCalculating f_pv {:.2f} f_wind {:.2f} '.format(f_pv, f_wind) )

                sample_hist = storage_grid_config(demand, wind, pv, eta, hourly, base, variable, hydrogen, method, f_wind, f_pv, threshold, constraints, store_max, debug, results)

    # get durations for sample store history
    sample_durations = storage_duration(sample_hist)

    print(" ")
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

# Calculate configuration cost assuming
#   dispatchable (gas)  85
#   wind                50   (mean of offshore 57 and onshore 46)
#   solar               44
#   base (nuclear)     123
#   storage (hydrogen) 100
# config base, wind, pv, variable, storage

def configuration_cost(config):
    # Day of energy in MWh
    one_day = 818386
    # Cost per MWh
    cost_variable = 85
    cost_wind = 50
    cost_solar = 44
    cost_base = 123
    cost_storage = 100
    #  normalise to the 2018 day of generation
    cost = (config['f_wind'] * cost_wind) + (config['f_pv'] * cost_solar) + (config['storage'] * cost_storage) + (config['base'] * cost_base) + (config['variable'] * cost_variable)

    config['cost'] = cost * one_day * 1e-6

# Calculate generation cost
# Inputs:
#  config
#  stype
#  one_day  - mean daily energy in Mwh
#  n_years
#  hourly
#  shore
#  model
# Output:
#  cost £/kWh

def generation_cost(config,stype,one_day,n_years=1,hourly=False,shore='both', model='A'):
#   print(stype,one_day,n_years,hourly,shore,model)
    if model == 'A' :
        generation_cost_a(config,stype,one_day,n_years,hourly,shore)
    else:
        generation_cost_b(config,stype,one_day,n_years,hourly,shore,model)

def generation_cost_a(config,stype,one_day,n_years=1,hourly=False,shore='both', model='A'):
    # number of days
    number_of_days = n_years * 365.25
    # exchange rate euro to £
    eu2p = 0.87
    # Day of energy in MWh
#   one_day = 818386
    # Fix Cost per kW capacity in £
    cost_variable = 692
    if shore == 'both':
        cost_wind = ( 2638.7 + 1544.71 ) / 2
    else:
        if shore == 'off':
            cost_wind = 2638.7
        else:
            cost_wind = 1544.71
    cost_solar = 921.9
    cost_base = 3089.5
    # Variarble Cost per kWh energy generated in £
    gen_wind = 0.005
    gen_solar = 0.0
    gen_base = 0.0055
    gen_variable = 0.0266

    # storage cost
    cost_storage = 0.1 * eu2p
    # Pumped storage
    if stype=='pumped':
        cost_storage = 0.3 * eu2p

    storage_kwh = config['storage'] * one_day * 1e3 * number_of_days
    storage_cost = storage_kwh * cost_storage

    #  fixed generation costs based on capacity
    hourly_factor = 24
    if hourly:
        hourly_factor = 1
    wind_kw = config['f_wind'] * one_day * 1e3 / hourly_factor
    pv_kw = config['f_pv'] * one_day * 1e3 / hourly_factor
    base_kw = config['base'] * one_day * 1e3 / hourly_factor
    variable_kw = config['variable'] * one_day * 1e3 / hourly_factor
    fixed_cost = (wind_kw * cost_wind + pv_kw * cost_solar + base_kw * cost_base + variable_kw * cost_variable) * n_years

    #  variable generation costs. 
    #  base load is assumed always on so we just use the capacity
    pv_kwh = config['pv_energy'] * one_day * 1e3 * number_of_days
    wind_kwh = config['wind_energy'] * one_day * 1e3 * number_of_days
    base_kwh = config['base'] * one_day * 1e3 * number_of_days
    variable_kwh = config['variable'] * one_day * 1e3 * number_of_days
    variable_cost = (wind_kwh * gen_wind) + (base_kwh * gen_base) + ( variable_kwh * gen_variable )
    # overall cost in £
    cost_value = fixed_cost + variable_cost + storage_cost
    # cost per MWh ( divide by the total demand )
    config['cost'] = cost_value / ( one_day * 1e3 * number_of_days )

# Calculate generation cost based on [capacity v renewable gen UK]

def generation_cost_b(config,stype,one_day,n_years=1,hourly=False,shore='both', model='A' ):
    # number of days
    number_of_days = n_years * 365.25
    # Fixed of storage capacity
    alpha = 3            # £/kWh
    beta  = 300          # £/kW
    life_time = 30
    if stype=='hydrogen':
        alpha = 0.67
        betac  = 1100
        betad  = 450
        life_time = 30
    if stype=='pumped':
        alpha = 1188
        beta  = 66.4
        life_time = 30
    # LCOE in £ per MWh
    gen_offshore = 57.5
    gen_onshore = 46
    gen_fraction = 0.5
    if model == 'C':
        gen_offshore = 40.0
        gen_onshore = 44.0
        gen_fraction = 0.77
    if shore == 'both':
        gen_wind = gen_offshore*gen_fraction + gen_onshore*(1-gen_fraction)
    else:
        if shore == 'off':
            gen_wind = gen_offshore
        else:
            gen_wind = gen_onshore
    gen_solar = 60
    gen_base = 100
    gen_variable = 66
    if model == 'C':
        gen_solar = 33
        gen_variable = 120

    storage_kwh = days2capacity(config['storage'], one_day * 1e3, False)
    storage_cap = storage_kwh * alpha
    if stype=='hydrogen':
        ratec_kw = days2capacity(config['charge_rate'], one_day * 1e3, True)
        rated_kw = days2capacity(config['discharge_rate'], one_day * 1e3, True)
        storage_pow = ratec_kw * betac + rated_kw * betad
    else:
        rate_kw = days2capacity(config[['discharge_rate','charge_rate']].max(axis=1), one_day * 1e3, True)
        storage_pow = rate_kw * beta
    c_store = storage_cap + storage_pow
    storage_cost = c_store * (n_years /life_time)

    #  variable generation costs. 
    #  base load is assumed always on so we just use the capacity
    pv_mwh = days2energy(config['pv_energy'], one_day, number_of_days, False)
    wind_mwh = days2energy(config['wind_energy'] , one_day , number_of_days, False)
    base_mwh = days2energy(config['base'] , one_day , number_of_days, False)
    variable_mwh = days2energy(config['variable'] , one_day , number_of_days, False)

    variable_cost = (wind_mwh * gen_wind) + (pv_mwh * gen_solar) + (base_mwh * gen_base) + ( variable_mwh * gen_variable )

    # overall cost in £
    cost_value = variable_cost + storage_cost
    # cost per kWh ( divide by the total demand )
    # ( note paper with the cost model has in MWh )
    config['cost'] = cost_value / ( one_day * number_of_days * 1e3 )

# get the minimum energy point in a storage line
def min_point(storage_line, variable='energy', wind_var='f_wind', pv_var='f_pv'):
#   configuration_cost(storage_line)
    min_value = storage_line[variable].min()
    min_points = storage_line[storage_line[variable]==min_value]
    values = {
      'storage'        : min_points['storage'].mean(),
      'f_wind'         : min_points[wind_var].mean(),
      'f_pv'           : min_points[pv_var].mean(),
      'np'             : len(min_points),
      'energy'         : min_points['energy'].mean(),
      'fraction'       : min_points['fraction'].mean(),
      'lost'           : min_points['lost'].mean(),
      'slost'          : min_points['slost'].mean(),
      'discharge'      : min_points['discharge'].mean(),
      'charge'         : min_points['charge'].mean(),
      'cost'           : min_points['cost'].mean(),
      'charge_rate'    : min_points['charge_rate'].mean(),
      'discharge_rate' : min_points['discharge_rate'].mean(),
      'last'           : min_points['last'].mean(),
      'wind_energy'    : min_points['wind_energy'].mean(),
      'pv_energy'      : min_points['pv_energy'].mean(),
      'variable_energy' : min_points['variable_energy'].mean(),
      'base' : min_points['base'].mean(),
      'variable' : min_points['variable'].mean(),
      'gw_wind' : min_points['gw_wind'].mean(),
      'gw_pv' : min_points['gw_pv'].mean(),
      'yearly_store_min' : min_points['yearly_store_min'].mean(),
      'yearly_store_max' : min_points['yearly_store_max'].mean(),
    }
    return values

# Compare to contour lines of equal storage based on wind and pv.
def compare_lines(line1, line2):
    line2_copy = line2.copy()
    # put NaNs in then call interpolate
    line2_copy['f_wind'] = float("NaN")
#   pv_min = max(line1['f_pv'].min(), line2['f_pv'].min() )
#   pv_max = min(line1['f_pv'].max(), line2['f_pv'].max() )
    merged = pd.concat([line1, line2_copy])
    # sort by Ps
    merged = merged.sort_values('f_pv',ascending=True)
#   print(merged)
    # calculate the wind values for the pv values from line2 which are 
    # by linear interpolation from the wind value in line1
    merged = merged.interpolate()
#   print(merged)
    merged.dropna(subset=['f_wind'], inplace=True)
#   print(merged)
    # get only the values in the original line 2
    merged = pd.merge(merged, line2, how='inner', on=['f_pv'])
    # calculate mean values
    wind_diff = (merged['f_wind_x'] - merged['f_wind_y']).mean()
    ratio1  = (merged['f_pv'] / merged['f_wind_x']).mean()
    ratio2  = (merged['f_pv'] / merged['f_wind_y']).mean()

    return wind_diff, ratio1, ratio2

def get_point(df, wind_val, pv_val, wind_var, pv_var):
    if df[wind_var].min() > wind_val or df[wind_var].max() < wind_val or df[pv_var].min() > pv_val or df[pv_var].max() < pv_val:
        print("WARNING: can't interpolat to {} {} , {} {}".format(wind_var, wind_val, pv_var, pv_val))

    new_row = df.head(1).copy()
    new_row_ind = new_row.index[0]
    for col in new_row.columns:
        new_row.loc[new_row_ind, col] = float("Nan")
        
    new_row.loc[new_row_ind, wind_var] = wind_val
    new_row.loc[new_row_ind, pv_var] = pv_val
    df = df.append(new_row, ignore_index = True)
#   print(df)
    df = df.sort_values([pv_var, wind_var], ascending=[True, True])
    # fill in the NaNs by interpolation
    df = df.interpolate()
    # get the point
    winds = df[df[wind_var]==wind_val]
    pvs = winds[winds[pv_var]==pv_val].copy()
#   pvs = pvs.rename(columns={pv_var: 'pv', wind_var: 'wind', 'storage': 'days'})
    pvs['np'] = 1
    row = pvs.to_dict('records')
#   print(row[0])
    return row[0]

# new storage model which finds pv and wind combinations matching a set list
# of storage values

def storage_grid_new(demand, wind, pv, eta, hourly=False, npv=14, nwind=14, step=0.5, base=0.0, hydrogen=None, constraints='new', hist_wind=1.0, hist_pv=1.0, hist_days=30, threshold=0.01, variable=0.0, contours='med', debug=False):
    print('storage_grid new: constraints {}  demand max {} min {} mean {}'.format(constraints, demand.max(), demand.min(), demand.mean()) )

    # try one example and get the store history
    balanced, sample_hist, variable_total = storage_balance(demand, wind, pv, eta, base, hydrogen, hist_pv, hist_wind, hist_days, constraints, variable, debug)
    print('Sample store history wind {} pv {} days {} balance {} max {} min {} start {} '.format(hist_wind,hist_pv,hist_days,balanced, sample_hist.max(), sample_hist.min(), sample_hist.iloc[0] ))
    # get durations for sample store history
    sample_durations = storage_duration(sample_hist)

    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [], 'charge_rate' : [], 'discharge_rate' : [], 'charge' : [], 'discharge' : [], 'last' : [], 'wind_energy' : [], 'pv_energy' : [] }
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
            if contours == 'large':
                days = [40, 60, 100]
            else:
                days = [int(contours)]
    for store_days in days:
        # converts to hourly if needed.
        store_size = store_days / store_factor
        # For each percent of PV
        for i_pv in range(0,npv):
            f_pv = i_pv * step
      
            # try different amounts of wind 
            wind_max = nwind * step
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
                charge_rate=0.0
                #  TODO discharge needs splitting into hydrogen for boilers and
                #  hydrogen for electricity to estimate generation capacity
                discharge=0.0
                discharge_rate=0.0
                rate = store_hist.diff()
                plus_rates = rate[rate>0]
                if len(plus_rates)>0:
                    charge_rate = plus_rates.max()
                    charge = plus_rates.sum()
                minus_rates = rate[rate<0]
                if len(minus_rates)>0:
                    discharge_rate = minus_rates.min() * -1.0
                    discharge = minus_rates.sum() * -1.0
                # store the results
                results['f_pv'].append(f_pv)
                results['f_wind'].append(wind_max)
                results['storage'].append(store_size * store_factor)
                results['last'].append(store_last)
                results['charge_rate'].append(charge_rate)
                results['discharge_rate'].append(discharge_rate)
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
        print('DEBUG f_wind {} f_pv {} base {} variable {} store_end {} store_start {} '.format(f_wind, f_pv, base, variable, store_end, store_start))

    for index, net_value in net_demand.items():
        value = net_value
        # variable capacity generation eg biomass, gas:
        #  if demand exceeds supply add up to a maximum of 'variable'
        variable_supplied = 0
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
#           print(baseline)
            feb28 = baseline['2018-02-28 00:00:00+00:00' : '2018-02-28 23:00:00+00:00'].values
            mar1 = baseline['2018-03-01 00:00:00+00:00' : '2018-03-01 23:00:00+00:00'].values
            feb29 = np.add(feb28, mar1) * 0.5
            year_values = np.concatenate([baseline['2018-01-01 00:00:00+00:00' : '2018-02-28 23:00:00+00:00'].values, feb29,  baseline['2018-03-01 00:00:00+00:00' : '2018-12-31 23:00:00+00:00'].values])
#           print(year_values)
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
    return d_series

# new storage model to modify the storage until it balances
def storage_all(demand, wind, pv, base, variable, eta, hydrogen, f_wind, f_pv, threshold=0.01, constraints = 'new', store_max = 60, debug=False):
    # try different amounts of storage 
    store_min = 0.0
    store_size = store_max
    balanced, store_hist, variable_total = storage_balance(demand, wind, pv, eta, base, hydrogen, f_pv, f_wind, store_size, constraints, variable, debug)
    if not balanced:
        print(' No solution found for {} wind, pv {} at max {} '.format(f_wind, f_pv, store_max) )
        store_size = None
    else:
        while abs(store_max - store_min) > threshold:
            store_size = ( store_max + store_min ) / 2.0
            sys.stdout.write('\rCalculating f_pv {:.2f} f_wind {:.2f} days {:.2f} '.format(f_pv, f_wind, store_size) )
            balanced, last_hist, variable_total = storage_balance(demand, wind, pv, eta, base, hydrogen, f_pv, f_wind, store_size, constraints, variable, debug)
            # store_max is the always the last one that balanced
            # store_min is the always the last one that didn't
            if balanced:
                store_max = store_size
                store_hist = last_hist
            else:
                store_min = store_size

        print(" ")
        print('Got balance at days {:.2f}'.format(store_max) )

        # this is always the last one that balanced.
        store_size = store_max
    return store_size, store_hist, variable_total

# convert energy to days
def capacity2days(energy, mean_daily_energy, hourly=False):
    normalise_factor = mean_daily_energy
    if hourly:
        normalise_factor = normalise_factor / 24.0
    days = energy / normalise_factor
    return days


def days2capacity(days, mean_daily_energy, hourly=False):
    normalise_factor = mean_daily_energy
    if hourly:
        normalise_factor = normalise_factor / 24.0
    energy = days * normalise_factor
    return energy
    
def days2energy(days, mean_daily_energy, number_of_days, hourly=False):
    normalise_factor = mean_daily_energy
    if hourly:
        normalise_factor = normalise_factor / 24.0
    energy = days * normalise_factor * number_of_days
    return energy
    
# convert energy to days
def energy2days(energy, mean_daily_energy, number_of_days, hourly=False):
    normalise_factor = mean_daily_energy
    if hourly:
        normalise_factor = normalise_factor / 24.0
    days = energy / ( normalise_factor * number_of_days )
    return days
