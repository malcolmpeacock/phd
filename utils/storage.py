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

# calculate storage

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
        # record the size of the store
        history.iat[count] = store
        count += 1
    return history

# constant storage line

def storage_line(df,storage_value, wind_parm='f_wind', pv_parm='f_pv'):
#   print('storage_line for {} days '.format(storage_value) )
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
#           print(df_xs)
            # interpolate a pv value for the storage
            y_interp = scipy.interpolate.interp1d(df_xs['storage'], df_xs[pv_parm])
            f_pv = y_interp(storage_value)
            # store the points
            x.append(f_wind)
            y.append(f_pv.item())
#           print('Point: f_wind {} f_pv {}'.format(f_wind, f_pv.item()) )

    sline = { 'Pw' : x, 'Ps' :y }
    df = pd.DataFrame(data=sline)
#   print('Line: Pw max {} min {} '.format(df['Pw'].max(), df['Pw'].min() ) )
#   print(df)
    return df

def storage_grid(demand, wind, pv, eta, hourly=False, grid=14, step=0.5, base=0.0, hydrogen=None):
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
            print('Calculating f_pv {} f_wind {} '.format(f_pv, f_wind) )
            # energy supply is calculated using the capacity factors
            wind_energy = wind * f_wind
            pv_energy = pv * f_pv
#           supply = (wind * f_wind) + (pv * f_pv)
            supply = wind_energy + pv_energy
            net = demand - supply - base

            #  calculate how much storage we need.
            store_hist = storage(net, eta, hydrogen)
            #  store starts of zero and gets more nagative
            #  hence amount of storage needed is minimum value it had.
            store_size = store_hist.min()
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
            storage_days = store_size * store_factor * -1.0
            store_last = store_hist.iat[-1] * store_factor * -1.0
            if store_last <0:
                store_last = 0
#           if store_size == store_hist.iat[-1] or storage_days>200:
#               storage_days = 200
#           leave negative values in so we can look at excess.
#           if storage_days <0.0:
#               storage_days = 0.0
            results['storage'].append(storage_days)
            results['last'].append(store_last)
            results['charge'].append(charge)
            results['discharge'].append(discharge)
            results['wind_energy'].append(wind_energy.sum() / demand.sum())
            results['pv_energy'].append(pv_energy.sum() / demand.sum())

            # yearly values
#           store_max_yearly = store_hist.resample('Y', axis=0).max()
#           store_min_yearly = store_hist.resample('Y', axis=0).min()
#           store_diff_yearly = store_max_yearly - store_max_yearly

    df = pd.DataFrame(data=results)
    return df

