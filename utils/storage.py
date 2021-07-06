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

def storage(net_demand, eta=0.75):
    history = net_demand.copy()
    store = 0.0
    min_store = 0.0
    eta_charge = eta
    eta_discharge = eta
    count=0
    for index, value in net_demand.items():
        history.iat[count] = store
        count += 1
        # Note: both subtract because value is negative in the 2nd one!
        if value > 0.0:
            store = store - value * eta_discharge
        else:
            store = store - value * eta_charge
        if store < min_store:
            min_store = store
#   print('Storage {:.2f} GWh'.format(min_store))
    return min_store, history

# constant storage line

def storage_line(df,storage_value):
    x=[]
    y=[]
    # for each wind value ...
    for i_wind in range(0,14):
        f_wind = i_wind * 0.5
        # extract those values with a wind=xs
        is_xs = df['f_wind'] == f_wind
        df_xs = df[is_xs]
        # check storage in range
        if storage_value < df_xs['storage'].max() and storage_value > df_xs['storage'].min():
            # sort them by storage
            df_xs = df_xs.sort_values('storage',ascending=False)
            # interpolate a pv value for the storage
            y_interp = scipy.interpolate.interp1d(df_xs['storage'], df_xs['f_pv'])
            f_pv = y_interp(storage_value)
            # store the points
            x.append(f_wind)
            y.append(f_pv.item())

    sline = { 'Pw' : x, 'Ps' :y }
    df = pd.DataFrame(data=sline)
    print('Line: Pw max {} min {} '.format(df['Pw'].max(), df['Pw'].min() ) )
#   print(df)
    return df

def storage_grid(demand, wind, pv, eta, hourly=False):
    print('storage_grid: demand max {} min {} mean {}'.format(demand.max(), demand.min(), demand.mean()) )
    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [] }
    # For hourly the storage will be in hours, so divide by 24 to convert to 
    # days
    if hourly:
        store_factor = 1 / 24
    else:
        store_factor = 1

    # For each percent of PV/Wind
    for i_pv in range(0,14):
        for i_wind in range(0,14):
            f_pv = i_pv * 0.5
            f_wind = i_wind * 0.5
#           print('Calculating f_pv {} f_wind {} '.format(f_pv, f_wind) )
            # energy supply is calculated using the capacity factors
            supply = (wind * f_wind) + (pv * f_pv)
            net = demand - supply

            #  calculate how much storage we need

            store_size, store_hist = storage(net, eta)
            results['f_pv'].append(f_pv)
            results['f_wind'].append(f_wind)
            results['storage'].append(store_size * store_factor)

            # yearly values
#           store_max_yearly = store_hist.resample('Y', axis=0).max()
#           store_min_yearly = store_hist.resample('Y', axis=0).min()
#           store_diff_yearly = store_max_yearly - store_max_yearly

    df = pd.DataFrame(data=results)
    return df

