# python script to convert hourly demand time series to daily.

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm

# custom code
import stats
import readers

# main program
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018HDD15.5.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018Ruhnau.csv'
# demand_filename = '/home/malcolm/uclan/tools/python/output/2018/heatCopRef2018weather2018Watson.csv'
demand_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018Sflat.csv'
daily_filename = '/home/malcolm/uclan/tools/python/output/2018/Ref2018Weather2018SflatDaily.csv'

demand = readers.read_copheat(demand_filename,['space','water','heat'])
print('DEMAND')
print(demand.index)
print(demand)
daily = demand.resample('D').sum()
print('DAILY')
print(daily.index)
print(daily)
# Timestamp
index = pd.DatetimeIndex(daily.index)
daily.index = index.strftime('%Y-%m-%dT%H:%M:%SZ')
daily.to_csv(daily_filename, sep=',', decimal='.', float_format='%g')

