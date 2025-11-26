# Program to simulate energy storage required for different proportions of wind
# and solar generation and base load.

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

#
#  demand time series - reference year electricity with heat removed
#               wind  - wind capacity factor time series
#                 pv  - pv capacity factor time series
def supply_and_storage(demand, wind, pv, baseload, variable):
    hourly = True
    h_input = None
    nwind=args.nwind    # number of points (60)
    npv=args.npv    # number of points (60)
    step=args.step    # step size (0.1)

    df, sample_hist, sample_durations, sample_net = storage.storage_grid(demand, wind, pv, eta, etad, hourly, npv, nwind, step, baseload, variable, h_input, 'all', args.wind, args.pv, args.threshold, 'new', False, args.store_max)

    df['base'] = df['storage'] * 0.0 + baseload
    df['variable'] = df['storage'] * 0.0 + variable

    return df, sample_hist, sample_durations, sample_net

# main program

# process command line
parser = argparse.ArgumentParser(description='Calculate storage needed for given electricity demand time series')
parser.add_argument('--dir', action="store", dest="dir", help='Output directory', default='new/results' )
parser.add_argument('--demand', action="store", dest="demand", help='Demand file name', default='3years_hp41' )
parser.add_argument('--normalise', action="store_true", dest="normalise", help='Normalise the time series', default=False)
parser.add_argument('--plot', action="store_true", dest="plot", help='Produce plots', default=False)
parser.add_argument('--debug', action="store_true", dest="debug", help='Debug outut', default=False)
parser.add_argument('--end_full', action="store_true", dest="end_full", help='Do not insist on the store being fuller at the end', default=False)
parser.add_argument('--eta', action="store", dest="eta", help='Round Trip Efficiency.', type=int, default=85)
parser.add_argument('--etad', action="store", dest="etad", help='Discharge Efficiency. If this is specified non zero, then --eta is the charge efficiency', type=int, default=0)
parser.add_argument('--npv', action="store", dest="npv", help='Number of points in pv grid.', type=int, default=60)
parser.add_argument('--nwind', action="store", dest="nwind", help='Number of points in wind grid.', type=int, default=60)
parser.add_argument('--baseload', action="store", dest="baseload", help='Base load capacity.', type=float, default=0.0)
parser.add_argument('--step', action="store", dest="step", help='Step size.', type=float, default=0.1)
parser.add_argument('--shore', action="store", dest="shore", default="all", help='on=Use only onshore wind off=only offshore, all=all' )
parser.add_argument('--ninja', action="store", dest="ninja", default="near", help='Which ninja to use: near, current, future', choices=['near', 'current', 'future'] )
parser.add_argument('--wind', action="store", dest="wind", help='Wind value of store history to output', type=float, default=0)
parser.add_argument('--pv', action="store", dest="pv", help='Pv value of store history to output', type=float, default=0)
parser.add_argument('--days', action="store", dest="days", help='Example store size to find for store hist plotting', type=float, default=0)
parser.add_argument('--threshold', action="store", dest="threshold", help='Threshold for considering 2 wind values the same in new storage model', type=float, default=0.01)
parser.add_argument('--variable', action="store", dest="variable", help='Amount of variable generation, default-0.0', type=float, default=0.0)
parser.add_argument('--store_max', action="store", dest="store_max", help='Maximum value of storage in days, default=80.0', type=float, default=80.0)
parser.add_argument('--store_size', action="store", dest="store_size", help='One value of storage to test if its enough', type=float, default=0.0)
parser.add_argument('--store_start', action="store", dest="store_start", help='Starting size of the store', type=float, default=0.7)

args = parser.parse_args()

output_dir = "/home/malcolm/uclan/output/" + args.dir
if not os.path.isdir(output_dir):
    print('Error output dir {} does not exist'.format(output_dir))
    quit()

# Read in demand
demand_filename = '/home/malcolm/uclan/output/new/demand/' + args.demand + '.csv'
demand = readers.read_demand(demand_filename, parm='demand_twh')

# deduce list of years from it
start_year = demand.index.year[0]
end_year = demand.index.year[len(demand)-1]
years = range(start_year, end_year+1)
    
# print arguments
print('Start year {} End Year {}'.format(start_year, end_year, args.demand) )

# calculate charge and discharge efficiency from round trip efficiency
if args.etad > 0:
    etad = args.etad / 100
    eta = args.eta / 100
else:
    eta = math.sqrt(args.eta / 100)
    etad = eta
print('Efficiency Charge {} Discharge {} '.format(eta, etad) )

if args.normalise:
    normalise_factor = 0.8183877082 / 24.0
    daily_demand = demand.resample('D').sum()
    demand = demand / normalise_factor
    print('PEAK DEMAND {} Annual Demand {} Mean Daily Demand {} Normalise Factor {}'.format(daily_demand.max(), daily_demand.sum()/len(years), daily_demand.mean(), normalise_factor))
else:
    normalise_factor = 1.0


# Read in pv and wind

ninja_start = str(years[0]) + '-01-01 00:00:00'
ninja_end = str(years[-1]) + '-12-31 23:00:00'
print('Ninja start {} end {}'.format(ninja_start, ninja_end))

# Ninja capacity factors for pv
ninja_filename_pv = '/home/malcolm/uclan/data/ninja/ninja_pv_country_GB_merra-2_corrected.csv'
print('Loading ninja ...')
ninja_pv = readers.read_ninja_country(ninja_filename_pv)
print('Extracting PV ...')
ninja_pv = ninja_pv[ninja_start : ninja_end]
pv = ninja_pv['national']

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

if args.plot:
    # daily plot
    wind_daily = wind.resample('D').mean()
    pv_daily = pv.resample('D').mean()
    wind_daily.plot(color='blue', label='ninja wind generation')
    pv_daily.plot(color='red', label='ninja pv generation')
    plt.title('Wind and solar generation')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel('Electricity generation capacity factor per day', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

print('Quantity           mean     total    length')
print('Normalised demand  {:.2f}   {:.2f} {}'.format(demand.mean(), demand.sum(), len(demand) ))
print('pv                 {:.2f}   {:.2f} {}'.format(pv.mean(), pv.sum(), len(pv)))
print('wind               {:.2f}   {:.2f} {}'.format(wind.mean(), wind.sum(), len(wind)))

# Finding the storage
if args.store_size == 0.0:
    df, sample_hist, sample_durations, sample_net = supply_and_storage(demand, wind, pv, args.baseload, args.variable)
    print("Max storage {} Min Storage {}".format(df['storage'].max(), df['storage'].min()) )

    # store actual capacity in GW
    df['gw_wind'] = df['f_wind'] * normalise_factor / ( 24 * 1000.0 )
    df['gw_pv'] = df['f_pv'] * normalise_factor / ( 24 * 1000.0 )

    sfile='{}/shares{}.csv'.format(output_dir, args.demand)
    print(sfile)
    df.to_csv('{}/shares{}.csv'.format(output_dir, args.demand))
    sample_hist.to_csv('{}/store{}.csv'.format(output_dir, args.demand))
    sample_durations.to_csv('{}/duration{}.csv'.format(output_dir, args.demand))
    sample_net.to_csv('{}/net{}.csv'.format(output_dir, args.demand))
    # for comparing with old runs
    demand.to_csv('{}/demand{}.csv'.format(output_dir, args.demand))
    hydrogen = demand * 0.0
    hydrogen.to_csv('{}/hydrogen{}.csv'.format(output_dir, args.demand))

    dfs_max = df['storage'].max()
    dfs_min = df['storage'].min()


    # Testing the storage
else:
    # converts to hourly.
    if args.normalise:
        store_factor = 1 / 24
        store_size = args.store_size / store_factor
    else:
        store_size = args.store_size

    balanced, sample_hist, variable_total = storage.storage_balance(demand, wind, pv, eta, etad, args.baseload, None, args.pv, args.wind, store_size, args.store_start, args.variable, args.debug, not args.end_full)
    if balanced:
        print('Store size was enough')
    else:
        print('Store size was not enough')
    if args.normalise:
        sample_hist = ( sample_hist / 24 ) * 0.81838
    sample_hist.to_csv('{}/store{}.csv'.format(output_dir, args.demand))
    demand.to_csv('{}/demand{}.csv'.format(output_dir, args.demand))
    dfs_max = args.store_size
    dfs_min = args.store_size

end_time = math.floor(datetime.timestamp(datetime.now()) - datetime.timestamp(start_time)) / 360
# output settings
settings = {
    'start'     : start_year,
    'end'       : end_year,
    'reference' : 2018,
    'ev'        : False,
    'storage'   : 'all',
    'variable'  : args.variable,
    'baseload'  : args.baseload,
    'hist_pv'   : args.pv,
    'hist_wind' : args.wind,
    'eta'       : args.eta,
    'etad'      : args.etad,
    'cfpv'      : 1.0,
    'cfwind'    : 1.0,
    'demand'    : args.demand,
    'dmethod'   : 'baseline',
    'hourly'    : True,
    'kfpv'      : False,
    'kfwind'    : False,
    'shore'     : args.shore,
    'threshold' : args.threshold,
    'normalise' : normalise_factor * 1e6,
    'max_storage' : dfs_max,
    'min_storage' : dfs_min,
    'run_time'  : end_time
}
settings_df = pd.DataFrame.from_dict(data=settings, orient='index')
settings_df.to_csv('{}/settings{}.csv'.format(output_dir, args.demand), header=False)

print('Finished in {:.2f} hours '.format(end_time))
