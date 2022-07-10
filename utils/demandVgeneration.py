# python script to compare KFs renewable generation with Ninja

# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# custom code
import stats
import readers

def normalize(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def read_demand(filename):
    ninja_start = '1984-01-01 00:00:00'
    ninja_end = '2013-12-31 23:00:00'
    demand_dir = '/home/malcolm/uclan/output/timeseries_kf/'
    demand = pd.read_csv(demand_dir+filename, header=0, squeeze=True)
    demand.index = pd.DatetimeIndex(demand['time'])
    print(demand)
    demand = demand[ninja_start : ninja_end]
    return demand['demand']

# process command line
parser = argparse.ArgumentParser(description='Compare demand profile with generation')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()


# load ninja genration data and normalise
ninja_start = '1984-01-01 00:00:00'
ninja_end = '2013-12-31 23:00:00'
print(ninja_start, ninja_end)
# Ninja capacity factors for pv
ninja_filename_pv = '/home/malcolm/uclan/data/ninja/ninja_pv_country_GB_merra-2_corrected.csv'
# Ninja capacity factors for wind
ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_near-termfuture-merra-2_corrected.csv'

print('Loading ninja ...')
pv_hourly = readers.read_ninja_country(ninja_filename_pv)
wind_hourly = readers.read_ninja_country(ninja_filename_wind)

print('Extracting PV ...')
ninja_pv = pv_hourly[ninja_start : ninja_end]
ninja_pv = ninja_pv['national']
ninja_pv_daily = ninja_pv.resample('D').mean()

norm_ninja_pv = normalize(ninja_pv_daily)

print('Extracting Wind ...')
ninja_wind = wind_hourly[ninja_start : ninja_end]
ninja_onshore = ninja_wind['onshore']
#ninja_onshore = ninja_onshore.resample('D').mean()
ninja_offshore = ninja_wind['offshore']
ninja_onshore = ninja_wind['onshore']
ninja_both = ninja_wind['national']
ninja_offshore_daily = ninja_offshore.resample('D').mean()
ninja_onshore_daily = ninja_onshore.resample('D').mean()
ninja_both_daily = ninja_both.resample('D').mean()

norm_ninja_onshore = normalize(ninja_onshore_daily)
norm_ninja_offshore = normalize(ninja_offshore_daily)
norm_ninja_both = normalize(ninja_both_daily)

# Load kf generation data and normalise

print('Loading kf generation ...')

wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
kf_wind = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
kf_pv = pd.read_csv(pv_filename, header=None, squeeze=True)

kf_wind.index = ninja_onshore_daily.index
kf_pv.index = ninja_pv_daily.index

norm_kf_wind = normalize(kf_wind)
norm_kf_pv = normalize(kf_pv)

# load baseline demand without heat, 41% heat pumps, all heat pumps
# and extract only the daily 1984 to 2013

baseline = read_demand('baseline_daily_all.csv')
hp_41 = read_demand('heatpumps_41_all_daily_all.csv')
hp_all = read_demand('heatpumps_41_all_daily_all.csv')
existing = read_demand('existing_all_daily.csv')

norm_baseline = normalize(baseline)
norm_hp41 = normalize(hp_41)
norm_hp_all = normalize(hp_all)
norm_existing = normalize(existing)

# plot 

if args.plot:
    norm_ninja_pv.plot(label='PV Generation from Renewables Ninja')
    norm_ninja_both.plot(label='Wind Generation from Renewables Ninja all')
    norm_ninja_onshore.plot(label='Wind Generation from Renewables Ninja onshore')
    norm_ninja_offshore.plot(label='Wind Generation from Renewables Ninja offshore')
    norm_kf_pv.plot(label='PV Generation from KF')
    norm_kf_wind.plot(label='Wind Generation from KF')
    norm_baseline.plot(label='Baseline demand')
    norm_hp41.plot(label='Demand with 41% hp')
    norm_hp_all.plot(label='dEmand with all hp')

    plt.title('Comparison of normalised demand and generation')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

print('Comparison to baseline')
stats.print_stats_header()
stats.print_stats(norm_kf_pv, norm_baseline,     'KF PV   ')
stats.print_stats(norm_kf_wind, norm_baseline,   'KF Wind ')
stats.print_stats(norm_ninja_offshore, norm_baseline,   'Ninja offshore Wind ')
stats.print_stats(norm_ninja_onshore, norm_baseline,   'Ninja onshore Wind ')
stats.print_stats(norm_ninja_both, norm_baseline,   'Ninja combined Wind ')
stats.print_stats(norm_ninja_pv, norm_baseline,   'Ninja PV ')

print('Comparison to existing')
stats.print_stats_header()
stats.print_stats(norm_kf_pv, norm_existing,     'KF PV   ')
stats.print_stats(norm_kf_wind, norm_existing,   'KF Wind ')
stats.print_stats(norm_ninja_offshore, norm_existing,   'Ninja offshore Wind ')
stats.print_stats(norm_ninja_onshore, norm_existing,   'Ninja onshore Wind ')
stats.print_stats(norm_ninja_both, norm_existing,   'Ninja combined Wind ')
stats.print_stats(norm_ninja_pv, norm_existing,   'Ninja PV ')

print('Comparison to 41% heat pumps')
stats.print_stats_header()
stats.print_stats(norm_kf_pv, norm_hp41,     'KF PV   ')
stats.print_stats(norm_kf_wind, norm_hp41,   'KF Wind ')
stats.print_stats(norm_ninja_offshore, norm_hp41,   'Ninja offshore Wind ')
stats.print_stats(norm_ninja_onshore, norm_hp41,   'Ninja onshore Wind ')
stats.print_stats(norm_ninja_both, norm_hp41,   'Ninja combined Wind ')
stats.print_stats(norm_ninja_pv, norm_hp41,   'Ninja PV ')
print('Comparison to all heat pumps')
stats.print_stats_header()
stats.print_stats(norm_kf_pv, norm_hp_all,     'KF PV   ')
stats.print_stats(norm_kf_wind, norm_hp_all,   'KF Wind ')
stats.print_stats(norm_ninja_offshore, norm_hp_all,   'Ninja offshore Wind ')
stats.print_stats(norm_ninja_onshore, norm_hp_all,   'Ninja onshore Wind ')
stats.print_stats(norm_ninja_both, norm_hp_all,   'Ninja combined Wind ')
stats.print_stats(norm_ninja_pv, norm_hp_all,   'Ninja PV ')
