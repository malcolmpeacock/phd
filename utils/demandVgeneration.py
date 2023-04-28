# python script to compare KFs renewable generation with Ninja and the demand
# looks at area under the net demand curve

# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np

# custom code
import stats
import readers

# plot net demand
def plot_net_demand(wind, pv, fraction, demand):
    supply = fraction * wind + (1-fraction) * pv
    net = demand - supply
    zero = net * 0.0
    net.plot(color='blue')
    zero.plot(color='red')
    plt.title('Net demand for wind fraction {}'.format(fraction))
    plt.xlabel('Day of the year', fontsize=15)
    plt.ylabel('Normalised net demand', fontsize=15)
#   plt.legend(loc='upper center')
    plt.show()

# normalize a series
def normalize(s):
    max_value = s.max()
    min_value = s.min()
    n = (s - min_value) / (max_value - min_value)
    return n

# area under curve
def curve_area(s):
    area = np.trapz(s.clip(0.0).values)
    return area

# variance
def variance(s):
    var = s.var()
    return var

# standard deviation
def sd(s):
    sd = s.std()
    return sd

# Pearsons correlation coefficient
def correlation(s1, s2):
    corr = s1.corr(s2)
    return corr


def normalize(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def read_demand(filename, ninja_start, ninja_end):
    demand_dir = '/home/malcolm/uclan/output/timeseries_kf/'
    demand = pd.read_csv(demand_dir+filename, header=0, squeeze=True)
    demand.index = pd.DatetimeIndex(demand['time'])
    demand = demand[ninja_start : ninja_end]
    return demand['demand']

# process command line
parser = argparse.ArgumentParser(description='Compare demand profile with generation')
parser.add_argument('--stats', action="store_true", dest="stats", help='Print out correlation between series' , default=False)
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--ngrid', action="store_true", dest="ngrid", help='Include national grid wind', default=False)
parser.add_argument('--cardenas', action="store_true", dest="cardenas", help='Uses 9 years of Cardenas et. al.', default=False)
parser.add_argument('--monthly', action="store_true", dest="monthly", help='Do monthly correlations', default=False)
parser.add_argument('--step', action="store", dest="step", help='Step size for wind PV combinations', default=0, type=float)
parser.add_argument('--base', action="store", dest="base", help='Proportion of base load', default=0, type=float)
args = parser.parse_args()


# load ninja generation data and normalise
ninja_start = '1984-01-01 00:00:00'
ninja_end = '2013-12-31 23:00:00'
if args.ngrid:
    ninja_start = '2011-01-01 00:00:00'
#   ninja_end = '2013-12-31 23:00:00'
    ninja_end = '2019-12-31 23:00:00'
if args.cardenas:
    ninja_start = '2011-01-01 00:00:00'
    ninja_end = '2019-12-31 23:00:00'
print(ninja_start, ninja_end)
# Ninja capacity factors for pv
ninja_filename_pv = '/home/malcolm/uclan/data/ninja/ninja_pv_country_GB_merra-2_corrected.csv'
# Ninja capacity factors for wind
ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_near-termfuture-merra-2_corrected.csv'
# ninja future
ninja_filename_future = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_long-termfuture-merra-2_corrected.csv'
# ninja current
ninja_filename_current = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_current-merra-2_corrected.csv'

print('Loading ninja ...')
pv_hourly = readers.read_ninja_country(ninja_filename_pv)
wind_hourly = readers.read_ninja_country(ninja_filename_wind)
wind_future = readers.read_ninja_country(ninja_filename_future)
wind_current = readers.read_ninja_country(ninja_filename_current)

print('Extracting PV ...')
ninja_pv = pv_hourly[ninja_start : ninja_end]
ninja_pv = ninja_pv['national']
ninja_pv_daily = ninja_pv.resample('D').mean()

norm_ninja_pv = normalize(ninja_pv_daily)

print('Extracting Wind ...')
ninja_wind = wind_hourly[ninja_start : ninja_end]
ninja_onshore = ninja_wind['onshore']
ninja_offshore = ninja_wind['offshore']
ninja_onshore = ninja_wind['onshore']
ninja_both = ninja_wind['national']

ninja_future = wind_future[ninja_start : ninja_end]
ninja_future_both = ninja_future['national']

ninja_current = wind_current[ninja_start : ninja_end]
ninja_current_both = ninja_current['national']
ninja_current_offshore = ninja_current['offshore']
ninja_current_onshore = ninja_current['onshore']

# load national grid wind
if args.ngrid:
    ngrid_wind_filename = '/home/malcolm/uclan/data/electricity/wind_national_grid.csv'
    wind_ngrid_hourly = readers.read_ngrid_wind(ngrid_wind_filename)
    wind_ngrid_hourly = wind_ngrid_hourly[ninja_start : ninja_end]
    wind_ngrid_daily = wind_ngrid_hourly.resample('D').mean()
    norm_ngrid_wind = normalize(wind_ngrid_daily)
# monthly
    if args.monthly:
        norm_ngrid_wind = norm_ngrid_wind.resample('M').mean()

print('#### Capacity Factors')
print('Series                 mean    min     max    length')
print('Ninja current offshore {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_current_offshore.mean(), ninja_current_offshore.min(), ninja_current_offshore.max(), len(ninja_current_offshore) ) )
print('Ninja current onshore  {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_current_onshore.mean(), ninja_current_onshore.min(), ninja_current_onshore.max(), len(ninja_current_onshore) ) )
print('Ninja current combined {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_current_both.mean(), ninja_current_both.min(), ninja_current_both.max(), len(ninja_current_both) ) )
print('Ninja near offshore    {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_offshore.mean(), ninja_offshore.min(), ninja_offshore.max(), len(ninja_offshore) ) )
print('Ninja near onshore     {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_onshore.mean(), ninja_onshore.min(), ninja_onshore.max(), len(ninja_onshore) ) )
print('Ninja near combined    {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_both.mean(), ninja_both.min(), ninja_both.max(), len(ninja_both) ) )
print('Ninja long combined    {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_future_both.mean(), ninja_future_both.min(), ninja_future_both.max(), len(ninja_future_both) ) )
print('Ninja near PV          {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_pv.mean(), ninja_pv.min(), ninja_pv.max(), len(ninja_pv) ) )
if args.ngrid:
    print('National grid wind     {:.4f}  {:.4f}  {:.4f}  {}'.format(wind_ngrid_hourly.mean(), wind_ngrid_hourly.min(), wind_ngrid_hourly.max(), len(wind_ngrid_hourly) ) )
print(' ---------------- ')

ninja_offshore_daily = ninja_offshore.resample('D').mean()
ninja_onshore_daily = ninja_onshore.resample('D').mean()
ninja_both_daily = ninja_both.resample('D').mean()
ninja_future_daily = ninja_future_both.resample('D').mean()
ninja_current_daily = ninja_current_both.resample('D').mean()

norm_ninja_onshore = normalize(ninja_onshore_daily)
norm_ninja_offshore = normalize(ninja_offshore_daily)
norm_ninja_both = normalize(ninja_both_daily)
norm_ninja_future = normalize(ninja_future_daily)
norm_ninja_current = normalize(ninja_current_daily)

print('Ninja PV daily min {} Wind daily min {}'.format(ninja_pv_daily.min(), ninja_both_daily.min()) )

# Load kf generation data and normalise

print('Loading kf generation ...')

wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
kf_wind = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
kf_pv = pd.read_csv(pv_filename, header=None, squeeze=True)

kf_start = '1984-01-01'
kf_end = '2013-12-31'
kf_index = pd.date_range(start = kf_start, end = kf_end, freq='D', tz='UTC' )

kf_wind.index = kf_index
kf_pv.index = kf_index
kf_wind = kf_wind[ninja_start : ninja_end]
kf_pv = kf_pv[ninja_start : ninja_end]

norm_kf_wind = normalize(kf_wind)
norm_kf_pv = normalize(kf_pv)

# load baseline demand without heat, 41% heat pumps, all heat pumps
# and extract only the daily 1984 to 2013

baseline = read_demand('baseline_daily_all.csv', ninja_start, ninja_end)
hp_41 = read_demand('heatpumps_41_all_daily_all.csv', ninja_start, ninja_end)
hp_all = read_demand('heatpumps_all_daily_all.csv', ninja_start, ninja_end)
existing = read_demand('existing_all_daily.csv', ninja_start, ninja_end)

print('Demand    min(days)  max(days)');
print('Existing  {:.2f}         {:.2f}'.format(existing.min()/existing.mean(), existing.max()/existing.mean()));
print('HP 41     {:.2f}         {:.2f}'.format(hp_41.min()/existing.mean(), hp_41.max()/existing.mean()));
print('HP all    {:.2f}         {:.2f}'.format(hp_all.min()/existing.mean(), hp_all.max()/existing.mean()));


norm_baseline = normalize(baseline)
norm_hp41 = normalize(hp_41)
norm_hp_all = normalize(hp_all)
norm_existing = normalize(existing)


if args.monthly:
    norm_ninja_pv = norm_ninja_pv.resample('M').mean()
    norm_ninja_both = norm_ninja_both.resample('M').mean()
    norm_ninja_onshore = norm_ninja_onshore.resample('M').mean()
    norm_ninja_offshore = norm_ninja_offshore.resample('M').mean()
    norm_ninja_future = norm_ninja_future.resample('M').mean()
    norm_ninja_current = norm_ninja_current.resample('M').mean()
    norm_kf_pv = norm_kf_pv.resample('M').mean()
    norm_kf_wind = norm_kf_wind.resample('M').mean()
    norm_existing = norm_existing.resample('M').mean()
    norm_baseline = norm_baseline.resample('M').mean()
    norm_hp41 = norm_hp41.resample('M').mean()
    norm_hp_all = norm_hp_all.resample('M').mean()
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
    if args.ngrid:
        norm_ngrid_wind.plot(label='Wind Generation from National grid')

    plt.title('Comparison of normalised demand and generation')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

if args.stats:
    print('Comparison to baseline')
    stats.print_stats_header()
    if not args.cardenas:
        stats.print_stats(norm_kf_pv, norm_baseline,     'KF PV   ')
        stats.print_stats(norm_kf_wind, norm_baseline,   'KF Wind ')
    stats.print_stats(norm_ninja_offshore, norm_baseline,   'Ninja offshore Wind ')
    stats.print_stats(norm_ninja_onshore, norm_baseline,   'Ninja onshore Wind ')
    stats.print_stats(norm_ninja_both, norm_baseline,   'Ninja combined Wind ')
    stats.print_stats(norm_ninja_future, norm_baseline,   'Ninja future Wind ')
    stats.print_stats(norm_ninja_current, norm_baseline,   'Ninja current Wind ')
    stats.print_stats(norm_ninja_pv, norm_baseline,   'Ninja PV ')
    if args.ngrid:
        stats.print_stats(norm_ngrid_wind, norm_baseline,   'Nat grid wind ')

    print('Comparison to existing')
    stats.print_stats_header()
    if not args.cardenas:
        stats.print_stats(norm_kf_pv, norm_existing,     'KF PV   ')
        stats.print_stats(norm_kf_wind, norm_existing,   'KF Wind ')
    stats.print_stats(norm_ninja_offshore, norm_existing,   'Ninja offshore Wind ')
    stats.print_stats(norm_ninja_onshore, norm_existing,   'Ninja onshore Wind ')
    stats.print_stats(norm_ninja_both, norm_existing,   'Ninja combined Wind ')
    stats.print_stats(norm_ninja_future, norm_existing,   'Ninja future Wind ')
    stats.print_stats(norm_ninja_current, norm_existing,   'Ninja current Wind ')
    stats.print_stats(norm_ninja_pv, norm_existing,   'Ninja PV ')
    if args.ngrid:
        stats.print_stats(norm_ngrid_wind, norm_existing,   'Nat grid wind ')

    print('Comparison to 41% heat pumps')
    stats.print_stats_header()
    if not args.cardenas:
        stats.print_stats(norm_kf_pv, norm_hp41,     'KF PV   ')
        stats.print_stats(norm_kf_wind, norm_hp41,   'KF Wind ')
    stats.print_stats(norm_ninja_offshore, norm_hp41,   'Ninja offshore Wind ')
    stats.print_stats(norm_ninja_onshore, norm_hp41,   'Ninja onshore Wind ')
    stats.print_stats(norm_ninja_both, norm_hp41,   'Ninja combined Wind ')
    stats.print_stats(norm_ninja_future, norm_hp41,   'Ninja future Wind ')
    stats.print_stats(norm_ninja_current, norm_hp41,   'Ninja current Wind ')
    stats.print_stats(norm_ninja_pv, norm_hp41,   'Ninja PV ')
    if args.ngrid:
        stats.print_stats(norm_ngrid_wind, norm_hp41,   'Nat grid wind ')

    print('Comparison to all heat pumps')
    stats.print_stats_header()
    if not args.cardenas:
        stats.print_stats(norm_kf_pv, norm_hp_all,     'KF PV   ')
        stats.print_stats(norm_kf_wind, norm_hp_all,   'KF Wind ')
    stats.print_stats(norm_ninja_offshore, norm_hp_all,   'Ninja offshore Wind ')
    stats.print_stats(norm_ninja_onshore, norm_hp_all,   'Ninja onshore Wind ')
    stats.print_stats(norm_ninja_both, norm_hp_all,   'Ninja combined Wind ')
    stats.print_stats(norm_ninja_future, norm_hp_all,   'Ninja future Wind ')
    stats.print_stats(norm_ninja_current, norm_hp_all,   'Ninja current Wind ')
    stats.print_stats(norm_ninja_pv, norm_hp_all,   'Ninja PV ')
    if args.ngrid:
        stats.print_stats(norm_ngrid_wind, norm_hp_all,   'Nat grid wind ')

if args.step>0:

    ra = []
    re = []
    fs = []
    area_e = []
    area_f = []
    variance_e = []
    variance_f = []
    sd_e = []
    sd_f = []
    
    for fraction in np.arange(0.0,1.0+args.step, args.step):
        fs.append(fraction)
        supply = fraction * norm_ninja_both + (1-fraction) * norm_ninja_pv
        if args.base>0:
            supply = supply + args.base
            supply = normalize(supply)
        re.append(correlation(supply, norm_existing))
        ra.append(correlation(supply, norm_hp_all))
        area_e.append(curve_area(norm_existing - supply))
        area_f.append(curve_area(norm_hp_all - supply))
        variance_e.append(variance(norm_existing - supply))
        variance_f.append(variance(norm_hp_all - supply))
        sd_e.append(sd(norm_existing - supply))
        sd_f.append(sd(norm_hp_all - supply))
    data = { 'fraction' : fs, 're' : re, 'ra' : ra, 'area_e' : area_e, 'area_f' : area_f, 'variance_e': variance_e, 'variance_f': variance_f, 'sd_e' : sd_e, 'sd_f' : sd_f }
    df = pd.DataFrame(data=data)
    print(df)
    if args.plot:
        # Correlation
        plt.plot(df['fraction'], df['re'], label='existing heating')
        plt.plot(df['fraction'], df['ra'], label='all heat pumps')
        freq = 'Daily '
        if args.monthly:
            freq = 'Monthly '
        plt.title('{}correlation of wind fraction in the supply to the demand'.format(freq))
        plt.xlabel('Wind Capacity Fraction', fontsize=15)
        plt.ylabel('Pearsons correlation coefficient (R)', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
        df['area_e'] = normalize(df['area_e'])
        df['area_f'] = normalize(df['area_f'])
        # Area under the net demand curve
        plt.plot(df['fraction'], df['area_e'], label='existing heating')
        plt.plot(df['fraction'], df['area_f'], label='all heat pumps')
        freq = 'Daily '
        plt.title('Wind fraction vs area under the net demand curve')
        plt.xlabel('Wind Capacity Fraction', fontsize=15)
        plt.ylabel('Area under net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
        df['variance_e'] = normalize(df['variance_e'])
        df['variance_f'] = normalize(df['variance_f'])
        # Area under the net demand curve
        plt.plot(df['fraction'], df['variance_e'], label='existing heating')
        plt.plot(df['fraction'], df['variance_f'], label='all heat pumps')
        freq = 'Daily '
        plt.title('Wind fraction vs variance of the net demand curve')
        plt.xlabel('Wind Capacity Fraction', fontsize=15)
        plt.ylabel('variance of net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
        df['sd_e'] = normalize(df['sd_e'])
        df['sd_f'] = normalize(df['sd_f'])
        # Area under the net demand curve
        plt.plot(df['fraction'], df['sd_e'], label='existing heating')
        plt.plot(df['fraction'], df['sd_f'], label='all heat pumps')
        freq = 'Daily '
        plt.title('Wind fraction vs standard deviation of the net demand curve')
        plt.xlabel('Wind Capacity Fraction', fontsize=15)
        plt.ylabel('standard deviation of net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        plot_net_demand(norm_ninja_both, norm_ninja_pv, 0.80, norm_existing)

        print(df)

        min_e = df[df['area_e'] == df['area_e'].min() ]
        min_f = df[df['area_f'] == df['area_f'].min() ]
        print('Area under the net demand curve: ')
        print(min_e)
        print(min_f)
        min_e = df[df['variance_e'] == df['variance_e'].min() ]
        min_f = df[df['variance_f'] == df['variance_f'].min() ]
        print('Varince of the net demand curve: ')
        print(min_e)
        print(min_f)
        min_e = df[df['sd_e'] == df['sd_e'].min() ]
        min_f = df[df['sd_f'] == df['sd_f'].min() ]
        print('Standard deviation of the net demand curve: ')
        print(min_e)
        print(min_f)
