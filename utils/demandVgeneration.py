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

def stat_print(s1, s2, method):
    stats.print_stats(s1, s2, method, 1, False, ' ', ' ', False)

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

# charge rate
def charge_rate(s):
    charge_rate=0.0
    rate = s.diff()
    plus_rates = rate[rate>0]
    if len(plus_rates)>0:
        charge_rate = plus_rates.max()
    return charge_rate

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
    demand_dir = '/home/malcolm/uclan/output/timeseries2/40yearsHourly/'
    demand = pd.read_csv(demand_dir+filename, header=0).squeeze()
    demand.index = pd.DatetimeIndex(demand['time'])
    demand = demand[ninja_start : ninja_end]
#   return demand['demand']
#   return demand
    return demand['0']

# process command line
parser = argparse.ArgumentParser(description='Compare demand profile with generation')
parser.add_argument('--stats', action="store_true", dest="stats", help='Print out correlation between series' , default=False)
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--energy', action="store_true", dest="energy", help='Do energy frfaction, not capacity', default=False)
parser.add_argument('--ngrid', action="store_true", dest="ngrid", help='Include national grid wind', default=False)
parser.add_argument('--cardenas', action="store_true", dest="cardenas", help='Uses 9 years of Cardenas et. al.', default=False)
parser.add_argument('--frequency', action="store", dest="frequency", help='Frequency M=monthly, D=Daily, H=Hourly of correlations', default='D')
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
if args.frequency == 'D':
    ninja_pv = ninja_pv.resample('D').mean()
if args.frequency == 'M':
    ninja_pv = ninja_pv.resample('M').mean()

norm_ninja_pv = normalize(ninja_pv)

print('Extracting Wind ...')
ninja_wind = wind_hourly[ninja_start : ninja_end]
ninja_onshore = ninja_wind['onshore']
ninja_offshore = ninja_wind['offshore']
ninja_onshore = ninja_wind['onshore']
ninja_both = ninja_wind['national']

ninja_future = wind_future[ninja_start : ninja_end]
ninja_future = ninja_future['national']

ninja_current = wind_current[ninja_start : ninja_end]
ninja_current_both = ninja_current['national']
ninja_current_offshore = ninja_current['offshore']
ninja_current_onshore = ninja_current['onshore']

# load national grid wind
if args.ngrid:
    ngrid_wind_filename = '/home/malcolm/uclan/data/electricity/wind_national_grid.csv'
    wind_ngrid_hourly = readers.read_ngrid_wind(ngrid_wind_filename)
    wind_ngrid_hourly = wind_ngrid_hourly[ninja_start : ninja_end]
    if args.frequency == 'H':
        wind_ngrid = wind_ngrid_hourly
    if args.frequency == 'D':
        wind_ngrid = wind_ngrid_hourly.resample('D').mean()
    if args.frequency == 'M':
        wind_ngrid = wind_ngrid_hourly.resample('M').mean()
    norm_ngrid_wind = normalize(wind_ngrid)

print('#### Capacity Factors')
print('Series                 mean    min     max    length')
print('Ninja current offshore {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_current_offshore.mean(), ninja_current_offshore.min(), ninja_current_offshore.max(), len(ninja_current_offshore) ) )
print('Ninja current onshore  {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_current_onshore.mean(), ninja_current_onshore.min(), ninja_current_onshore.max(), len(ninja_current_onshore) ) )
print('Ninja current combined {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_current_both.mean(), ninja_current_both.min(), ninja_current_both.max(), len(ninja_current_both) ) )
print('Ninja near offshore    {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_offshore.mean(), ninja_offshore.min(), ninja_offshore.max(), len(ninja_offshore) ) )
print('Ninja near onshore     {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_onshore.mean(), ninja_onshore.min(), ninja_onshore.max(), len(ninja_onshore) ) )
print('Ninja near combined    {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_both.mean(), ninja_both.min(), ninja_both.max(), len(ninja_both) ) )
print('Ninja long combined    {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_future.mean(), ninja_future.min(), ninja_future.max(), len(ninja_future) ) )
print('Ninja near PV          {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_pv.mean(), ninja_pv.min(), ninja_pv.max(), len(ninja_pv) ) )
if args.ngrid:
    print('National grid wind     {:.4f}  {:.4f}  {:.4f}  {}'.format(wind_ngrid_hourly.mean(), wind_ngrid_hourly.min(), wind_ngrid_hourly.max(), len(wind_ngrid_hourly) ) )
print(' ---------------- ')

if args.frequency == 'D':
    ninja_offshore = ninja_offshore.resample('D').mean()
    ninja_onshore = ninja_onshore.resample('D').mean()
    ninja_both = ninja_both.resample('D').mean()
    ninja_future = ninja_future.resample('D').mean()
    ninja_current = ninja_current_both.resample('D').mean()
if args.frequency == 'M':
    ninja_offshore = ninja_offshore.resample('M').mean()
    ninja_onshore = ninja_onshore.resample('M').mean()
    ninja_both = ninja_both.resample('M').mean()
    ninja_future = ninja_future.resample('M').mean()
    ninja_current = ninja_current_both.resample('M').mean()
if args.frequency == 'H':
    ninja_current = ninja_current_both

norm_ninja_onshore = normalize(ninja_onshore)
norm_ninja_offshore = normalize(ninja_offshore)
norm_ninja_both = normalize(ninja_both)
norm_ninja_future = normalize(ninja_future)
norm_ninja_current = normalize(ninja_current)

print('Ninja PV min {} Wind min {}'.format(ninja_pv.min(), ninja_both.min()) )

# Load kf generation data and normalise

print('Loading kf generation ...')

wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
kf_wind = pd.read_csv(wind_filename, header=None).squeeze()
pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
kf_pv = pd.read_csv(pv_filename, header=None).squeeze()

kf_start = '1984-01-01'
kf_end = '2013-12-31'
kf_index = pd.date_range(start = kf_start, end = kf_end, freq='D', tz='UTC' )

kf_wind.index = kf_index
kf_pv.index = kf_index
kf_wind = kf_wind[ninja_start : ninja_end]
kf_pv = kf_pv[ninja_start : ninja_end]

if args.frequency == 'M':
    kf_pv = kf_pv.resample('M').mean()
    kf_wind = kf_wind.resample('M').mean()

norm_kf_wind = normalize(kf_wind)
norm_kf_pv = normalize(kf_pv)

# load baseline demand without heat, 41% heat pumps, all heat pumps
# and extract only the daily 1984 to 2013

baseline = read_demand('demandNNS.csv', ninja_start, ninja_end)
hp_41 = read_demand('demandFNS.csv', ninja_start, ninja_end)
hp_all = read_demand('demandPNS.csv', ninja_start, ninja_end)
existing = read_demand('demandENS.csv', ninja_start, ninja_end)
ev = read_demand('demandEV.csv', ninja_start, ninja_end)

if args.frequency == 'M':
    baseline = baseline.resample('M').mean()
    hp_41 = hp_41.resample('M').mean()
    hp_all = hp_all.resample('M').mean()
    existing = existing.resample('M').mean()
    ev = ev.resample('M').mean()
if args.frequency == 'D':
    baseline = baseline.resample('D').mean()
    hp_41 = hp_41.resample('D').mean()
    hp_all = hp_all.resample('D').mean()
    existing = existing.resample('D').mean()
    ev = ev.resample('D').mean()
#
rate = existing.diff()
plus_rates = rate[rate>0]
e_ramp_up = plus_rates.max()
e_loc = plus_rates.idxmax()

rate = hp_41.diff()
plus_rates = rate[rate>0]
f_ramp_up = plus_rates.max()
f_loc = plus_rates.idxmax()

rate = hp_all.diff()
plus_rates = rate[rate>0]
a_ramp_up = plus_rates.max()
a_loc = plus_rates.idxmax()

rate = ev.diff()
plus_rates = rate[rate>0]
v_ramp_up = plus_rates.max()
v_loc = plus_rates.idxmax()

print('Demand    min     max    max ramp up');
print('          (days)  (days) (days)');
print('Existing  {:.2f}  {:.2f}  {:.2f} {}'.format(existing.min()/existing.mean(), existing.max()/existing.mean(), e_ramp_up, e_loc));
print('HP 41     {:.2f}  {:.2f}  {:.2f} {}'.format(hp_41.min()/existing.mean(), hp_41.max()/existing.mean(), f_ramp_up, f_loc));
print('HP all    {:.2f}  {:.2f}  {:.2f} {}'.format(hp_all.min()/existing.mean(), hp_all.max()/existing.mean(), a_ramp_up, a_loc));
print('EVs       {:.2f}  {:.2f}  {:.2f} {}'.format(ev.min()/existing.mean(), ev.max()/existing.mean(), v_ramp_up, v_loc));

print('EV max ramp up occurs at:')
print(v_loc)


norm_baseline = normalize(baseline)
norm_hp41 = normalize(hp_41)
norm_hp_all = normalize(hp_all)
norm_existing = normalize(existing)
norm_ev = normalize(ev)

# plot 

if args.plot:
    norm_ninja_pv.plot(label='PV Generation from Renewables Ninja')
    norm_ninja_both.plot(label='Wind Generation from Renewables Ninja all')
    norm_ninja_onshore.plot(label='Wind Generation from Renewables Ninja onshore')
    norm_ninja_offshore.plot(label='Wind Generation from Renewables Ninja offshore')
    if args.frequency != 'H':
        norm_kf_pv.plot(label='PV Generation from KF')
        norm_kf_wind.plot(label='Wind Generation from KF')
    norm_baseline.plot(label='Baseline demand')
    norm_hp41.plot(label='Demand with 41% hp')
    norm_hp_all.plot(label='Demand with all hp')
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
    if not args.cardenas and args.frequency != 'H':
        stat_print(norm_kf_pv, norm_baseline,     'KF PV   ')
        stat_print(norm_kf_wind, norm_baseline,   'KF Wind ')
    stat_print(norm_ninja_offshore, norm_baseline,   'Ninja offshore Wind ')
    stat_print(norm_ninja_onshore, norm_baseline,   'Ninja onshore Wind ')
    stat_print(norm_ninja_both, norm_baseline,   'Ninja combined Wind ')
    stat_print(norm_ninja_future, norm_baseline,   'Ninja future Wind ')
    stat_print(norm_ninja_current, norm_baseline,   'Ninja current Wind ')
    stat_print(norm_ninja_pv, norm_baseline,   'Ninja PV ')
    if args.ngrid:
        stat_print(norm_ngrid_wind, norm_baseline,   'Nat grid wind ')

    print('Comparison to existing')
    stats.print_stats_header()
    if not args.cardenas and args.frequency != 'H':
        stat_print(norm_kf_pv, norm_existing,     'KF PV   ')
        stat_print(norm_kf_wind, norm_existing,   'KF Wind ')
    stat_print(norm_ninja_offshore, norm_existing,   'Ninja offshore Wind ')
    stat_print(norm_ninja_onshore, norm_existing,   'Ninja onshore Wind ')
    stat_print(norm_ninja_both, norm_existing,   'Ninja combined Wind ')
    stat_print(norm_ninja_future, norm_existing,   'Ninja future Wind ')
    stat_print(norm_ninja_current, norm_existing,   'Ninja current Wind ')
    stat_print(norm_ninja_pv, norm_existing,   'Ninja PV ')
    if args.ngrid:
        stat_print(norm_ngrid_wind, norm_existing,   'Nat grid wind ')

    print('Comparison to 41% heat pumps')
    stats.print_stats_header()
    if not args.cardenas and args.frequency != 'H':
        stat_print(norm_kf_pv, norm_hp41,     'KF PV   ')
        stat_print(norm_kf_wind, norm_hp41,   'KF Wind ')
    stat_print(norm_ninja_offshore, norm_hp41,   'Ninja offshore Wind ')
    stat_print(norm_ninja_onshore, norm_hp41,   'Ninja onshore Wind ')
    stat_print(norm_ninja_both, norm_hp41,   'Ninja combined Wind ')
    stat_print(norm_ninja_future, norm_hp41,   'Ninja future Wind ')
    stat_print(norm_ninja_current, norm_hp41,   'Ninja current Wind ')
    stat_print(norm_ninja_pv, norm_hp41,   'Ninja PV ')
    if args.ngrid:
        stat_print(norm_ngrid_wind, norm_hp41,   'Nat grid wind ')

    print('Comparison to all heat pumps')
    stats.print_stats_header()
    if not args.cardenas and args.frequency != 'H':
        stat_print(norm_kf_pv, norm_hp_all,     'KF PV   ')
        stat_print(norm_kf_wind, norm_hp_all,   'KF Wind ')
    stat_print(norm_ninja_offshore, norm_hp_all,   'Ninja offshore Wind ')
    stat_print(norm_ninja_onshore, norm_hp_all,   'Ninja onshore Wind ')
    stat_print(norm_ninja_both, norm_hp_all,   'Ninja combined Wind ')
    stat_print(norm_ninja_future, norm_hp_all,   'Ninja future Wind ')
    stat_print(norm_ninja_current, norm_hp_all,   'Ninja current Wind ')
    stat_print(norm_ninja_pv, norm_hp_all,   'Ninja PV ')
    if args.ngrid:
        stat_print(norm_ngrid_wind, norm_hp_all,   'Nat grid wind ')

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
    cr_e = []
    cr_f = []
    cr_v = []
    cf_pv = 1
    cf_wind = 1
    clabel = 'Wind Capacity Fraction'
    if args.energy:
        cf_pv = 0.1085
        cf_wind = 0.3878
        clabel = 'Wind Energy Fraction'
    
    for fraction in np.arange(0.0,1.0+args.step, args.step):
        supply = fraction * norm_ninja_both + (1-fraction) * norm_ninja_pv
        ewind = fraction * cf_wind
        epv   = (1-fraction) * cf_pv
        efraction = ewind / ( ewind + epv )
        fs.append(efraction)
        if args.base>0:
            supply = supply + args.base
            supply = normalize(supply)
        re.append(correlation(supply, norm_existing))
        ra.append(correlation(supply, norm_hp41))
        area_e.append(curve_area(norm_existing - supply))
        area_f.append(curve_area(norm_hp41 - supply))
        cr_e.append(charge_rate(norm_existing - supply))
        cr_f.append(charge_rate(norm_hp41 - supply))
        cr_v.append(charge_rate(norm_ev - supply))
        variance_e.append(variance(norm_existing - supply))
        variance_f.append(variance(norm_hp41 - supply))
        sd_e.append(sd(norm_existing - supply))
        sd_f.append(sd(norm_hp41 - supply))
    data = { 'fraction' : fs, 're' : re, 'ra' : ra, 'area_e' : area_e, 'area_f' : area_f, 'variance_e': variance_e, 'variance_f': variance_f, 'sd_e' : sd_e, 'sd_f' : sd_f, 'charge_rate_e' : cr_e, 'charge_rate_f' : cr_f, 'charge_rate_v' : cr_v }
    df = pd.DataFrame(data=data)
    if args.plot:
        # Correlation
        plt.plot(df['fraction'], df['re'], label='existing heating')
        plt.plot(df['fraction'], df['ra'], label='41% heat pumps')
        freq = 'Daily '
        if args.frequency == 'M':
            freq = 'Monthly '
        if args.frequency == 'H':
            freq = 'Hourly '
        plt.title('{}correlation of wind fraction in the supply to the demand'.format(freq))
        plt.xlabel(clabel, fontsize=15)
        plt.ylabel('Pearsons correlation coefficient (R)', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
        df['area_e'] = normalize(df['area_e'])
        df['area_f'] = normalize(df['area_f'])
        # Area under the net demand curve
        plt.plot(df['fraction'], df['area_e'], label='existing heating')
        plt.plot(df['fraction'], df['area_f'], label='41% heat pumps')
        freq = 'Daily '
        plt.title('Wind fraction vs area under the net demand curve')
        plt.xlabel(clabel, fontsize=15)
        plt.ylabel('Area under net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
#j      df['charge_rate_e'] = normalize(df['charge_rate_e'])
#       df['charge_rate_f'] = normalize(df['charge_rate_f'])
        # Charge rate
        plt.plot(df['fraction'], df['charge_rate_e'], label='existing heating')
        plt.plot(df['fraction'], df['charge_rate_f'], label='41% heat pumps')
        plt.plot(df['fraction'], df['charge_rate_v'], label='Mostly EVs')
        freq = 'Daily '
        plt.title('Wind fraction vs charge rate of the net demand curve')
        plt.xlabel(clabel, fontsize=15)
        plt.ylabel('Charge rate of net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
        df['variance_e'] = normalize(df['variance_e'])
        df['variance_f'] = normalize(df['variance_f'])
        # variance under the net demand curve
        plt.plot(df['fraction'], df['variance_e'], label='existing heating')
        plt.plot(df['fraction'], df['variance_f'], label='41% heat pumps')
        freq = 'Daily '
        plt.title('Wind fraction vs variance of the net demand curve')
        plt.xlabel(clabel, fontsize=15)
        plt.ylabel('variance of net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        # normalize
        df['sd_e'] = normalize(df['sd_e'])
        df['sd_f'] = normalize(df['sd_f'])
        # Standard deviation of the net demand curve
        plt.plot(df['fraction'], df['sd_e'], label='existing heating')
        plt.plot(df['fraction'], df['sd_f'], label='41% heat pumps')
        freq = 'Daily '
        plt.title('Wind fraction vs standard deviation of the net demand curve')
        plt.xlabel(clabel, fontsize=15)
        plt.ylabel('standard deviation of net demand curve', fontsize=15)
        plt.legend(loc='upper center')
        plt.show()

        plot_net_demand(norm_ninja_both, norm_ninja_pv, 0.80, norm_existing)


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
