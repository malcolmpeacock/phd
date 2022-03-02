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

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--scale', action="store_true", dest="scale", help='Scale to ninja CF', default=False)
args = parser.parse_args()

wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
kf_wind = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
kf_pv = pd.read_csv(pv_filename, header=None, squeeze=True)

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
ninja_pv = ninja_pv.resample('D').mean()

print('Extracting Wind ...')
ninja_wind = wind_hourly[ninja_start : ninja_end]
ninja_wind = ninja_wind['onshore']
ninja_wind = ninja_wind.resample('D').mean()

print(ninja_wind)
print(ninja_pv)

kf_wind.index = ninja_wind.index
kf_pv.index = ninja_pv.index
# total energy to supply the load ( MWh )
#total_energy = 305000000.0   # Kwh per year?
#total_demand = 9169197139968000.00
total_demand = kf_pv.sum()
energy_per_year = total_demand / 30
print(kf_wind)
print(kf_pv)
print('KF energy totals: wind {} pv {} Number of values: wind {} pv {} Energy per year {} wh'.format(kf_wind.sum(), kf_pv.sum(), len(kf_wind), len(kf_pv), energy_per_year ) )

num_days = len(kf_wind)
# Solar Capacity ( peak in Watts? )
wp1 = 83320707917.63
wp2 = 74494182927.31
wp3 = 75454611985.58
wp4 = 69439139838.44

pv_energy_joules = (wp1 + wp2 + wp3 + wp4) * num_days * 60 * 60 * 24
# kwh
pv_energy = pv_energy_joules / ( 60 * 60 )
pv_cf = total_demand / pv_energy
#pv_cf = total_energy / pv_energy

# wind capacity factors
wcf1 = 0.39
wcf2 = 0.33
wcf3 = 0.46
wcf4 = 0.37
wcf5 = 0.49
wcf6 = 0.39
wind_cf = (wcf1 + wcf2 + wcf3 + wcf4 + wcf5 + wcf6) / 6

print('Capacity Factor : wind {} pv {}'.format(wind_cf, pv_cf ) )
ninja_wind_cf = ninja_wind.mean()
ninja_pv_cf = ninja_pv.mean()
print('Ninja Capacity Factor : wind {} pv {}'.format(ninja_wind_cf, ninja_pv_cf ) )

energy_per_day = total_demand / num_days
# convert to ninja cf
if args.scale:
    kf_pv = kf_pv * ninja_pv_cf /energy_per_day
    kf_wind = kf_wind * ninja_wind_cf /energy_per_day
else:
    kf_pv = kf_pv * pv_cf /energy_per_day
    kf_wind = kf_wind * wind_cf /energy_per_day
print('Capacity Factor after change : wind {} pv {} energy/day {}'.format(kf_wind.mean(), kf_pv.mean(), energy_per_day ) )

stats.print_stats_header()
stats.print_stats(kf_pv, ninja_pv,     'PV   Compared to Ninja')
stats.print_stats(kf_wind, ninja_wind, 'Wind Compared to Ninja')

print('            Ninja      KF')
print('PV   max    {:.2f}    {:.2f} '.format(ninja_pv.max(), kf_pv.max() ) )
print('PV   min    {:.2f}    {:.2f} '.format(ninja_pv.min(), kf_pv.min() ) )
print('PV   mean   {:.2f}    {:.2f} '.format(ninja_pv.mean(), kf_pv.mean() ) )
print('PV   std    {:.2f}    {:.2f} '.format(ninja_pv.std(), kf_pv.std() ) )
print('PV   var    {:.2f}    {:.2f} '.format(ninja_pv.var(), kf_pv.var() ) )
print('            Ninja      KF')
print('WIND max    {:.2f}    {:.2f} '.format(ninja_wind.max(), kf_wind.max() ) )
print('WIND min    {:.2f}    {:.2f} '.format(ninja_wind.min(), kf_wind.min() ) )
print('WIND mean   {:.2f}    {:.2f} '.format(ninja_pv.mean(), kf_pv.mean() ) )
print('WIND std    {:.2f}    {:.2f} '.format(ninja_wind.std(), kf_wind.std() ) )
print('WIND var    {:.2f}    {:.2f} '.format(ninja_wind.var(), kf_wind.var() ) )


kf_pv.plot(color='blue', label='PV Generation from Fragaki et. al')
ninja_pv.plot(color='red', label='PV Generation from ninja')
plt.title('Comparison of daily UK PV genreation')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Energy', fontsize=15)
plt.legend(loc='upper right')
plt.show()

kf_wind.plot(color='blue', label='Wind Generation from Fragaki et. al ')
ninja_wind.plot(color='red', label='Wind Generation from ninja')
plt.title('Comparison of daily UK Wind genreation')
plt.xlabel('Time', fontsize=15)
plt.ylabel('Energy', fontsize=15)
plt.legend(loc='upper right')
plt.show()

# Distributions
#ax = kf_pv.plot.hist(bins=20, label='kf', facecolor="None", edgecolor='red')
ax = kf_pv.plot.hist(bins=20, label='kf' )
plt.title('PV distribution')
plt.xlabel('Capacity Factor', fontsize=15)

#ninja_pv.plot.hist(bins=20, ax=ax, label='ninja', facecolor="None", edgecolor='blue')
ninja_pv.plot.hist(bins=20, ax=ax, label='ninja' )
plt.xlabel('Capacity Factor', fontsize=15)
plt.legend(loc='upper right')
plt.show()

ax = kf_wind.plot.hist(bins=20, label='kf')
plt.title('Wind distribution')
plt.xlabel('Capacity Factor', fontsize=15)

ninja_wind.plot.hist(bins=20, ax=ax, label='ninja' )
plt.xlabel('Capacity Factor', fontsize=15)
plt.legend(loc='upper right')
plt.show()
