# python script to compare wind and wind generation from:
#  midas, era5, renewables ninja

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

# PV Power generation for MIDAS weather stations from ERA5 data
era_filename = '/home/malcolm/uclan/output/pv/ERA520181.csv'
# PV Power generation for MIDAS weather stations from MIDAS data
midas_filename = '/home/malcolm/uclan/output/pv/midas20181.csv'
# PV Power generation for Cambourne from ninja.
ninja_filename = '/home/malcolm/uclan/data/ninja/ninja_pv_50.2178_-5.3266_uncorrected.csv'
# Ninja capacity factors
ninja_filename_cfs = '/home/malcolm/uclan/data/ninja/ninja_pv_country_GB_merra-2_corrected.csv'
# national grid embedded generation
national_grid_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'

# read in the data
ninja = readers.read_ninja(ninja_filename)
era = pd.read_csv(era_filename, header=0, parse_dates=[0], index_col=0 )
midas = pd.read_csv(midas_filename, header=0, parse_dates=[0], index_col=0 )

# convert to daily by summing the hours, so this is energy/day in kWh
ninja = ninja.resample('D').sum()
era = era.resample('D').sum()
midas = midas.resample('D').sum()

# output GHI plots for ninja and midas combourne

era['ghi_c'].plot(label='era5')
midas['ghi_c'].plot(label='midas')
plt.title('2018 Cambourne GHI from midas and ERA5')
plt.xlabel('Day of the year')
plt.ylabel('GHI - Irradiance on POA (Watts/Square Metre)')
plt.legend(loc='upper right')
plt.show()

# output pv power plots
power_era_c = era['power_c'] * 0.001
power_mid_c = midas['power_c'] * 0.001
# power_nin_c = ninja * 0.001
power_nin_c = ninja
pv_generation_compare.py
power_era_c.plot(label='era5')
power_mid_c.plot(label='midas')
power_nin_c.plot(label='ninja')

plt.title('2018 Cambourne PV power from midas and ERA5 and ninja')
plt.xlabel('Day of the year')
plt.ylabel('Energy generated (MWh)')
plt.legend(loc='upper right')
plt.show()

ninja.index = ninja.index.tz_localize('Europe/London')

stats.print_stats_header()
stats.print_stats(era['ghi_c'], midas['ghi_c'], 'era and midas ghi')
stats.print_stats(era['power_c'], midas['power_c'], 'era and midas power')
stats.print_stats(era['power_c'], ninja, 'era and ninja power')

annual_era = {}
annual_midas = {}
locations =  ['a', 'c', 'e', 'u']
for location in locations:
    annual_era[location] = era['power_' + location].sum()
    print(location, annual_era[location])
    annual_midas[location] = midas['power_' + location].sum()
max_era = max(annual_era.values())
max_midas = max(annual_midas.values())
print(max_era, max_midas)

# PV rated power in kW ????
rated_power = 1000.0

for location in locations:
    # scale to max energy
    era['power_' + location] = era['power_' + location] * (max_era / annual_era[location])
    midas['power_' + location] = midas['power_' + location] * (max_midas / annual_midas[location])

for location in locations:
    # convert to capacity factor
    era['power_' + location] = era['power_' + location] / (rated_power * 24)
    midas['power_' + location] = midas['power_' + location] / (rated_power * 24)

era_cf = pd.concat([era['power_' + location] for location in locations], axis=1)
midas_cf = pd.concat([midas['power_' + location] for location in locations], axis=1)

# create mean to represent whole country
era_total = era_cf.sum(axis=1) / len(locations)
midas_total = midas_cf.sum(axis=1) / len(locations)
print(era_total)
print(midas_total)

# read in the ninja PV cf's for 2018
ninja = readers.read_ninja_country(ninja_filename_cfs)
ninja = ninja['2018-01-01 00:00:00' : '2018-12-31 23:00:00']
# convert to hourly by summing
ninja = ninja.resample('D').mean()
print(ninja)

# read in national grid embedded generation and capacity
national_grid = readers.read_electric(national_grid_filename)
print(national_grid)
# and calculate capacity factor.
national_grid_cf = national_grid['EMBEDDED_SOLAR_GENERATION'] / national_grid['EMBEDDED_SOLAR_CAPACITY']
print(national_grid_cf)

# plot all 3
era_total.plot(label='era5')
midas_total.plot(label='midas')
ninja['national'].plot(label='ninja')
national_grid_cf.plot(label='national grid')
plt.title('2018 PV power capacity factors')
plt.xlabel('Day of the year')
plt.ylabel('Capacity factor')
plt.legend(loc='upper right')
plt.show()

# do stats for all 3
ninja.index = ninja.index.tz_localize('Europe/London')
print('Stats compared to national grid')
stats.print_stats_header()
stats.print_stats(era_total, national_grid_cf, 'era')
stats.print_stats(midas_total, national_grid_cf, 'midas')
stats.print_stats(ninja['national'], national_grid_cf, 'ninja')

print("Annual Capacity factors")
annual_cf_era = era_total.sum() / 365
print("ERA            {0:.2f}".format(annual_cf_era))
annual_cf_midas = midas_total.sum() / 365
print("MIDAS          {0:.2f}".format(annual_cf_midas))
annual_cf_ninja_nat = ninja['national'].sum() / 365
print("Ninja combined {0:.2f}".format(annual_cf_ninja_nat))
annual_cf_ng = national_grid_cf.sum() / 365
print("National Grid  {0:.2f}".format(annual_cf_ng))
