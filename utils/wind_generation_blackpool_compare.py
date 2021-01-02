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

# Time series of wind power for midas weather stations from ERA5 weather.
era_filename = '/home/malcolm/uclan/output/wind/era520181.csv'
# Time series of wind for power midas weather stations from MIDAS weather.
midas_filename = '/home/malcolm/uclan/output/wind/midas20181.csv'
# Time series of wind power for blackpool from squires gate
ninja_filename = '/home/malcolm/uclan/data/ninja/ninja_wind_53.7746_-3.0365_corrected.csv'
# Ninja capacity factors
ninja_filename_cfs = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_near-termfuture-merra-2_corrected.csv'
# national grid generation
national_grid_filename = '/home/malcolm/uclan/data/ElectricityDemandData_2018.csv'
# elexon supply
elexon_filename = '/home/malcolm/uclan/data/elexon/GenerationbyFuelType_2018.csv'

# read in the data
ninja = readers.read_ninja(ninja_filename)
era = pd.read_csv(era_filename, header=0, parse_dates=[0], index_col=0 )
midas = pd.read_csv(midas_filename, header=0, parse_dates=[0], index_col=0 )

# convert to daily by mean to get average wind speed.
wind_era = era['wind_b'].resample('D').mean()
wind_midas = midas['wind_b'].resample('D').mean()

# output wind plots

wind_era.plot(label='era5')
wind_midas.plot(label='midas')
plt.title('2018 Blackpool wind speed from midas and ERA5')
plt.xlabel('Day of the year')
plt.ylabel('Wind Speed (M/S)')
plt.legend(loc='upper right')
plt.show()

# convert to daily by summing the hours, so this is energy/day in kWh
ninja = ninja.resample('D').sum()
era = era.resample('D').sum()
midas = midas.resample('D').sum()

# output power plots
power_era_b = era['power_b'] * 0.001
power_mid_b = midas['power_b'] * 0.001
power_nin_b = ninja * 0.001

# era['power_b'].plot(label='era5')
# midas['power_b'].plot(label='midas')
# ninja.plot(label='ninja')
power_era_b.plot(label='era5')
power_mid_b.plot(label='midas')
power_nin_b.plot(label='ninja')

plt.title('2018 Blackpool wind power from midas and ERA5 and ninja')
plt.xlabel('Day of the year')
plt.ylabel('Energy generated (MWh)')
plt.legend(loc='upper right')
plt.show()

#ninja.index = ninja.index.tz_localize('Europe/London')
ninja.index = ninja.index.tz_localize('UTC')

stats.print_stats_header()
stats.print_stats(wind_era, wind_midas, 'era and midas wind', 1, True, 'ERA5 wind speed (m/s)', 'MIDAS wind speed (m/s)' )
stats.print_stats(era['power_b'], midas['power_b'], 'era and midas power', 1, True, 'ERA Energy (MWh)', 'MIDAS Energy (MWh)' )
stats.print_stats(era['power_b'], ninja, 'era and ninja power', 1, True, 'ERA Energy (MWh)', 'Ninja Energy (MWh)' )

annual_era = {}
annual_midas = {}
locations =  ['a', 'b', 'c', 'l', 's', 'w']

for location in locations:
    annual_era[location] = era['power_' + location].sum()
    print(location, annual_era[location])
    annual_midas[location] = midas['power_' + location].sum()
max_era = max(annual_era.values())
max_midas = max(annual_midas.values())
# print(max_era, max_midas)

# turbine rated power in kW
rated_power = 2500

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

# adjust the capacity factor inline with the assumed load factor of 0.28
# from KF

era_total = era_total * ( 0.28 / era_total.mean() )
midas_total = midas_total * ( 0.28 / midas_total.mean() )
# print(era_total)
# print(midas_total)

# read in the ninja wind cf's for 2018
ninja = readers.read_ninja_country(ninja_filename_cfs)
ninja = ninja['2018-01-01 00:00:00' : '2018-12-31 23:00:00']
# convert to hourly by summing
ninja = ninja.resample('D').mean()
print(ninja)

# read in national grid embedded generation and capacity
national_grid = readers.read_electric(national_grid_filename)
# print('national_grid')
# print(national_grid)
# and calculate capacity factor.
national_grid_cf = national_grid['EMBEDDED_WIND_GENERATION'] / national_grid['EMBEDDED_WIND_CAPACITY']
# print(national_grid_cf)

# read in elexon generation
elexon = readers.read_elexon_supply(elexon_filename)
print('ELEXON')
# print(elexon)
# calculate elexon wind generation capacity factor.
wind_capacity_2018 = 9360.0
elexon_cf = elexon['WIND'] / wind_capacity_2018
# convert to daily (mean, because on CF's not energy)
elexon_cf = elexon_cf.resample('D').mean()
print(elexon_cf)

print('era_total {} midas_total {} ninja on {} off {} nat {} national_grid_cf {} elexon_cf {}'.format( len(era_total), len(midas_total), len(ninja['offshore']), len(ninja['onshore']), len(ninja['national']), len(national_grid_cf), len(elexon_cf) ) )


# plot all 6
era_total.plot(label='era5')
midas_total.plot(label='midas')
ninja['offshore'].plot(label='ninja offshore')
ninja['onshore'].plot(label='ninja onshore')
national_grid_cf.plot(label='national grid embedded')
elexon_cf.plot(label='elexon generation')
plt.title('2018 Wind power capacity factors')
plt.xlabel('Day of the year')
plt.ylabel('Capacity factor')
plt.legend(loc='upper right')
plt.show()

ninja.index = ninja.index.tz_localize('UTC')

# print original CFS

print("Capacity factor  Annual   Max")
annual_cf_era = era_total.sum() / 365
max_cf_era = era_total.max()
print("ERA                  {0:.2f} {1:.2f}".format(annual_cf_era, max_cf_era))
annual_cf_midas = midas_total.sum() / 365
max_cf_midas = midas_total.max()
print("MIDAS                {0:.2f} {1:.2f}".format(annual_cf_midas, max_cf_midas))
annual_cf_ninja_nat = ninja['national'].sum() / 365
max_cf_ninja_nat = ninja['national'].max()
print("Ninja combined       {0:.2f} {1:.2f}".format(annual_cf_ninja_nat, max_cf_ninja_nat))
annual_cf_ninja_on = ninja['onshore'].sum() / 365
max_cf_ninja_on = ninja['onshore'].max()
print("Ninja onshore        {0:.2f} {1:.2f}".format(annual_cf_ninja_on, max_cf_ninja_on))
annual_cf_ninja_off = ninja['offshore'].sum() / 365
max_cf_ninja_off = ninja['offshore'].max()
print("Ninja offshore       {0:.2f} {1:.2f}".format(annual_cf_ninja_off, max_cf_ninja_off))
annual_cf_ng = national_grid_cf.sum() / 365
max_cf_ng = national_grid_cf.max()
print("National Grid        {0:.2f} {1:.2f}".format(annual_cf_ng, max_cf_ng))
annual_cf_ex = elexon_cf.sum() / 365
max_cf_ex = elexon_cf.max()
print("Elexon Generation    {0:.2f} {1:.2f}".format(annual_cf_ex, max_cf_ex))

ninja_national = ninja['national']
# Normalise CF before stats

elexon_cf = stats.normalize(elexon_cf)
era_total = stats.normalize(era_total)
midas_total = stats.normalize(midas_total)
ninja_national = stats.normalize(ninja_national)
national_grid_cf = stats.normalize(national_grid_cf)

# do stats for all 4
print('Stats compared to elexon')
stats.print_stats_header()
stats.print_stats(era_total, elexon_cf, 'era elexon', 1, True, 'Capacity Factor ERA', 'Capacity Factor Elexon')
stats.print_stats(midas_total, elexon_cf, 'midas', 1, True, 'Capacity Factor MIDAS', 'Capacity Factor Elexon')
stats.print_stats(ninja['national'], elexon_cf, 'ninja', 1, True, 'Capacity Factor Ninja', 'Capacity Factor Elexon')
stats.print_stats(national_grid_cf, elexon_cf, 'national grid', 1, True, 'Capacity Factor Embedded', 'Capacity Factor Elexon')
