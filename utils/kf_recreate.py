# python script to recreate KFs storage model

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
import storage

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

# Demand MWh
demand_filename = '/home/malcolm/uclan/data/kf/UKDailyELD19832014.csv'
demand = pd.read_csv(demand_filename, header=None, squeeze=True)
d = pd.date_range(start = '1984-01-01', end = '2013-12-31', freq='D' )
used_values = demand.values[365:]
print(len(used_values))
electric = pd.Series(demand.values[365:11323], d, dtype='float64', name='demand')
print(electric)
print('Demand peak {} min {} mean {} total {}'.format(electric.max(), electric.min(), electric.mean(), electric.sum() ) )

# scale
total_energy = electric.sum() / 30
new_values = np.empty(0)
for year in range(1984,2014):
    print(year)
    year_electric = electric[str(year)]
#       print(year_electric)
    adjustment = (total_energy - year_electric.sum()) / year_electric.size
    print("Year {} len {} adjustment {} total {}".format(year, year_electric.size, adjustment, total_energy) )
    year_electric = year_electric + adjustment
    new_values = np.concatenate((new_values, year_electric.values))
electric = pd.Series(new_values, d, dtype='float64', name='electric')

# scale england and wales to scotland - comes after the adjustment!
scotland_factor = 1.104
electric = electric * scotland_factor

# compare with KF version
kf_adjusted_filename = '/home/malcolm/uclan/data/kf/JDsame8413EL.csv'
kf_adjusted = pd.read_csv(kf_adjusted_filename, header=0, squeeze=True)
print(kf_adjusted)
kf_electric = kf_adjusted['el']
kf_electric = kf_electric * scotland_factor
print(kf_electric)
print('Scaled Electric min max mean n_values')
print('KF              {:.2f}  {:.2f}  {:.2f}   {}'.format(kf_electric.min(), kf_electric.max(), kf_electric.mean(), len(kf_electric) ) )
print('MP              {:.2f}  {:.2f}  {:.2f}   {}'.format(electric.min(), electric.max(), electric.mean(), len(electric) ) )
quit()

# generation
wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
kf_wind = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
kf_pv = pd.read_csv(pv_filename, header=None, squeeze=True)
wind = pd.Series(kf_wind.values[0:len(d)], electric.index, dtype='float64', name='wind')
pv = pd.Series(kf_pv.values[0:len(d)], d, dtype='float64', name='pv')
pv.index = electric.index

# convert to MWh
wind = wind * 1e-6
pv = pv * 1e-6

print(wind)
print(pv)
print('KF energy totals: wind {} pv {} Number of values: wind {} pv {}'.format(wind.sum(), pv.sum(), len(wind), len(pv)) )
print('Demand peak {} min {} mean {} total {}'.format(electric.max(), electric.min(), electric.mean(), electric.sum() ) )

# normalise into days
wind = wind / wind.mean()
pv = pv / pv.mean()
electric = electric / electric.mean()

# convert to capacity factors
kf_wcf = 0.28
kf_pcf = 0.1156
wind = wind * kf_wcf
pv = pv * kf_pcf

# calculate storage

eta = 0.75
df = storage.storage_grid(electric, wind, pv, eta, False, 60, 0.1, 0.0, None)

output_file = '/home/malcolm/uclan/output/kf_recreate/sharesENH.csv'
df.to_csv(output_file)
