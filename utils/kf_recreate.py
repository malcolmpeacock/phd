# python script to recreate KFs storage model

# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math

# custom code
import stats
import readers
import storage

def kf_storage(demand, wind, pv, eta, fine=False):
    ndays = len(demand)
    print('Number of days {}'.format(ndays))
    results = { 'f_pv' : [], 'f_wind' : [], 'storage' : [], 'sf' : [], 'cw' : [], 'kf_storage' : [], 'last_zero' : [], 'non_zero' : [], 'last' : [] }
    sf_step = 10
    cw_step = 5
    sf_start = 170
    sf_stop = 210
    if fine:
        sf_step = 1
        cw_step = 1
        sf_start = 100
        sf_stop = 230

    for i_sf in range(sf_start,sf_stop,sf_step):
        sf = i_sf / 100
        sys.stdout.write('\rCalculating sf {:.2f}'.format(sf) )
        for icw in range(0, 101, cw_step):
            cw = icw/100
            cs = 1 - cw
            f_wind = sf * cw / kf_wcf
            f_pv = sf * cs / kf_pcf
            net = demand - (wind * f_wind + pv * f_pv)
            store_hist = storage.storage(net, eta)
            store_size = store_hist.min()
            storage_days = store_size * -1.0
            store_last = store_hist.iat[-1]
            results['f_pv'].append(f_pv)
            results['f_wind'].append(f_wind)
            results['storage'].append(storage_days)
            results['sf'].append(sf)
            results['cw'].append(cw)
            results['kf_storage'].append(storage_days * 100 / ndays)
            results['last'].append(store_last)
            #
            last_zero, non_zero = storage.check_zero(store_hist)
            results['last_zero'].append(last_zero)
            results['non_zero'].append(non_zero)

    df = pd.DataFrame(data=results)
    print(' ')
    print('Storage calculation Finished')
    return df

# process command line
parser = argparse.ArgumentParser(description='Recreate kf output')
parser.add_argument('--kf', action="store_true", dest="kf", help='Do shares as per kf method', default=False)
parser.add_argument('--fine', action="store_true", dest="fine", help='Fine resolition', default=False)
parser.add_argument('--eta', action="store", dest="eta", help='Round Trip Efficiency', default=75, type=int)
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

# calculate charge and discharge efficiency from round trip efficiency
eta = math.sqrt(args.eta / 100)
print('Round trip efficiency {} Charge/Discharge {} '.format(args.eta / 100, eta) )


# calculate storage
if args.kf:
    df = kf_storage(electric, wind, pv, eta, args.fine)
    output_file = '/home/malcolm/uclan/output/kf_recreate/sharesKFS{:02d}.csv'.format(args.eta)

else:

    df = storage.storage_grid(electric, wind, pv, eta, False, 60, 0.1, 0.0, None)
    output_file = '/home/malcolm/uclan/output/kf_recreate/sharesENHS{:02d}.csv'.format(args.eta)

df.to_csv(output_file)
