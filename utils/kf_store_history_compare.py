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

def hist_comp(sf, cw):
    print('sf {:02d} cw {:02d} '.format(sf, cw) )
    kf_filename = '/home/malcolm/uclan/data/kf/ProfileSFCW{:02d}{:02d}.csv'.format(sf, cw)
    kf_hist = pd.read_csv(kf_filename, header=None, squeeze=True)
    kf_hist.columns = ['y', 'm', 'd', 's']
    mp_filename = '/home/malcolm/uclan/output/kf_recreate/hist/sf{:02d}cw{:02d}.csv'.format(sf, cw)
    mp_hist = pd.read_csv(mp_filename, header=0, squeeze=True)
    mp_hist.columns = ['d', 's']


    mp_store = mp_hist['s'] * 1e12
    kf_store = kf_hist['s'] * 1e12
#   print(kf_store)
#   print(mp_store)
    print('Store history number of values: kf {} mp {} minimum kf {} mp {} '.format(len(kf_store), len(mp_store), kf_store.min(), mp_store.min() ) )

    energy_per_day = 836757995855.537
    num_days = 10958
    total_energy = energy_per_day * num_days
    kf_s_kf = kf_store.min() * 100 / total_energy
    kf_s_mp = mp_store.min() * 100 / total_energy
    days_kf = kf_s_kf / num_days
    days_mp = kf_s_mp / num_days
    print('kf_storage kf {} mp {} days kf {} mp {} '.format(kf_s_kf, kf_s_mp, days_kf, days_mp) )

    if args.plot:
        mp_store.plot(color='blue', label='KF Defecit')
        kf_store.plot(color='red', label='MP Defecit')
        plt.title('Comparison of defecit MP and KF')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Energy', fontsize=15)
        plt.legend(loc='upper right')
        plt.show()

    stats.print_stats_header()
    stats.print_stats(mp_store, kf_store,     'MP Compared to KF')

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

hist_comp(17, 0)
hist_comp(17, 5)
hist_comp(18, 5)
hist_comp(19, 95)
