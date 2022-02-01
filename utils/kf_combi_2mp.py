# Convert kf combi.txt output to mp

# library stuff
import sys
import pandas as pd
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy import interpolate
import math

# custom code
import stats
import readers
import storage
import bilinear2 as bil

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--min', action="store_true", dest="min", help='Plot the minimum generation line', default=False)
parser.add_argument('--annotate', action="store_true", dest="annotate", help='Annotate the shares heat map', default=False)
args = parser.parse_args()

# read kf data
kf_filename = "/home/malcolm/uclan/data/kf/Combi.csv"
combi = pd.read_csv(kf_filename, header=0)
combi.columns = ['SF', 'CW', 'storage']
# put storage back to days
ndays = 11323 - 366
combi['storage'] = combi['storage'] * ndays / 100.0
combi['CS'] = 1 - combi['CW']
#
combi['f_wind'] = combi['CW'] * combi['SF'] / 0.28
combi['f_pv'] = combi['CS'] * combi['SF'] / 0.116
#
s75 = combi[['f_wind', 'f_pv', 'storage']].copy()
# add extra columns
s75['last'] = 0.0
s75['charge'] = 0.0
s75['discharge'] = 0.0
s75['wind_energy'] = 0.0
s75['pv_energy'] = 0.0
s75['gw_wind'] = 0.0
s75['gw_pv'] = 0.0
s75['fg'] = combi['CW'] * combi['SF']

# sort the same way
shares = s75.sort_values(['f_pv', 'f_wind'], ascending=[True, True])
print(shares)

output_filename = "/home/malcolm/uclan/output/kfig8/sharesCM.csv"
shares.to_csv(output_filename)

if args.plot:
    # read in mp shares data
    mp = pd.read_csv("/home/malcolm/uclan/output/kfig8/sharesNNH.csv")
    # Only include viable storage
    mp = mp[mp['last']==0]
    print(mp)

    # interpolate from my data to kf points
#   spline = RectBivariateSpline(f_wind, f_pv, s)
    mp_storage=[]
    kf_storage=[]
    pv_storage=[]
    wd_storage=[]
    df = mp[['f_wind', 'f_pv', 'storage']]
#   print(df)
    for index, value in s75.iterrows():
        print(value)
        print('Bilinear {} {} '.format(value['f_wind'], value['f_pv']) )
        bl = bil.bilinear(value['f_wind'], value['f_pv'], df, 'storage')
        if not math.isnan(bl):
            mp_storage.append(bl)
            kf_storage.append(value['storage'])
            wd_storage.append(value['f_wind'])
            pv_storage.append(value['f_pv'])

    data = {'f_wind': wd_storage, 'f_pv': pv_storage, 'kf_storage': kf_storage, 'mp_storage' : mp_storage }
    df = pd.DataFrame(data)
    print(df)

    df['diff_storage'] = df['mp_storage'] - df['kf_storage']

    # plot a heat map of the kf storage. 
    ax = df.plot.scatter(x='f_wind', y='f_pv', c='kf_storage', colormap='viridis')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('Data from running MatLab')
#   if annotate:
#       for i, point in df.iterrows():
#           ax.text(point['f_wind'],point['f_pv'],'{:.1f}'.format(point['storage']))
    plt.show()

    # plot a heat map of the mp storage. 
    ax = df.plot.scatter(x='f_wind', y='f_pv', c='mp_storage', colormap='viridis')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('MP ninja interpolated to original')
    plt.show()

    # plot a heat map of the storage difference. 
    ax = df.plot.scatter(x='f_wind', y='f_pv', c='diff_storage', colormap='viridis')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('Difference between original and mp')
    plt.show()

    # mp and kf storage about 30 - wind variation
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df30 = df[(df['kf_storage']>28) & (df['kf_storage']<32)]
    ax1.scatter(df30['f_wind'], df30['kf_storage'], label='original excel', color='blue' )
    ax1.scatter(df30['f_wind'], df30['mp_storage'], label='mp interpolated', color='red' )
    plt.xlabel('Proportion of wind')
    plt.ylabel('Days storage')
    plt.title('Storage about 30 days mp and original paper MatLab (pv annotated)')
    plt.legend(loc='lower left', fontsize=15)
    for i, point in df30.iterrows():
        ax1.text(point['f_wind'],point['kf_storage'],'{:.2f}'.format(point['f_pv']))
        ax1.text(point['f_wind'],point['mp_storage'],'{:.2f}'.format(point['f_pv']))
    plt.show()

    # mp and kf storage about 30 - solar variation
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df30 = df[(df['kf_storage']>28) & (df['kf_storage']<32)]
    ax1.scatter(df30['f_pv'], df30['kf_storage'], label='original excel', color='blue' )
    ax1.scatter(df30['f_pv'], df30['mp_storage'], label='mp interpolated', color='red' )
    plt.xlabel('Proportion of solar')
    plt.ylabel('Days storage')
    plt.title('Storage about 30 days mp and original paper MatLab (wind annotated)')
    plt.legend(loc='lower right', fontsize=15)
    for i, point in df30.iterrows():
        ax1.text(point['f_pv'],point['kf_storage'],'{:.2f}'.format(point['f_wind']))
        ax1.text(point['f_pv'],point['mp_storage'],'{:.2f}'.format(point['f_wind']))
    plt.show()

    # plot storage vs wind for a particular days of storage.
    days30 = s75[(s75['storage']>28) & (s75['storage']<32)]
    print(days30)
    ax = days30.plot.scatter(x='f_wind', y='f_pv', c='storage', colormap='viridis')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('Storage about 30 days from original paper MatLab')
    for i, point in days30.iterrows():
        ax.text(point['f_wind'],point['f_pv'],'{:.1f}'.format(point['storage']))
    plt.show()

    # Energy
    mp_energy=[]
    kf_energy=[]
    kf_storage=[]
    pv_energy=[]
    wd_energy=[]
    mp['energy'] = mp['wind_energy'] + mp['pv_energy']
    df = mp[['f_wind', 'f_pv', 'energy']]
#   print(df)
    for index, value in s75.iterrows():
        print('Bilinear {} {} '.format(value['f_wind'], value['f_pv']) )
        bl = bil.bilinear(value['f_wind'], value['f_pv'], df, 'energy')
        if not math.isnan(bl):
            mp_energy.append(bl)
            kf_energy.append(value['fg'])
            kf_storage.append(value['storage'])
            wd_energy.append(value['f_wind'])
            pv_energy.append(value['f_pv'])

    data = {'f_wind': wd_energy, 'f_pv': pv_energy, 'kf_energy': kf_energy, 'mp_energy' : mp_energy, 'kf_storage': kf_storage }
    df = pd.DataFrame(data)
    df['energy_diff'] = df['kf_energy'] - df['mp_energy']
    print(df)
    # plot a heat map of the storage difference. 
    ax = df.plot.scatter(x='f_wind', y='f_pv', c='energy_diff', colormap='viridis')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('Difference between original and mp energy')
    plt.show()

    # mp and kf energy about 30 - solar variation
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    df30 = df[(df['kf_storage']>28) & (df['kf_storage']<32)]
    ax1.scatter(df30['f_pv'], df30['kf_energy'], label='original excel', color='blue' )
    ax1.scatter(df30['f_pv'], df30['mp_energy'], label='mp interpolated', color='red' )
    plt.xlabel('Proportion of solar')
    plt.ylabel('Energy')
    plt.title('Storage about 30 days mp and original paper Excel (wind annotated)')
    plt.legend(loc='lower left', fontsize=15)
    for i, point in df30.iterrows():
        ax1.text(point['f_pv'],point['kf_energy'],'{:.2f}'.format(point['f_wind']))
        ax1.text(point['f_pv'],point['mp_energy'],'{:.2f}'.format(point['f_wind']))
    plt.show()

