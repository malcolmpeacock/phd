# python script to prove that KF renewable generation
# scales with different SF, CW

# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

# custom code
import stats
import readers

# kf generation different proportions

# CW CS 1

wind_filename = '/home/malcolm/uclan/data/kf/wind.txt'
kf_wind = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv.txt'
kf_pv = pd.read_csv(pv_filename, header=None, squeeze=True)

we1 = kf_wind.mean() * 1e-9
pe1 = kf_pv.mean() * 1e-9

# CW CS 0.2

wind_filename = '/home/malcolm/uclan/data/kf/wind02.txt'
kf_wind02 = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv02.txt'
kf_pv02 = pd.read_csv(pv_filename, header=None, squeeze=True)

we02 = kf_wind02.mean() * 5 * 1e-9
pe02 = kf_pv02.mean() * 5 * 1e-9

# CW CS 0.5

wind_filename = '/home/malcolm/uclan/data/kf/wind05.txt'
kf_wind05 = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pv05.txt'
kf_pv05 = pd.read_csv(pv_filename, header=None, squeeze=True)

we05 = kf_wind05.mean() * 2 * 1e-9
pe05 = kf_pv05.mean() * 2 * 1e-9

# CW CS SF 1.7

wind_filename = '/home/malcolm/uclan/data/kf/windsf17.txt'
kf_wind17 = pd.read_csv(wind_filename, header=None, squeeze=True)
pv_filename = '/home/malcolm/uclan/data/kf/pvsf17.txt'
kf_pv17 = pd.read_csv(pv_filename, header=None, squeeze=True)

we17 = kf_wind17.mean() * 1e-9 / 1.7
pe17 = kf_pv17.mean() * 1e-9 / 1.7

print('Energy SF 1.7 {:.2f} CW 1 {:.2f} CW 0.5 {:.2f} CW 0.2 {:.2f} '.format(we17, we1, we05, we02))
print('Energy SF 1.7 {:.2f} CS 1 {:.2f} CS 0.5 {:.2f} CS 0.2 {:.2f} '.format(pe17, pe1, pe05, pe02))
