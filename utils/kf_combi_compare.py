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

# read mp data
mp_filename = "/home/malcolm/uclan/output/kf_copy/sharesKF.csv"
mp = pd.read_csv(mp_filename, header=0)

print(combi)
print(mp)

diffs = (combi['storage'] - mp['kf_storage']) / combi['storage'].mean()
print('Compare percent max diff {} average diff {}'.format(diffs.max(), diffs.mean() ) )

mrow = diffs.idxmax()
print(mrow)
print(mp.iloc[mrow])
print(combi.iloc[mrow])
