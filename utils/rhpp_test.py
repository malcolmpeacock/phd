# python script to:
#   - test out RHpp data

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob
import os.path

# custom code
import stats
import readers

# main program
rhpp_dir = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/csv/'
rhpp_filename = '/home/malcolm/uclan/data/rhpp-heatpump/UKDA-8151-csv/csv/processed_rhpp5570.csv'

max_start = datetime(2012, 1, 1)
min_end   = datetime(2015, 12, 30)
nans_zero = []
data = { 'location' : [], 'start' : [], 'end' : [], 'nans_Hhp': [], 'nans_Ehp': [], 'nans_tin': [], 'nans_tsf': []}
# for each file ....
for name in glob.glob(rhpp_dir + 'processed*'):
    filename = os.path.basename(name)
    print(filename)
    location = filename[14:18]
    print(location)

    # read heat demand and temp
    df, stats = readers.read_rhpp(name, location)
#   print(df)
#   print(stats)
    # store the stats for the location
    for key in data:
        data[key].append(stats[key])
    # count how many have no nans
    if stats['nans_Hhp'] == 0 and stats['nans_Ehp'] == 0 and stats['nans_tin'] == 0 and stats['nans_tsf'] == 0:
        nans_zero.append(location)
    # work out the latest start and earliest finish
    if stats['start'] > max_start:
        max_start = stats['start']
    if stats['end'] < min_end:
        min_end = stats['end']

print(nans_zero)
print(max_start, min_end)
df_stats = pd.DataFrame(data)

output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
df_stats.to_csv(output_dir + 'stats.csv')
