# python script to extract days of similar storage.

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
parser = argparse.ArgumentParser(description='Extract days of storage')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--source', action="store", dest="source", help='Input file to Use ', default='combi')
parser.add_argument('--eta', action="store", dest="eta", help='Efficiency ', default='75')
args = parser.parse_args()

# capacity factors
kf_wcf = 0.28
kf_pcf = 0.1156
ndays = 10958
total_demand = 9169197139968000.0
print('Daily energy {}'.format(total_demand / ndays) )

if args.source == 'combi':

    filename = '/home/malcolm/uclan/data/kf/Combi.csv'
    data = pd.read_csv(filename, header=0, squeeze=True)
    print(data)
    cw = data['cw']
    cs = 1- cw
    data['f_wind'] = data['sf'] * cw / kf_wcf
    data['f_pv'] = data['sf'] * cs / kf_pcf
    data['days'] = data['storage'] * ndays / 100.0

if args.source == 'recreate':
    filename = '/home/malcolm/uclan/output/kf_recreate/sharesKFS{}.csv'.format(args.eta)
    data = pd.read_csv(filename, header=0, squeeze=True, index_col=0)
    data['days'] = data['storage']
if args.source == 'copy':
    filename = '/home/malcolm/uclan/output/kf_copy/sharesKFS{}.csv'.format(args.eta)
    data = pd.read_csv(filename, header=0, squeeze=True, index_col=0)
    data['days'] = data['storage']

print(data)

capacities = [25,30,40,60]
for days in capacities:
    upper = days + (days * 0.5 / 100.0)
    lower = days - (days * 0.5 / 100.0)
    line_data = data[(data['days']<upper) & (data['days']>lower)]
    print(line_data)
    output_filename = '/home/malcolm/uclan/output/kf_{}/S{}days{:02d}.csv'.format(args.source, args.eta,days)
    print(output_filename)
    line_data.to_csv(output_filename, float_format='%g')
