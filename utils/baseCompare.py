# Compare and plot the impact of different amounts of base load and
# wind and solar shares and storage
# using 40 years weather

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy.stats import wasserstein_distance
from icecream import ic 

# custom code
import stats
import readers
import storage

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

# scenario files
#scenarios = {'HNS' : 'Half Heat Pumps',
#             'NNS' : 'No   Heat Pumps'
#            }
scenarios = {'NNH' : 'Scaled Historic Time Series',
             'ENS' : 'Synthetic Time Series From Weather'
            }
#scenarios = {'PNH' : 'Scaled Historic Time Series + heat',
#            'PNS' : 'Synthetic Time Series From Weather + heat'
#           }
filename = 'sharesPNS.csv'
output_dir = "/home/malcolm/uclan/output/40years/"
path = output_dir + filename
dfb = pd.read_csv(path, header=0, index_col=0)
print(dfb)
base_values = np.sort(dfb['base'].unique())
print(base_values)
for val in base_values:
    df = dfb[dfb['base'] == val]
    print(df)

    if args.plot:
        dfb.plot.scatter(x='f_wind', y='f_pv', c='storage', colormap='viridis')
        plt.xlabel('Proportion of wind')
        plt.ylabel('Porportion of solar')
        plt.title('Storage in days for different proportions of wind and solar. Base load {}.'.format(val))
        plt.show()

first = True
for val in base_values:
    df = dfb[dfb['base'] == val]
    # calculate constant storage line for 40 days
    storage_40 = storage.storage_line(df,40.0)
    # save axis for the first one, and plot
    if first:
        first = False
        ax = storage_40.plot(x='Pw',y='Ps',label='storage 40 days. Base {}'.format(val))
    else:
        storage_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days. Base {}'.format(val))

    # calcuate constant storage line for 25 days and plot
#   storage_25 = storage.storage_line(df,25.0)
#   storage_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days. Base {}'.format(val))

    # calcuate constant storage line for 2 days and plot
#   storage5_2 = storage.storage_line(df,2.0)
#   storage5_2.plot(x='Pw',y='Ps',ax=ax,label='storage 2 days. Base {}'.format(val))


plt.title('Constant storage lines for different proportions of base load')
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()

