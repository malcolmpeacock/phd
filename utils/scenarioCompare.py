# Compare and plot different scenarios of wind and solar shares and storage
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

output_dir = "/home/malcolm/uclan/output/40years"
dfs={}

first = True
for filename, label in scenarios.items():
    path = '{}/shares{}.csv'.format(output_dir, filename)
    df = pd.read_csv(path, header=0, index_col=0)
    print(df)
    dfs[filename] = df

    if args.plot:
        df.plot.scatter(x='f_wind', y='f_pv', c='storage', colormap='viridis')
#       plt.colorbar(label='Storage (days)')
        plt.xlabel('Proportion of wind')
        plt.ylabel('Porportion of solar')
        plt.title('Storage in days for different proportions of wind and solar ({} ).'.format(label))
        plt.show()

first = True
for filename, label in scenarios.items():
    df = dfs[filename]
    # calculate constant storage line for 40 days
    storage_40 = storage.storage_line(df,40.0)
    # save axis for the first one, and plot
    if first:
        first = False
        ax = storage_40.plot(x='Pw',y='Ps',label='storage 40 days. {}'.format(label))
    else:
        storage_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days. {}'.format(label))

    # calcuate constant storage line for 25 days and plot
    storage_25 = storage.storage_line(df,25.0)
    storage_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days. {}'.format(label))

    # calcuate constant storage line for 2 days and plot
    storage5_2 = storage.storage_line(df,2.0)
    storage5_2.plot(x='Pw',y='Ps',ax=ax,label='storage 2 days. {}'.format(label))


plt.title('Constant storage lines for different scenarios}')
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()

# compare the yearly files

first = True
for filename, label in scenarios.items():
    path = '{}/yearly{}.csv'.format(output_dir, filename)
    df = pd.read_csv(path, header=0, index_col=0)


    df['storage'].plot(label='Yearly Storage {}'.format(label) )

plt.title('Difference on yearly term storage')
plt.xlabel('year', fontsize=15)
plt.ylabel('Days of storage', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()

# plot the demand

first = True
for filename, label in scenarios.items():
    path = '{}/demand{}.csv'.format(output_dir, filename)
    demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
#   demand = pd.read_csv(path, header=0, index_col=0, squeeze=True, parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y-%d-%m'))
    demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
    print(demand)

    demand.plot(label='Electricity Demand {}'.format(label) )

plt.title('Daily Electricity demand')
plt.xlabel('year', fontsize=15)
plt.ylabel('Demand (MWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()
