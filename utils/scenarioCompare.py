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

def scatterHeat(df, variable, title, threshold=200):
    pdf = df[df[variable]<threshold]
    pdf.plot.scatter(x='f_wind', y='f_pv', c=variable, colormap='viridis')
    plt.xlabel('Proportion of wind')
    plt.ylabel('Porportion of solar')
    plt.title('{} for different proportions of wind and solar ({} ).'.format(title, label))
    plt.show()

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--rate', action="store_true", dest="rate", help='Plot the charge and discharge rates', default=False)
args = parser.parse_args()

# scenario files
#  Of the 3 chars:
#   1 - scenario P, F, H, G, N, E
#   2 - climate  N, C
#   3 - demand method S=synthetic, H=historic
#
hvh = 'hydrogenVpumps/'
y40 = '40years/'
#
#scenarios = {'HNS' : 'Half Heat Pumps',
#             'NNS' : 'No   Heat Pumps'
#            }
#scenarios = {'NNH' : 'Scaled Historic Time Series',
#             'ENS' : 'Synthetic Time Series From Weather'
#            }
#scenarios = {'PNH' : 'Scaled Historic Time Series + heat',
#            'PNS' : 'Synthetic Time Series From Weather + heat'
#           }
scenarios = {'HNS' : {'file': 'HNS', 'dir' : hvh, 'title': 'Half heat pumps, half hydrogen'}, 'PNS' : {'file': 'PNS', 'dir' : hvh, 'title': 'All heat pumps'}, 'FNS' : {'file': 'FNS', 'dir' : hvh, 'title': 'FES 2019 Net Zero: heat pumps, hydrogen and hybrid heat pumps'} }
#scenarios = {'HNSh' : {'file': 'HNS', 'dir' : hvh, 'title': 'Half heat pumps, gen hydrogen'}, 'HNSy' : {'file': 'HNS', 'dir' : y40, 'title': 'Half heat pumps, electric only'} }

output_dir = "/home/malcolm/uclan/output"

# Load the shares dfs

dfs={}
for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/shares{}.csv'.format(output_dir, folder, filename)
    df = pd.read_csv(path, header=0, index_col=0)
    print(df)
    dfs[key] = df

# Plot storage heat maps

if args.plot:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'storage', 'Storage in days ')

if args.plot:
    # Plot viable solutions
    for filename, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'last', 'Store remaining in days ', 0.0)

if args.rate:
    # Plot max charge rate. 
    for filename, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'charge', 'Max charge rate in %peak')

    # Plot max discharge rate. 
    for filename, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'discharge', 'Max discharge rate in %peak')

# Plot constant storage lines

first = True
for key, scenario in scenarios.items():
    df = dfs[key]
    filename = scenario['file']
    label = scenario['title']
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


plt.title('Constant storage lines for different scenarios')
plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
plt.show()

# compare the yearly files

for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/yearly{}.csv'.format(output_dir, folder, filename)
    df = pd.read_csv(path, header=0, index_col=0)

    df['storage'].plot(label='Yearly Storage {}'.format(label) )

plt.title('Difference on yearly term storage')
plt.xlabel('year', fontsize=15)
plt.ylabel('Days of storage', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()

# plot the electricity demand

for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/demand{}.csv'.format(output_dir, folder, filename)
    demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
    demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
    print(demand)

    demand.plot(label='Electricity Demand {}'.format(label) )

plt.title('Daily Electricity demand')
plt.xlabel('year', fontsize=15)
plt.ylabel('Demand (MWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()

# plot the hydrogen demand

for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/hydrogen{}.csv'.format(output_dir, folder, filename)
    demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
    demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
    print(demand)

    demand.plot(label='Hydrogen Demand {}'.format(label) )

plt.title('Daily Hydrogen demand')
plt.xlabel('year', fontsize=15)
plt.ylabel('Demand (MWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()
