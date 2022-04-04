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
from icecream import ic 

# custom code
import stats
import readers
import storage

# scale to attempt to reproduce KF plot
def scalekf(df):
    kf_p = 0.116
    kf_w = 0.28
    df['Ps'] = df['Ps'] * ( (1 - df['Pw'] * kf_w) / kf_p )

#
# Functions to convert to and from actual capacity based on demandNNH.csv
def cf2gw(x):
    return x * generation_capacity
def gw2cf(x):
    return x / generation_capacity

def scatterHeat(df, variable, title, label, annotate=False):
    ax = df.plot.scatter(x='f_wind', y='f_pv', c=variable, colormap='viridis')
    plt.xlabel('Normalised Wind Capacity')
    plt.ylabel('Normalised PV Capacity')
    plt.title('{} for different proportions of wind and solar ({} ).'.format(title, label))
    if annotate:
        for i, point in df.iterrows():
            ax.text(point['f_wind'],point['f_pv'],'{:.1f}'.format(point['storage']))
    plt.show()

# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--pfit', action="store_true", dest="pfit", help='Show 2d plots', default=False)
parser.add_argument('--energy', action="store_true", dest="energy", help='Plot energy instead of capacity', default=False)
parser.add_argument('--rate', action="store_true", dest="rate", help='Plot the charge and discharge rates', default=False)
parser.add_argument('--min', action="store_true", dest="min", help='Plot the minimum generation line', default=False)
parser.add_argument('--kf', action="store_true", dest="kf", help='Scale PV axis as per KF', default=False)
parser.add_argument('--last', action="store_true", dest="last", help='Only consider solutions where store ended up full ie last=0', default=False)
parser.add_argument('--annotate', action="store_true", dest="annotate", help='Annotate the shares heat map', default=False)
parser.add_argument('--scenario', action="store", dest="scenario", help='Scenarion to plot', default='adhoc')
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
parser.add_argument('--adverse', action="store", dest="adverse", help='Adverse file mnemonic', default='5s1')
args = parser.parse_args()

# scenario files
#  Of the 3 chars:
#   1 - scenario P, F, H, B, N, E
#     ( from hydrogenVsHeatpumps.py )
#   2 - climate  N, C
#   3 - demand method S=synthetic, H=historic
#
hvh = 'hydrogenVpumps/'
y40 = '40years/'
ev40 = 'ev40years/'
kf = 'kf/'
kfev = 'kfev/'
kfig8 = 'kfig8/'
kfig6 = 'kfig6/'
mfig8 = 'mfig8/'
adv = 'adv/'
fixed = 'fixed/'
sm = 'smodel/'
fixeds = 'fixed_scaleKF/'
temp = 'temp/'
hp = 'heatpaper/'
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
#scenarios = {'HNS' : {'file': 'HNS', 'dir' : hvh, 'title': 'Half heat pumps, half hydrogen'}, 'PNS' : {'file': 'PNS', 'dir' : hvh, 'title': 'All heat pumps'}, 'FNS' : {'file': 'FNS', 'dir' : hvh, 'title': 'FES 2019 Net Zero: heat pumps, hydrogen and hybrid heat pumps'} }
if args.scenario == 'models':
    scenarios = {'HNSh' :
       {'file': 'PNS', 'dir' : sm, 'title': 'All heat pumps, mp storage model'},
                 'HNSy' : 
       {'file': 'PNS', 'dir' : y40, 'title': 'All heat pumps, kf storage model'}    }
if args.scenario == 'eheat':
    scenarios = {'NNS' :
       {'file': 'NNS', 'dir' : kf, 'title': '2018 with electricity for heating removed'},
                 'PNS' :
       {'file': 'PNS', 'dir' : kf, 'title': 'All heating is heat pumps'}
    }
if args.scenario == 'hp':
    scenarios = {'GNS' :
       {'file': 'GNS', 'dir' : hp, 'title': '41% Heat Pumps'},
                 'FNS' :
       {'file': 'FNS', 'dir' : hp, 'title': '13% Hybrid Heat pumps'}
    }
#scenarios = {'NNS' :
#   {'file': 'NNS', 'dir' : kf, 'title': 'No Added Electric Heating'},
#             'PNS' :
#   {'file': 'PNS', 'dir' : kfev, 'title': '100% heat pumps and evs'}
#}
# scenarios = {'NNS' : {'file': 'NNS', 'dir' : kf, 'title': 'No Added Electric Heating'} }
if args.scenario == 'mlmp':
    scenarios = {'MP' :
      {'file': 'ENH', 'dir' : kfig8, 'title': 'MP method with historic electric'},
                 'KF' :
      {'file': 'CM', 'dir' : kfig8, 'title': 'KF matlab data'}
    }
if args.scenario == 'mpml':
    scenarios = {
     'KF' : {'file': 'CM' , 'dir' : kfig8, 'title': 'KF matlab data'},
     'MP' : {'file': 'ENH', 'dir' : kfig8, 'title': 'MP method with historic electric'}
    }
if args.scenario == 'temp':
    scenarios = {'temp' : {'file': 'NNS', 'dir' : temp, 'title': 'Synthetic Electric with shift'} }
if args.scenario == 'kfcm':
    scenarios = {'KF' : {'file': 'CM', 'dir' : kfig8, 'title': 'Output from matlab code'} }
if args.scenario == 'kfmp':
    scenarios = {'MP' :
    {'file': 'NNH', 'dir' : kfig8, 'title': 'MP method with historic electric'},
              'KF' :
    {'file': 'KF', 'dir' : kfig8, 'title': 'KF data'}
    }
if args.scenario == 'fixeds':
    scenarios = {'NNS' : {'file': 'ENS', 'dir' : fixeds, 'title': 'Synthetic Electric Series'} }
if args.scenario == 'fixed':
    scenarios = {'NNS' : {'file': 'ENS', 'dir' : fixed, 'title': 'Synthetic Electric Series'} }
if args.scenario == 'kfig8':
    scenarios = {'NNH' : {'file': 'NNH', 'dir' : kfig8, 'title': 'Historic Electric Series'} }
if args.scenario == 'kfig6':
    scenarios = {'NNH' : {'file': 'NNH', 'dir' : kfig6, 'title': 'Historic Electric Series'} }
if args.scenario == 'adv':
    file23 = 'a{}NNS'.format(args.adverse)
    file4 = 'c{}NNS'.format(args.adverse)
    scenarios = {'NNSa' :
       {'file': file23, 'dir' : adv, 'title': 'ADV {} 2-3 deg warming (normal heat) '.format(args.adverse)},
             'NNSc' :
       {'file': file4, 'dir' : adv, 'title': 'ADV {} 4 deg warming (normal heat)'.format(args.adverse)}
    }
if args.scenario == 'advp':
    file23 = 'a{}PNS'.format(args.adverse)
    file4 = 'c{}PNS'.format(args.adverse)
    scenarios = {'NNSa' :
       {'file': file23, 'dir' : adv, 'title': 'ADV {} 2-3 degrees warming (all heat punps) '.format(args.adverse)},
             'NNSc' :
       {'file': file4, 'dir' : adv, 'title': 'ADV {} 4 degrees warming (all heat pumps)'.format(args.adverse)}
    }

output_dir = "/home/malcolm/uclan/output"

# load the demands
demands = {}
capacities = {}
print('number of annual  capacity')
print('days      energy  to supply load')
for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/demand{}.csv'.format(output_dir, folder, filename)
    demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
    demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
#   print(demand)
    demands[key] = demand
    ndays = len(demand)
    annual_energy = demand.sum() * 365 / ndays
#   capacity = annual_energy * 1000.0 / 8760
    capacity = demand.max() * 1000.0 / 24.0
    print('{}  {}  {:.2f}  {:.2f}'.format(key, ndays, annual_energy, capacity))
    capacities[key] = capacity

# Load the shares dfs

dfs={}
gen_cap={}
for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/shares{}.csv'.format(output_dir, folder, filename)
    df = pd.read_csv(path, header=0, index_col=0)
#   print(df)
    storage_values = df['storage'].values
    for i in range(storage_values.size):
        if storage_values[i] < 0.0:
            storage_values[i] = 0.0;
    df['storage'] = storage_values
    wind_at_1 = df[df['f_wind']==1.0]
    if len(wind_at_1.index) == 0 or 'gw_wind' not in wind_at_1.columns:
        gen_cap[key] = 30
    else:
        gen_cap[key] = wind_at_1['gw_wind'].values[0]
    if args.last:
       df = df[df['last']==0]
#   print(df['storage'])
    dfs[key] = df

# Plot storage heat maps

if args.plot:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        pdf = df[df['storage']<200]
        scatterHeat(pdf, 'storage', 'Storage in days ', label, args.annotate)

if args.plot:
    # Plot viable solutions
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
#       df['last'].clip(lower=0.0, inplace=True)
        scatterHeat(df, 'last', 'Store remaining in days ', label)

if args.pfit:
    # Plot viable solutions
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]

        pvwind3 = df[df['f_wind']==3.0]
#       print(pvwind3)
        plt.scatter(pvwind3['f_pv'], pvwind3['storage'], s=12)
        plt.title('Variation of PV and storage for wind=3.0')
        plt.xlabel('PV fraction')
        plt.ylabel('Storage (days)')
        plt.show()

        pvwind3 = df[(df['f_wind']==3.0) & (df['storage']<100) & (df['f_pv']>0.0)]
#       print(pvwind3)
        plt.scatter(pvwind3['f_pv'], pvwind3['storage'], s=12)
        plt.title('Variation of PV and storage for wind=3.0')
        plt.xlabel('PV fraction')
        plt.ylabel('Storage (days)')
        plt.show()

        windpv4 = df[df['f_pv']==4.0]
        plt.scatter(windpv4['f_wind'], windpv4['storage'], s=12)
        plt.title('Variation of Wind and storage for PV=4.0')
        plt.xlabel('Wind fraction')
        plt.ylabel('Storage (days)')
        plt.show()

        windpv4 = df[(df['f_pv']==4.0) & (df['storage']<100) & (df['f_wind']>0.0)]
        plt.scatter(windpv4['f_wind'], windpv4['storage'], s=12)
        plt.title('Variation of Wind and storage for PV=4.0')
        plt.xlabel('Wind fraction')
        plt.ylabel('Storage (days)')
        plt.show()

if args.rate:
    # Plot max charge rate. 
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'charge', 'Max charge rate in %peak', label)

    # Plot max discharge rate. 
    for filename, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'discharge', 'Max discharge rate in %peak', label )

# Plot constant storage lines

pws = {}
first = True
for key, scenario in scenarios.items():
    df = dfs[key].copy()
    filename = scenario['file']
    label = scenario['title']
    # get the generation capacity
    generation_capacity = gen_cap[key]
#   generation_capacity = 991948.1000000001 / 1e6
#   testx = wind2gw(2)
    print('generation_capacity {} '.format(generation_capacity))
    wind_parm = 'f_wind'
    pv_parm = 'f_pv'
    if args.energy:
        wind_parm = 'wind_energy'
        pv_parm = 'pv_energy'
#       generation_capacity = 1/generation_capacity
    # remove negative values of storage (excess)
    # calculate constant storage line for 40 days
    storage_40 = storage.storage_line(df,40.0, args.sline, wind_parm, pv_parm)
    if args.kf:
        scalekf(storage_40)
    # save axis for the first one, and plot
    if first:
        ax = storage_40.plot(x='Pw',y='Ps',label='storage 40 days. {}'.format(label))
    else:
        storage_40.plot(x='Pw',y='Ps',ax=ax,label='storage 40 days. {}'.format(label))

    # calcuate constant storage line for 25 days and plot
    storage_25 = storage.storage_line(df,25.0, args.sline, wind_parm, pv_parm)
    if args.kf:
        scalekf(storage_25)
    storage_25.plot(x='Pw',y='Ps',ax=ax,label='storage 25 days. {}'.format(label))

    # calcuate constant storage line for 60 days and plot
    storage_60 = storage.storage_line(df,60.0, args.sline, wind_parm, pv_parm)
    if args.kf:
        scalekf(storage_60)
    storage_60.plot(x='Pw',y='Ps',ax=ax,label='storage 60 days. {}'.format(label))

    # calcuate constant storage line for 30 days and plot
    storage_30 = storage.storage_line(df,30.0, args.sline, wind_parm, pv_parm)
    if args.kf:
        scalekf(storage_30)
    storage_30.plot(x='Pw',y='Ps',ax=ax,label='storage 30 days. {}'.format(label))
    # TODO if we are on the 2nd or greater scenario then compare the amount
    # of storage, pv and wind with the previous scenario
    if not first:
        storage_diff = df['storage'] - last_df['storage']
        print('Mean storage difference between {} {} and {} {} is {}'.format(key, df['storage'].mean(), last_key, last_df['storage'].mean(), storage_diff.mean() ) )
    last_df = df
    last_key = key
    first = False

    # store the pv and wind for the storage
    pws[key] = { 30 : {'wind': storage_30['Pw'].mean(), 'pv': storage_30['Ps'].mean() }, 
                 60 : {'wind': storage_60['Pw'].mean(), 'pv': storage_60['Ps'].mean() }, 
                 25 : {'wind': storage_25['Pw'].mean(), 'pv': storage_25['Ps'].mean() }, 
                 40 : {'wind': storage_60['Pw'].mean(), 'pv': storage_40['Ps'].mean() }
    }

plt.title('Constant storage lines for different scenarios')
if args.energy:
    plt.xlabel('Wind ( energy in proportion to nomarlised demand)')
    plt.ylabel('Solar PV ( energy in proportion to normalised demand)')
else:
    plt.xlabel('Wind ( capacity in proportion to nomarlised demand)')
    plt.ylabel('Solar PV ( capacity in proportion to normalised demand)')
# 2nd axis
if not args.energy:
    axx = ax.secondary_xaxis('top', functions=(cf2gw, gw2cf))
    axx.set_xlabel('Capacity GW')
    axy = ax.secondary_yaxis('right', functions=(cf2gw, gw2cf))
    axy.set_ylabel('Capacity GW')


plt.show()

# report PV and Wind
# TODO this probably doesn't make any sense as we need varying PV for
# a fixed storage and wind.  ????
keys = scenarios.keys()
print('Storage  PV  {}'.format(' '.join(keys) ) )
for storage in [25, 30, 40, 60]:
    pvs = [pws[key][storage]['pv'] for key in keys]
    print('{}         '.format(storage), ' '.join([' {:.2f} '.format(i) for i in pvs]) )
    
print('Storage  Wind  {}'.format(' '.join(keys) ) )
for storage in [25, 30, 40, 60]:
    pvs = [pws[key][storage]['wind'] for key in keys]
    print('{}         '.format(storage), ' '.join([' {:.2f} '.format(i) for i in pvs]) )
    

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
    label = scenario['title']
    demand = demands[key]
#   print(demand)

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
#   print(demand)

    demand.plot(label='Hydrogen Demand {}'.format(label) )

plt.title('Daily Hydrogen demand')
plt.xlabel('year', fontsize=15)
plt.ylabel('Demand (MWh)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
plt.show()
