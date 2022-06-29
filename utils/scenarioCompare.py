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
from os.path import exists

# custom code
import stats
import readers
import storage

def print_titles(sl=11):
    print('Scenario'.ljust(sl), 'Point             Wind  PV    Fraction-C/E NP Energy Storage Cost')

def print_min(point, point_title, scenario_title, sl=11):
    capacity_wind_fraction = point['wind'] / ( point['wind'] + point['pv'] )
    energy_wind_fraction = point['fraction']
    print('{} {}  {:.2f}  {:.2f}  {:.2f} {:.2f}    {}  {:.2f}   {:05.2f}   {:.2f}'.format(scenario_title.ljust(sl), point_title,  point['wind'], point['pv'], capacity_wind_fraction, energy_wind_fraction, point['np'], point['energy'], point['days'], point['cost'], capacity_wind_fraction) )
    output = point
    output['fraction'] = capacity_wind_fraction
    output['scenario'] = scenario_title
    output['point'] = point_title
    return output

def get_storage_line(df, storage_model, days, wind_parm='f_wind', pv_parm='f_pv', variable='storage'):
    if storage_model == 'new':
        storage_line = df[df[variable] == days].copy()
        storage_line.rename(columns={'f_pv': 'Ps', 'f_wind': 'Pw'}, inplace=True)
        storage_line = storage_line.sort_values(['Pw', 'Ps'], ascending=[True, True])
    else:
        storage_line = storage.storage_line(df, days, args.sline, wind_parm, pv_parm, variable)
    storage_line['energy'] = storage_line['wind_energy'] + storage_line['pv_energy']
    storage_line['fraction'] = storage_line['wind_energy'] / storage_line['energy']
    return storage_line

def get_viable(df, last, days):
    if args.last == 'full':
        pdf = df[df['last']==0.0]
    else:
        if args.last == 'p3':
            last_val = days * 0.03
            pdf = df[df['last']>-last_val]
        else:
            pdf = df
    return pdf

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

def min_gen_line(ax, days, marker):
    Lw = 0.38
    Ls = 0.1085
    # minimum generation
    Ps = []
    Pw = []

    # y intercept
    Pw.append(0.0)
    Ps.append( days / Ls )
    # x intercept
    Ps.append(0.0)
    Pw.append( days / Lw )

    min_energy_line = { 'Pw' : Pw, 'Ps' : Ps }
    df_min = pd.DataFrame(data=min_energy_line)
    df_min.plot(x='Pw', y='Ps', ax=ax, label='theorectical minimum generation from mean cf', marker=marker)


# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--pstore', action="store_true", dest="pstore", help='Plot the sample store history ', default=False)
parser.add_argument('--pdemand', action="store_true", dest="pdemand", help='Plot the demand ', default=False)
parser.add_argument('--pfit', action="store_true", dest="pfit", help='Show 2d plots', default=False)
parser.add_argument('--yearly', action="store_true", dest="yearly", help='Show Yearly plots', default=False)
parser.add_argument('--energy', action="store_true", dest="energy", help='Plot energy instead of capacity', default=False)
parser.add_argument('--rate', action="store_true", dest="rate", help='Plot the charge and discharge rates', default=False)
parser.add_argument('--min', action="store_true", dest="min", help='Plot the minimum generation line', default=False)
parser.add_argument('--annotate', action="store_true", dest="annotate", help='Annotate the shares heat map', default=False)
parser.add_argument('--scenario', action="store", dest="scenario", help='Scenarion to plot', default='adhoc')
parser.add_argument('--days', action="store", dest="days", help='Days of storage line to plot', default='0.5, 1, 3, 10, 25, 30, 40, 60' )
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1')
parser.add_argument('--svariable', action="store", dest="svariable", help='Variable to contour, default is storage', default='storage')
parser.add_argument('--adverse', action="store", dest="adverse", help='Adverse file mnemonic', default='5s1')
parser.add_argument('--last', action="store", dest="last", help='Only include configs which ended with store: any, full, p3=3 percent full ', default='p3')
parser.add_argument('--excess', action="store", dest="excess", help='Excess value to find minimum storage against', type=float, default=0.5)
parser.add_argument('--variable', action="store", dest="variable", help='Variable to plot from scenario', default=None)
args = parser.parse_args()

day_str = args.days.split(',')
day_list=[]
for d in day_str:
    day_list.append(float(d))


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
ninja85 = 'ninja85/'
ninja75 = 'ninja75/'
scenario_title = ' for different scenarios'
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
if args.scenario == 'historic':
    scenarios = {'historic' :
       {'file': 'ENH', 'dir' : 'demand/', 'title': 'Scaled Historic Time Series'},
                 'synthetic' : 
       {'file': 'ENS', 'dir' : 'demand/', 'title': 'Synthetic Time Series using 2018 Baseline'} 
    }
if args.scenario == 'ev':
    scenarios = {'noev' :
       {'file': 'ENS', 'dir' : 'ev40years/noev/', 'title': 'No Evs'},
                 'ev' : 
       {'file': 'ENS', 'dir' : 'ev40years/ev/', 'title': 'Evs '} 
    }
if args.scenario == 'hourly':
    scenarios = {'hourly' :
       {'file': 'ENS', 'dir' : 'hourly/ninjaS80/', 'title': 'Hourly Time Series'},
                 'daily' : 
       {'file': 'ENS', 'dir' : 'daily/ninjaS80/', 'title': 'Daily Time Series '} 
    }
if args.scenario == 'hydrogen30':
    scenarios = {'50' :
       {'file': 'ENS', 'dir' : 'hydrogen/hbase04/', 'title': 'Base load 0.4 efficiency 50%'},
                 '30' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase04eta30/', 'title': 'Base load 0.4 efficiency 30%'} 
    }
if args.scenario == 'hydrogenfes':
    scenarios = {'he' :
       {'file': 'ENS', 'dir' : 'hydrogen/gbase04/', 'title': 'Base load 0.4 existing heating'},
                 'hfes' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase04/', 'title': 'Base load 0.4 electrified heat FES Net Zero'} 
    }
if args.scenario == 'hydrogenev':
    scenarios = {'he' :
       {'file': 'ENS', 'dir' : 'hydrogen/hbase04/', 'title': 'Base load 0.4 existing heating'},
                 'hev' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase04ev/', 'title': 'Base load 0.4 electric vehicles'} 
    }
if args.scenario == 'hydrogenClimate':
    scenarios = {'he' :
       {'file': 'FNS', 'dir' : 'hydrogen/hbase04/', 'title': 'Base load 0.4 electirified heating FES Net Zero'},
                 'hec' : 
       {'file': 'FCS', 'dir' : 'hydrogen/hbase04/', 'title': 'Base load 0.4 electrified heating FES Net Zero. Climate Change Correction'} 
    }
if args.scenario == 'capacity':
    scenarios = {'c1' :
       {'file': 'ENS', 'dir' : 'capacity/c50/', 'title': 'Store starts at 50%'},
                 'c2' : 
       {'file': 'ENS', 'dir' : 'capacity/c60/', 'title': 'Store starts at 60%'} 
    }
if args.scenario == 'all_model':
    scenarios = {'old' :
       {'file': 'FNS', 'dir' : 'all_model/old/', 'title': 'New contours model'},
                 'new' : 
       {'file': 'FNS', 'dir' : 'all_model/new/', 'title': 'New all grid model'}
    }
if args.scenario == 'baseload_all':
    scenarios = {'0.0' :
       {'file': 'FNS', 'dir' : 'baseload_all/b00/', 'title': 'Base load 0.0'},
                 '0.05' : 
       {'file': 'FNS', 'dir' : 'baseload_all/b05/', 'title': 'Base load 0.05'},
                 '0.1' : 
       {'file': 'FNS', 'dir' : 'baseload_all/b10/', 'title': 'Base load 0.10'},
                 '0.15' : 
       {'file': 'FNS', 'dir' : 'baseload_all/b15/', 'title': 'Base load 0.15'},
                 '0.2' : 
       {'file': 'FNS', 'dir' : 'baseload_all/b15/', 'title': 'Base load 0.20'},
                 '0.25' : 
       {'file': 'FNS', 'dir' : 'baseload_all/b20/', 'title': 'Base load 0.25'}
    }
if args.scenario == 'baseload':
    scenarios = {'00' :
       {'file': 'FNS', 'dir' : 'baseload/b00/', 'title': 'Base load 0.0'},
                 '05' : 
       {'file': 'FNS', 'dir' : 'baseload/b05/', 'title': 'Base load 0.05'},
                 '10' : 
       {'file': 'FNS', 'dir' : 'baseload/b10/', 'title': 'Base load 0.10'},
                 '15' : 
       {'file': 'FNS', 'dir' : 'baseload/b15/', 'title': 'Base load 0.15'},
                 '20' : 
       {'file': 'FNS', 'dir' : 'baseload/b20/', 'title': 'Base load 0.20'},
                 '25' : 
       {'file': 'FNS', 'dir' : 'baseload/b25/', 'title': 'Base load 0.25'} }
if args.scenario == 'hydrogen':
    scenarios = {'b00' :
       {'file': 'ENS', 'dir' : 'hydrogen/hbase00/', 'title': 'Hydrogen 50% Base load 0.0'},
                 'b10' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase01/', 'title': 'Hydrogen 50% Base load 0.10'},
                 'b20' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase02/', 'title': 'Hydrogen 50% Base load 0.20'},
                 'b30' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase03/', 'title': 'Hydrogen 50% Base load 0.30'},
                 'b40' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase04/', 'title': 'Hydrogen 50% Base load 0.40'},
                 'b50' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase05/', 'title': 'Hydrogen 50% Base load 0.50'},
                 'b60' : 
       {'file': 'ENS', 'dir' : 'hydrogen/hbase06/', 'title': 'Hydrogen 50% Base load 0.60'} }
if args.scenario == 'today':
    scenarios = {'var12base020' :
       {'file': 'ENS', 'dir' : 'today/var12base020/', 'title': 'Base 020 Variable generation 1.2'},
                 'var11base020' : 
       {'file': 'ENS', 'dir' : 'today/var11base020/', 'title': 'Base 020 Variable generation 1.1'},
                 'var10base020' : 
       {'file': 'ENS', 'dir' : 'today/var10base020/', 'title': 'Base 020 Variable generation 1.0'},
                 'var09base020' : 
       {'file': 'ENS', 'dir' : 'today/var09base020/', 'title': 'Base 020 Variable generation 0.9'},
                 'var08base020' : 
       {'file': 'ENS', 'dir' : 'today/var08base020/', 'title': 'Base 020 Variable generation 0.8'},
                 'var07base020' : 
       {'file': 'ENS', 'dir' : 'today/var07base020/', 'title': 'Base 020 Variable generation 0.7'} 
    }
if args.scenario == 'basetoday':
    scenarios = {'var00base020' :
       {'file': 'ENS', 'dir' : 'today/var00base020/', 'title': 'Base 020 Variable generation 0.0'},
                 'var00base030' : 
       {'file': 'ENS', 'dir' : 'today/var00base030/', 'title': 'Base 030 Variable generation 0.0'},
                 'var00base040' : 
       {'file': 'ENS', 'dir' : 'today/var00base040/', 'title': 'Base 040 Variable generation 0.0'},
                 'var00base050' : 
       {'file': 'ENS', 'dir' : 'today/var00base050/', 'title': 'Base 050 Variable generation 0.0'},
                 'var00base060' : 
       {'file': 'ENS', 'dir' : 'today/var00base060/', 'title': 'Base 060 Variable generation 0.0'} 
    }
if args.scenario == 'variable':
    scenarios = {'b00' :
       {'file': 'FNS', 'dir' : 'variable/b00/', 'title': 'Variable generation 0.0'},
#                'b05' : 
#      {'file': 'FNS', 'dir' : 'variable/b05/', 'title': 'Base load 0.05'},
#                'b10' : 
#      {'file': 'FNS', 'dir' : 'variable/b10/', 'title': 'Base load 0.10'},
#                'b15' : 
#      {'file': 'FNS', 'dir' : 'variable/b15/', 'title': 'Base load 0.15'},
#                'b20' : 
#      {'file': 'FNS', 'dir' : 'variable/b20/', 'title': 'Base load 0.20'},
                 'b25' : 
       {'file': 'FNS', 'dir' : 'variable/b25/', 'title': 'Variable generation 0.25'} }
if args.scenario == 'capacities':
    scenarios = {'c30' :
       {'file': 'ENS', 'dir' : 'capacity/c30/', 'title': 'Store starts at 30%'},
                 'c40' : 
       {'file': 'ENS', 'dir' : 'capacity/c40/', 'title': 'Store starts at 40%'},
                 'c50' : 
       {'file': 'ENS', 'dir' : 'capacity/c50/', 'title': 'Store starts at 50%'},
                 'c60' : 
       {'file': 'ENS', 'dir' : 'capacity/c60/', 'title': 'Store starts at 60%'},
                 'c70' : 
       {'file': 'ENS', 'dir' : 'capacity/c70/', 'title': 'Store starts at 70%'},
                 'c80' : 
       {'file': 'ENS', 'dir' : 'capacity/c80/', 'title': 'Store starts at 80%'},
                 'c90' : 
       {'file': 'ENS', 'dir' : 'capacity/c90/', 'title': 'Store starts at 90%'}    }
if args.scenario == 'new4':
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'temp/', 'title': 'Existing heating 0.75 Old model'},
                 'half' : 
       {'file': 'ENS', 'dir' : 'new_model/end_half/', 'title': 'Existing heating 0.75 New model - half'},
                 'old2' : 
       {'file': 'ENS', 'dir' : 'new_model/old/', 'title': 'Existing heating 0.75 New model - old constraints'},
                 'any' : 
       {'file': 'ENS', 'dir' : 'new_model/end_any/', 'title': 'Existing heating 0.75 new model - any'}    }
if args.scenario == 'newdecade':
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'decade1/', 'title': 'Existing heating 0.75 old model 1980 - 1989'},
                 'new' : 
       {'file': 'ENS', 'dir' : 'new_model/decade1/', 'title': 'Existing heating 0.75 new model 1980 - 1989'}    }
if args.scenario == 'new':
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'temp/', 'title': 'Existing heating 0.75 Old model'},
                 'new' : 
       {'file': 'ENS', 'dir' : 'new_model/end_half/', 'title': 'Existing heating 0.75 new model'}    }
if args.scenario == 'newold':
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'temp/', 'title': 'Existing heating 0.75 Old model'},
                 'new' : 
       {'file': 'ENS', 'dir' : 'new_model/old/', 'title': 'Existing heating 0.75 new model with old constraints'}    }
if args.scenario == 'newfig8':
    scenarios = {'old' :
       {'file': 'NNH', 'dir' : 'old_fig8S75/', 'title': 'As per Fragaki et. al. Old storage model'},
                 'new' : 
       {'file': 'NNH', 'dir' : 'new_fig8S75/', 'title': 'As per Fragaki et. al. New storage model with old constraints'}    }
if args.scenario == 'newallS85':
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'allS85/', 'title': 'Ninja, synthetic 0.85 Old storage model'},
                 'new' : 
       {'file': 'ENS', 'dir' : 'new_allS85/', 'title': 'Ninja, synthetic 0.85 New storage model with old constraints'}    }
if args.scenario == 'halffull':
    scenarios = {'half' :
       {'file': 'ENS', 'dir' : 'new_allS85h/', 'title': 'Ninja, synthetic 0.75 New storage model half full'},
                 'full' : 
       {'file': 'ENS', 'dir' : 'new_allS85/', 'title': 'Ninja, synthetic 0.75 New storage model with old constraints'}    }
if args.scenario == 'halfhp':
    scenarios = {'allS75' :
       {'file': 'ENS', 'dir' : 'allS75/', 'title': 'Existing heating 0.75'},
                 'allS85' : 
       {'file': 'HNS', 'dir' : 'allS75/', 'title': 'Half Heat Pumps 0.75'}    }
if args.scenario == 'mfig8':
    scenarios = {'allS75' :
       {'file': 'NNS', 'dir' : 'allS75/', 'title': 'Existing heating 0.75'},
                 'allS85' : 
       {'file': 'NNS', 'dir' : 'allS85/', 'title': 'Existing heating 0.85'}    }
if args.scenario == 'years':
    scenarios = {'y4' :
       {'file': 'ENS', 'dir' : 'fouryears/y20092012', 'title': 'Existing heating 2009 - 2012'},
                 'y40' : 
       {'file': 'ENS', 'dir' : 'hydrogen/gbase04/', 'title': 'Existing heating 1980 - 2019'}    }
if args.scenario == 'yearsf':
    scenarios = {'y4' :
       {'file': 'FNS', 'dir' : 'fouryears/y20092012', 'title': 'Existing heating 2009 - 2012'},
                 'y40' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase04/', 'title': 'Existing heating 1980 - 2019'}    }
if args.scenario == 'decades':
    scenarios = {'decade1' :
       {'file': 'ENS', 'dir' : 'decade1/', 'title': 'Existing heating 1980 - 1989'},
                 'decade4' : 
       {'file': 'ENS', 'dir' : 'decade4/', 'title': 'Existing heating 2010 - 2019'}    }
if args.scenario == 'decades4':
    scenarios = {'decade1' :
       {'file': 'ENS', 'dir' : 'decade1/', 'title': 'Existing heating 1980 - 1989'},
                 'decade2' : 
       {'file': 'ENS', 'dir' : 'decade2/', 'title': 'Existing heating 1990 - 1999'},
                 'decade3' : 
       {'file': 'ENS', 'dir' : 'decade3/', 'title': 'Existing heating 2000 - 2019'},
                 'decade4' : 
       {'file': 'ENS', 'dir' : 'decade4/', 'title': 'Existing heating 2010 - 2019'}    }
if args.scenario == 'decadesnew':
    scenarios = {'decade1' :
       {'file': 'ENS', 'dir' : 'new_model/decade1/', 'title': 'Existing heating 1980 - 1989'},
                 'decade2' : 
       {'file': 'ENS', 'dir' : 'new_model/decade2/', 'title': 'Existing heating 1990 - 1999'},
                 'decade3' : 
       {'file': 'ENS', 'dir' : 'new_model/decade3/', 'title': 'Existing heating 2000 - 2019'},
                 'decade4' : 
       {'file': 'ENS', 'dir' : 'new_model/decade4/', 'title': 'Existing heating 2010 - 2019'}    }
if args.scenario == 'historic':
    scenarios = {'historic' :
       {'file': 'ENH', 'dir' : ninja85, 'title': 'Historic time series'},
                 'synthetic' : 
       {'file': 'ENS', 'dir' : ninja85, 'title': 'Synthetic time series'}    }
if args.scenario == 'generation':
    scenarios = {'ninja' :
       {'file': 'ENH', 'dir' : 'ninjaOnshore/', 'title': 'Generation from Renewables Ninja (onshore)'},
                 'kf' : 
       {'file': 'ENH', 'dir' : 'kfgen/', 'title': 'Generation from Fragaki et. al. '} }
if args.scenario == 'shore':
    scenarios = {'kf' :
       {'file': 'ENH', 'dir' : 'ninjaOffshore/', 'title': 'Ninja (offshore)'},
                 'ninja' : 
       {'file': 'ENH', 'dir' : 'ninjaOnshore/', 'title': 'Ninja (onshore)'}    }
if args.scenario == 'models':
    scenarios = {'HNSh' :
       {'file': 'PNS', 'dir' : sm, 'title': 'All heat pumps, mp storage model'},
                 'HNSy' : 
       {'file': 'PNS', 'dir' : y40, 'title': 'All heat pumps, kf storage model'}    }
if args.scenario == 'eheat':
    scenarios = {'PNS' :
       {'file': 'PNS', 'dir' : kf, 'title': 'All heating is provided by heat pumps'},
                 'NNS' :
       {'file': 'NNS', 'dir' : kf, 'title': '2018 with electricity for heating removed'}
    }
if args.scenario == 'eheat2':
    scenarios = {'BBB' :
       {'file': 'GNS', 'dir' : kf, 'title': '41% heating is provided by heat pumps'},
                 'AAA' :
       {'file': 'ENS', 'dir' : kf, 'title': '2018 with existing heating electricity'}
    }
if args.scenario == 'hp':
    scenarios = {'GNS' :
       {'file': 'GNS', 'dir' : hp, 'title': 'With 41% of heating supplied by Heat Pumps'},
                 'FNS' :
       {'file': 'FNS', 'dir' : hp, 'title': '13% Hybrid Heat pumps'}
    }
if args.scenario == 'hp2':
    scenarios = {'GNS' :
       {'file': 'GNS', 'dir' : hp, 'title': 'With 41% heating supplied by heat pumps'},
                 'ENS' :
       {'file': 'ENS', 'dir' : hp, 'title': 'With heating electricity at 2018 levels'}
    }
if args.scenario == 'hp1':
    scenarios = {'GNS' :
       {'file': 'GNS', 'dir' : hp, 'title': '41% Heat Pumps'},
    }
if args.scenario == 'zero':
    scenario_title = ' baseload 0.4 maximum and minimum storage'
    scenarios = {'FNS' :
       {'file': 'FNS', 'dir' : 'fouryears/zero', 'title': 'Four years with existing heat'},
    }
#scenarios = {'NNS' :
#   {'file': 'NNS', 'dir' : kf, 'title': 'No Added Electric Heating'},
#             'PNS' :
#   {'file': 'PNS', 'dir' : kfev, 'title': '100% heat pumps and evs'}
#}
# scenarios = {'NNS' : {'file': 'NNS', 'dir' : kf, 'title': 'No Added Electric Heating'} }
if args.scenario == 'feshp':
    scenarios = {'unchanged' :
      {'file': 'ENS', 'dir' : 'fes_hp/unchanged', 'title': 'Existing heating 80% efficieny'},
                 'hp' :
      {'file': 'FNS', 'dir' : 'fes_hp/hp', 'title': 'FES HP (inc hybrid) 41% and 80% efficiency'}
    }
if args.scenario == 'mpml':
    scenarios = {
     'KF' : {'file': 'CM' , 'dir' : kfig8, 'title': 'KF matlab data'},
     'MP' : {'file': 'ENH', 'dir' : kfig8, 'title': 'MP method with historic electric'}
    }
if args.scenario == 'temp':
    scenarios = {'temp' : {'file': 'NNS', 'dir' : temp, 'title': 'Synthetic Electric with shift'} }
if args.scenario == 'yearly':
    scenarios = {'all' : {'file': 'ENS', 'dir' : 'allS85', 'title': 'Synthetic 85% all years'} }
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
settings = {}
max_sl = 0
print('scenario     number of annual  capacity')
print('             days      energy  to supply load')
for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    if len(label)>max_sl:
        max_sl = len(label)
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
    print('{: <12} {}  {:.2f}  {:.2f}'.format(key, ndays, annual_energy, capacity))
    capacities[key] = capacity
    path = '{}/{}/settings{}.csv'.format(output_dir, folder, filename)
    if exists(path):
        setting = readers.read_settings(path)
    else:
        setting = {'storage' : 'kf' }
    settings[key] = setting

# Load the shares dfs

print('Scenario   zero  viable  total  max storage min')
dfs={}
gen_cap={}
stats={}
last_viable = pd.DataFrame()
for key, scenario in scenarios.items():
    folder = scenario['dir']
    label = scenario['title']
    filename = scenario['file']
    path = '{}/{}/shares{}.csv'.format(output_dir, folder, filename)
    df = pd.read_csv(path, header=0, index_col=0)
#   print(df)

    # should not need this since don't overfill ?
#   storage_values = df['storage'].values
#   for i in range(storage_values.size):
#       if storage_values[i] < 0.0:
#           storage_values[i] = 0.0;
#   df['storage'] = storage_values

    wind_at_1 = df[df['f_wind']==1.0]
    if len(wind_at_1.index) == 0 or 'gw_wind' not in wind_at_1.columns:
        gen_cap[key] = 30
    else:
        gen_cap[key] = wind_at_1['gw_wind'].values[0]
  
    # calculate cost and energy
#   storage.configuration_cost(df)
    storage.generation_cost(df)
    df['energy'] = df['wind_energy'] + df['pv_energy']
    df['fraction'] = df['wind_energy'] / df['energy']

    if args.last == 'full':
        viable = df[df['last']==0.0]
    else:
        if args.last == 'p3':
            if len(day_list) == 1:
                last_val = day_list[0] * 0.03
            else:
                last_val = 25 * 0.03
            viable = df[df['last']>-last_val]
        else:
            viable = df
    zero = df[df['storage']==0.0]
    dfs[key] = viable
    store_max = viable['storage'].max()
    store_min = viable['storage'].min()
    print('{: <12}  {}  {}    {}   {:.2f}    {:.2f} '.format(key, len(zero), len(viable), len(df), store_max, store_min ) )

    if len(last_viable)>0:
        last_viable = pd.merge(last_viable, viable, how='inner', on=['f_pv', 'f_wind'])
    else:
        last_viable = viable

# output comparison values

print('COMPARISONS ****')
print_titles(max_sl)

outputs=[]
excess_wind=None
excess_pv=None
for key, scenario in scenarios.items():
    label = scenario['title']
    df = dfs[key]
    # Minimum storage point for 50% excess energy from first scenario
    edf = df[df['energy']<1.0 + args.excess]
    min_excess = storage.min_point(edf, 'storage', 'f_wind', 'f_pv')
    output = print_min(min_excess, '{} excess this '.format(args.excess), label, max_sl)
    outputs.append(output)
    if not excess_wind:
        excess_wind = min_excess['wind']
        excess_pv = min_excess['pv']
#       print('EXCESS ENERGY compare point: wind {} pv {}'.format(excess_wind, excess_pv) )
    excess_point = storage.get_point(df, excess_wind, excess_pv, 'f_wind', 'f_pv')
    print_min(excess_point, '{} excess first'.format(args.excess), label, max_sl)
    # print the minimum cost point of the scenario
    min_energy = storage.min_point(df, 'cost', 'f_wind', 'f_pv')
    print_min(min_energy, 'minimum cost    ', label, max_sl)
    # print the minimum storage point of the scenario
    min_energy = storage.min_point(df, 'storage', 'f_wind', 'f_pv')
    print_min(min_energy, 'min storage     ', label, max_sl)

# Plot storage heat maps

if args.plot:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        pdf = get_viable(df, args.last, day_list[0])
        pdf = pdf[pdf['storage']>0.0]
        scatterHeat(pdf, 'storage', 'Storage in days ', label, args.annotate)

if args.plot:
    # Plot viable solutions
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'last', 'Store remaining in days ', label)

if args.pfit:
    # Plot viable solutions
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]

        pvwind3 = df[df['f_wind']==3.0]
        plt.scatter(pvwind3['f_pv'], pvwind3['storage'], s=12)
        plt.title('Variation of PV and storage for wind=3.0')
        plt.xlabel('PV fraction')
        plt.ylabel('Storage (days)')
        plt.show()

        pvwind3 = df[(df['f_wind']==3.0) & (df['storage']<100) & (df['f_pv']>0.0)]
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

# variables and axis labels
axis_labels = {
    'Ps': 'Solar PV ( energy in proportion to normalised demand)',
    'Pw': 'Wind ( energy in proportion to nomarlised demand)',
    'energy' : 'Energy generated ( normalised to demand )',
    'fraction' : 'Wind energy fraction',
}

# Plot constant storage lines

#pws = {}
first = True
#print('Scenario   Points in contour  Generation capacity')
markers = ['o', 'v', '+', '<', 'x', 'D']
styles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted' ]
scount=0
for key, scenario in scenarios.items():
    df = dfs[key].copy()
    filename = scenario['file']
    label = scenario['title']
    # get the generation capacity
    generation_capacity = gen_cap[key]
#   testx = wind2gw(2)
    wind_parm = 'f_wind'
    pv_parm = 'f_pv'
    if args.energy:
        wind_parm = 'wind_energy'
        pv_parm = 'pv_energy'
    # calculate constant storage line for 40 days
    # or the value specified
    storage_model = settings[key]['storage']
    baseload = float(settings[key]['baseload'])
    for days in day_list:
        storage_line = get_storage_line(df, storage_model, days, 'f_wind', 'f_pv', args.svariable)
        if len(storage_line) == 0:
            print('Skipping line {: <12} {} {} '.format(key, days, len(storage_line) ))
            continue

        # print the line details
#       print('LINE {} {}'.format(days, label))
#       print('FROM wind {} pv {} TO wind {} pv {}'.format(storage_line.head(1)['Pw'].values[0], storage_line.head(1)['Ps'].values[0], storage_line.tail(1)['Pw'].values[0], storage_line.tail(1)['Ps'].values[0]) )
        # print the minimum energy point in the contour
        min_energy = storage.min_point(storage_line, 'energy')
        print_min(min_energy, '{:.1f} days energy'.format(days), label, max_sl)
        # print the minimum cost point in the contour
        min_energy = storage.min_point(storage_line, 'cost')
        print_min(min_energy, '{:.1f} days cost  '.format(days), label, max_sl)
        # print the minimum storage point in the contour
        min_energy = storage.min_point(storage_line, 'storage')
        print_min(min_energy, '{:.1f} days storag'.format(days), label, max_sl)
        # save axis for the first one, and plot
        if first:
            ax = storage_line.plot(x='Pw',y='Ps',label='{} {:.1f} days. {}'.format(args.svariable, days, label), marker=markers[scount], linestyle=styles[scount])
#           line1 = storage_line
#           label1 = label
        else:
            storage_line.plot(x='Pw',y='Ps',ax=ax,label='{} {:.1f} days. {}'.format(args.svariable, days, label), marker=markers[scount], linestyle=styles[scount])
#           line2 = storage_line
#           label2 = label
            # If we are on the 2nd or greater scenario then compare the amount
            # of storage, pv and wind with the previous scenario
#           wind_diff, ratio1, ratio2 = storage.compare_lines(line1, line2)
#           print('Scenario {} to {}          {}      {} '.format(label2, label1, label1, label2) )
#           print('Difference in wind {:.2f} pv/wind {:.2f} {:.2f} '.format(wind_diff, ratio1, ratio2) )
#           line1 = line2
#           label1 = label2

#           storage_diff = df['storage'] - last_df['storage']
#           print('Mean storage difference between {} {} and {} {} is {}'.format(key, df['storage'].mean(), last_key, last_df['storage'].mean(), storage_diff.mean() ) )
#       last_df = df
#       last_key = key
        first = False
    scount+=1

    # plot energy generation of 1.0 line 
    if args.min:
        min_days = 1.0 - (baseload * 0.6)
        energy_line = get_storage_line(df, storage_model, min_days, 'f_wind', 'f_pv', 'energy')
        energy_line.plot(x='Pw',y='Ps',ax=ax,label='energy {:.2f} days. {}'.format(min_days,label), marker=markers[scount])
        scount+=1
        min_gen_line(ax, min_days, markers[scount])
        scount+=1

plt.title('Constant {} lines {}'.format(args.svariable, scenario_title) )
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
#print('Storage  PV  {}'.format(' '.join(keys) ) )
#for storage in [25, 30, 40, 60]:
#    pvs = [pws[key][storage]['pv'] for key in keys]
#    print('{}         '.format(storage), ' '.join([' {:.2f} '.format(i) for i in pvs]) )
#    
#print('Storage  Wind  {}'.format(' '.join(keys) ) )
#for storage in [25, 30, 40, 60]:
#    pvs = [pws[key][storage]['wind'] for key in keys]
#    print('{}         '.format(storage), ' '.join([' {:.2f} '.format(i) for i in pvs]) )
    

# compare the yearly files

yearly_dfs = {}
if args.yearly:
    markers=['^', 'o']
    axs=[]
    labels=[]
    # inter-annual variation of electricity demand
    count=0
    for key, scenario in scenarios.items():
        demand = demands[key].copy()
        label = scenario['title']
        yearly_demand = demand.resample('Y').sum()
        yearly_demand.index = yearly_demand.index.year
        yearly_demand.plot(linestyle='dotted', label='Annual Demand {}'.format(label), marker=markers[count])
#       ax = plt.scatter(yearly_demand.index, yearly_demand, s=12, marker=markers[count])
        count+=1
        axs.append(ax)
        labels.append(label)
        # winter months
        monthly_demand = demand.resample('M').sum()
        december = monthly_demand[monthly_demand.index.month==12]
        december.index = december.index.year
        january = monthly_demand[monthly_demand.index.month==1]
        january.index = january.index.year
        february = monthly_demand[monthly_demand.index.month==2]
        february.index = february.index.year
        winter = december + january + february
#       winter.plot(label='Winter Demand {}'.format(label) )
        print('Annual demand max {} min {} Winter demand max {} min {}'.format(yearly_demand.max(), yearly_demand.min(), winter.max(), winter.min() ) )

    plt.title('Interannual variation of electricity demand')
    plt.xlabel('year', fontsize=15)
    plt.ylabel('Annual Demand (TWh)', fontsize=15)
#   plt.legend(loc='upper left', fontsize=15)
    plt.legend(loc='center left', fontsize=15)
#   plt.legend(tuple(axs), tuple(labels), loc='center left', fontsize=15)
    plt.show()
    

    # storage
    for key, scenario in scenarios.items():
        folder = scenario['dir']
        label = scenario['title']
        filename = scenario['file']
        path = '{}/{}/yearly{}.csv'.format(output_dir, folder, filename)
        df = pd.read_csv(path, header=0, index_col=0)
        yearly_dfs[key] = df
        df['storage'].plot(label='Yearly Storage {}'.format(label) )

    plt.title('Difference on yearly term storage')
    plt.xlabel('year', fontsize=15)
    plt.ylabel('Days of storage', fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

    # december 31st
    for key, scenario in scenarios.items():
        folder = scenario['dir']
        label = scenario['title']
        df = yearly_dfs[key]
        df['dec31_wind'].plot(label='December 31st wind'.format(label) )
        df['dec31_pv'].plot(label='December 31st pv'.format(label) )
#       df['dec31_demand'].plot(label='December 31st demand'.format(label) )

    plt.title('December 31st Generation')
    plt.xlabel('year', fontsize=15)
    plt.ylabel('Normalised Energy', fontsize=15)
    plt.legend(loc='upper left', fontsize=15)
    plt.show()


if args.pdemand:
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

# compare the shares files

last_viable.dropna(inplace=True)
#print(last_viable)

viable_configs = last_viable[['f_pv', 'f_wind']]
total_storage = {}
for key, scenario in scenarios.items():
    common_viable = pd.merge(viable_configs, dfs[key], how='inner', on=['f_pv', 'f_wind'])
    total_storage[key] = common_viable['storage'].mean()

print('Scenario  common  storage')
for key, scenario in scenarios.items():
    print('{: <12}  {}    {:.2f} '.format(key, len(last_viable), total_storage[key] ) )

#wind_diff, ratio1, ratio2 = storage.compare_lines(line1, line2)
#print('Scenario {} to {}          {}      {} '.format(label2, label1, label1, label2) )
#print('Difference in wind {:.2f} pv/wind {:.2f} {:.2f} '.format(wind_diff, ratio1, ratio2) )

# scatter plot of storage and energy
if args.plot:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        pdf = get_viable(df, args.last, day_list[0])
        pdf['energy'] = (pdf['f_wind'] / 0.38) + ( pdf['f_pv'] / 0.1085 )
        pdf['ratio'] = pdf['f_wind'] / pdf['f_pv']
        pdf.plot.scatter(x='energy', y='storage', c='ratio', colormap='viridis')
        plt.xlabel('Energy generated')
        plt.ylabel('Storage days')
        plt.title('Colour by ratio wind/pv {}  '.format(label))
        plt.show()

# scatter plot of storage and wind
if args.plot:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        pdf = get_viable(df, args.last, day_list[0])
        pdf.plot.scatter(x='f_wind', y='storage', c='f_pv', colormap='viridis')
        plt.xlabel('Normalised Wind Capacity')
        plt.ylabel('Storage days')
        plt.title('{}  '.format(label))
        plt.show()

if args.pstore:
    durations = {}
    # sample store history
    for key, scenario in scenarios.items():
        label = scenario['title']
        filename = scenario['file']
        folder = scenario['dir']
        path = '{}/{}/store{}.csv'.format(output_dir, folder, filename)
        store = pd.read_csv(path, header=0, index_col=0, squeeze=True)
        store.index = pd.DatetimeIndex(pd.to_datetime(store.index).date)
        store.plot(label='Store size: {}'.format(label) )
        durations[key] = storage.storage_duration(store)

    plt.xlabel('Time')
    plt.ylabel('Storage days')
    plt.title('Store history ')
    plt.legend(loc='lower left', fontsize=15)
    plt.show()

    # storage duration
    for key, scenario in scenarios.items():
        label = scenario['title']
        filename = scenario['file']
        folder = scenario['dir']
        path = '{}/{}/duration{}.csv'.format(output_dir, folder, filename)
#       store = pd.read_csv(path, header=0, index_col=0, squeeze=True)
#       store.index = pd.DatetimeIndex(pd.to_datetime(store.index).date)
        store = durations[key]
        store.plot(label='Store duration: {}'.format(label) )

    plt.xlabel('Storage capacity in days')
    plt.ylabel('Time in days the store contained more than this')
    plt.title('Store duration ')
    plt.legend(loc='upper left', fontsize=15)
    plt.show()

if args.variable:
    # For PV=2, get the winds out of the contours
    # TODO need to get for a particular wind and pv what its 
    #      storage is for the different base loads.
    print('Plotting {} against storage'.format(args.variable))
    lines=[]
    for p_wind in np.arange(3.0, 4.0, 0.5):
        for p_pv in np.arange(1.0, 5.0, 1.0):
            values=[]
            stores=[]
            for key, scenario in scenarios.items():
                value = float(key)
                df = dfs[key]
                point = df[(df['f_wind']==p_wind) & (df['f_pv']==p_pv)]
                s_value = point['storage']
                if len(s_value)==0:
                    print('No value for wind {} pv {}'.format(p_wind, p_pv))
#                   print(df)
#                   quit()
                else:
                    stores.append(s_value.iloc[0])
                    values.append(value)
            data = { 'storage' : stores, 'values' :values }
            line = pd.DataFrame(data=data)
            label = 'Wind {} PV {} '.format(p_wind, p_pv)
#           print('Wind {} PV {} '.format(p_wind, p_pv))
#           print(line)
            lines.append(line)
            plt.plot(line['values'], line['storage'],label=label )

    plt.xlabel('Base load capacity in days')
    plt.ylabel('Days of storage')
    plt.title('Base load vs Storage')
    plt.legend(loc='upper right', fontsize=12)
    plt.show()

# output csv file
output_dict = {}
for name, value in outputs[0].items():
    output_dict[name] = []
for output in outputs:
    for name, value in output.items():
        output_dict[name].append(value)
df_out = pd.DataFrame(output_dict)
df_out.to_csv('/home/malcolm/uclan/output/scenarios/{}.csv'.format(args.scenario), float_format='%g', index=False)
