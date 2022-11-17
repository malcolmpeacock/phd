# Compare and plot different scenarios of wind and solar shares and storage
# using 40 years weather

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from os.path import exists
from skimage import measure

# custom code
import stats
import readers
import storage

def print_titles(sl=11):
    print('Scenario'.ljust(sl), 'Point             Wind  PV    Fraction-C/E NP Energy Storage Cost')

def print_min(point, point_title, scenario_title, sl=11):
#   print(point_title)
#   print(point)
    if point['f_wind']>0.0001:
        capacity_wind_fraction = point['f_wind'] / ( point['f_wind'] + point['f_pv'] )
    else:
        capacity_wind_fraction = 0.0
    energy_wind_fraction = point['fraction']
    print('{} {}  {:.2f}  {:.2f}  {:.2f} {:.2f}    {}  {:.2f}   {:05.2f}   {:.3f}'.format(scenario_title.ljust(sl), point_title,  point['f_wind'], point['f_pv'], capacity_wind_fraction, energy_wind_fraction, point['np'], point['energy'], point['storage'], point['cost'] ) )
    output = point
    output['fraction'] = energy_wind_fraction
    output['cfraction'] = capacity_wind_fraction
    output['scenario'] = scenario_title
    output['point'] = point_title
    return output

def get_skline(df, storage_model, days, wind_parm='f_wind', pv_parm='f_pv', variable='storage'):
    contours = measure.find_contours(df[variable], days)
    print(contours)
    quit()

def get_storage_line(df, storage_model, days, wind_parm='f_wind', pv_parm='f_pv', variable='storage'):
    if storage_model == 'new':
        storage_line = df[df[variable] == days].copy()
        storage_line = storage_line.sort_values([wind_parm, pv_parm], ascending=[True, True])
    else:
        if args.sline == 'skline':
            storage_line = get_skline(df, days, args.sline, wind_parm, pv_parm, variable)
        else:
            storage_line = storage.storage_line(df, days, args.sline, wind_parm, pv_parm, variable)
    storage_line['energy'] = storage_line['wind_energy'] + storage_line['pv_energy']
    storage_line['fraction'] = storage_line['wind_energy'] / storage_line['energy']
    storage_line['cfraction'] = storage_line['f_wind'] / (storage_line['f_pv'] + storage_line['f_wind'] )
    return storage_line

def get_viable(df, last, days):
    if args.last == 'full':
        pdf = df[df['last']==100.0]
    else:
        if args.last == 'p3':
            pdf = df[df['last']>97.0]
        else:
            pdf = df
    return pdf

#
# Functions to convert to and from actual capacity based on demandNNH.csv
def cf2gw(x):
    return x * generation_capacity
def gw2cf(x):
    return x / generation_capacity

# 3d scatter
def scatter3d(ax, df, variable, title, label):
    ax.set_xlabel(args.sx)
    ax.set_ylabel(args.sy)
    ax.set_zlabel(variable)
    ax.set_title(label)

    # Data for three-dimensional scattered points
    zdata = df[variable]
    xdata = df[args.sx]
    ydata = df[args.sy]
#   ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis');

def scatterHeat(df, variable, title, label, annotate=False):
    ax = df.plot.scatter(x=args.sx, y=args.sy, c=variable, colormap='viridis')
    plt.xlabel(axis_labels[args.sx])
    plt.ylabel(axis_labels[args.sy])
    plt.title('{} for different proportions of {} and {} ({} ).'.format(title, args.sx, args.sy, label))
    if annotate:
        for i, point in df.iterrows():
            ax.text(point[args.sx],point[args.sy],'{:.1f}'.format(point[variable]))
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

    min_energy_line = { 'f_wind' : Pw, 'f_pv' : Ps }
    df_min = pd.DataFrame(data=min_energy_line)
    df_min.plot(x='f_wind', y='f_pv', ax=ax, label='theorectical minimum generation from mean cf', marker=marker)


# main program

# process command line
parser = argparse.ArgumentParser(description='Compare and plot scenarios')
parser.add_argument('--rolling', action="store", dest="rolling", help='Rolling average window', default=0, type=int)
parser.add_argument('--decimals', action="store", dest="decimals", help='Number of decimal places', default=2, type=int)
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--mblack', action="store_true", dest="mblack", help='Plot min point marker in black', default=False)
parser.add_argument('--nolines', action="store_true", dest="nolines", help='Do not plot the contour lines', default=False)
parser.add_argument('--markevery', action="store", dest="markevery", help='Marker frequency', default=1, type=int)
parser.add_argument('--compare', action="store_true", dest="compare", help='Output comparison stats', default=False)
parser.add_argument('--pstore', action="store_true", dest="pstore", help='Plot the sample store history ', default=False)
parser.add_argument('--pdemand', action="store_true", dest="pdemand", help='Plot the demand ', default=False)
parser.add_argument('--heatdiff', action="store_true", dest="heatdiff", help='Create a heat map as difference of 2 scenarios', default=False)
parser.add_argument('--pfit', action="store_true", dest="pfit", help='Show 2d plots', default=False)
parser.add_argument('--pmin', action="store", dest="pmin", help='Plot minimum point of given variable', default=None)
parser.add_argument('--dcolour', action="store_true", dest="dcolour", help='Use same colour for same number of days', default=False)
parser.add_argument('--yearly', action="store_true", dest="yearly", help='Show Yearly plots', default=False)
#parser.add_argument('--energy', action="store_true", dest="energy", help='Plot energy instead of capacity', default=False)
parser.add_argument('--rate', action="store_true", dest="rate", help='Plot the charge and discharge rates', default=False)
parser.add_argument('--stype', action="store", dest="stype", help='Type of Storage: pumped, hydrogen, caes.', default='pumped', choices=['pumped','hydrogen','caes', 'none' ])
parser.add_argument('--min', action="store_true", dest="min", help='Plot the minimum generation line', default=False)
parser.add_argument('--annotate', action="store_true", dest="annotate", help='Annotate the shares heat map', default=False)
parser.add_argument('--scenario', action="store", dest="scenario", help='Scenarion to plot', default='adhoc')
parser.add_argument('--days', action="store", dest="days", help='Days of storage line to plot', default='0.5, 1, 3, 10, 25, 30, 40, 60' )
parser.add_argument('--sline', action="store", dest="sline", help='Method of creating storage lines', default='interp1', choices=['interp1','threshold','smooth','sboth','both','skline'])
parser.add_argument('--cvariable', action="store", dest="cvariable", help='Variable to contour, default is storage', default='storage')
parser.add_argument('--cx', action="store", dest="cx", help='X Variable for contour creation, default is f_wind', default='f_wind')
parser.add_argument('--cy', action="store", dest="cy", help='Y Variable for contour creation, default is f_pv', default='f_pv')
parser.add_argument('--sx', action="store", dest="sx", help='Variable to plot on the X axis, default is f_wind', default='f_wind')
parser.add_argument('--sy', action="store", dest="sy", help='Variable to plot on the Y axis, default is f_pv', default='f_pv')
parser.add_argument('--adverse', action="store", dest="adverse", help='Adverse file mnemonic', default='5s1')
parser.add_argument('--last', action="store", dest="last", help='Only include configs which ended with store: any, full, p3=3 percent full ', default='any')
parser.add_argument('--shore', action="store", dest="shore", help='Wind to base cost on both, on, off . default = both ', default='both')
parser.add_argument('--excess', action="store", dest="excess", help='Excess value to find minimum storage against', type=float, default=0.5)
parser.add_argument('--variable', action="store", dest="variable", help='Variable to plot from scenario', default=None)
parser.add_argument('--heat', action="store", dest="heat", help='Variable to plot a heat map of', default=None)
parser.add_argument('--surface', action="store", dest="surface", help='Variable to plot 3d surface with', default=None)
parser.add_argument('--pwind', action="store", dest="pwind", help='Print points with this wind proportion', default=None, type=float)
parser.add_argument('--ppv', action="store", dest="ppv", help='Print points with this PV proportion', default=None, type=float)
parser.add_argument('--costmodel', action="store", dest="costmodel", help='Cost model A or B', default='B', choices=['A', 'B', 'C'])
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
    scenario_title = 'for with and without electification of transport'
    scenarios = {'noev' :
       {'file': 'ENS', 'dir' : 'ev40years/noev/', 'title': 'No Evs'},
                 'ev' : 
       {'file': 'ENS', 'dir' : 'ev40years/ev/', 'title': 'Evs '} ,
                 'heat' : 
       {'file': 'FNS', 'dir' : 'ev40years/ev/', 'title': 'Evs and heat electrification'} 
    }
if args.scenario == 'hourly':
    scenario_title = 'for hourly and daily time series'
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
if args.scenario == 'hydrogenfesb':
    scenario_title = 'The impact of electrification of heating for difference base load'
    scenarios = {'he7' :
       {'file': 'ENS', 'dir' : 'hydrogen/gbase07/', 'title': 'Base load 0.7 existing heating'},
                 'hfes7' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase07/', 'title': 'Base load 0.7 41% heat pumps'},
                 'he5' : 
       {'file': 'ENS', 'dir' : 'hydrogen/gbase05/', 'title': 'Base load 0.5 existing heating'},
                 'hfes5' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase05/', 'title': 'Base load 0.5 41% heat pumps'},
                 'he3' : 
       {'file': 'ENS', 'dir' : 'hydrogen/gbase03/', 'title': 'Base load 0.3 existing heating'},
                 'hfes3' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase03/', 'title': 'Base load 0.3 41% heat pumps'},
                 'he0' : 
       {'file': 'ENS', 'dir' : 'hydrogen/gbase00/', 'title': 'Base load 0.0 exsting heating'},
                 'hfes0' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase00/', 'title': 'Base load 0.0 41% heat pumps'} 
    }
if args.scenario == 'hydrogencaes':
    scenario_title = 'The impact of electrification of heating'
    scenarios = {'he' :
       {'file': 'ENS', 'dir' : 'hydrogen/gbase04/', 'title': 'Storage 50% efficient. Existing heating'},
                 'hfes' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase04/', 'title': 'Storage 50% efficient. 41% heat pumps'},
                 'ce' : 
       {'file': 'ENS', 'dir' : 'caes/gbase04/', 'title': 'Storage 70% efficient. Existing heating'},
                 'cfes' : 
       {'file': 'FNS', 'dir' : 'caes/gbase04/', 'title': 'Storage 70% efficient. 41% Heat pumps'} 
    }
if args.scenario == 'hydrogenfes':
    scenario_title = 'The impact of electrification of heating'
    scenarios = {'he' :
       {'file': 'ENS', 'dir' : 'hydrogen/gbase04/', 'title': 'Base load 0.4 existing heating'},
                 'hfes' : 
       {'file': 'FNS', 'dir' : 'hydrogen/gbase04/', 'title': 'Base load 0.4 electrified heat FES Net Zero'} 
    }
if args.scenario == 'hydrogenfesh':
    scenario_title = 'The impact of electrification of heating'
    scenarios = {'he' :
       {'file': 'ENS', 'dir' : 'hourly/gbase04/', 'title': 'Base load 0.4 existing heating'},
                 'hfes' : 
       {'file': 'FNS', 'dir' : 'hourly/gbase04/', 'title': 'Base load 0.4 electrified heat 41% heat pumps'} 
    }
if args.scenario == 'zerowind':
    scenario_title = 'The impact of electrification of heating'
    scenarios = {'he' :
       {'file': 'ENS', 'dir' : 'hydrogen/zerowind/', 'title': 'Base load 0.4 existing heating'},
                 'hfes' : 
       {'file': 'FNS', 'dir' : 'hydrogen/zerowind/', 'title': 'Base load 0.4 electrified heat FES Net Zero'} 
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
if args.scenario == 'hvh':
    scenario_title = 'for hydrogen boilers vs heat pumps'
    scenarios = {'0.0' :
       {'file': 'BNS', 'dir' : 'hydrogen/vheatpumps/', 'title': 'All hydrogen boilers'},
                 '0.41' : 
       {'file': 'FNS', 'dir' : 'hydrogen/vheatpumps/', 'title': 'FES Net Zero 41% heap pumps'},
                 '0.5' : 
       {'file': 'HNS', 'dir' : 'hydrogen/vheatpumps/', 'title': 'Half heat pumps half hydrogen boilers'},
                 '1.0' : 
       {'file': 'PNS', 'dir' : 'hydrogen/vheatpumps/', 'title': 'All heat pumps'}
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
       {'file': 'FNS', 'dir' : 'baseload_all/b20/', 'title': 'Base load 0.20'},
                 '0.25' : 
       {'file': 'FNS', 'dir' : 'baseload_all/b25/', 'title': 'Base load 0.25'}
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
if args.scenario == 'todayc':
    scenarios = {'var11base020' :
       {'file': 'ENS', 'dir' : 'today/var11base020/', 'title': 'Base 020 Variable generation 1.1'},
    }
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
       {'file': 'ENS', 'dir' : 'today/var07base020/', 'title': 'Base 020 Variable generation 0.7'},
                 'var06base020' : 
       {'file': 'ENS', 'dir' : 'today/var06base020/', 'title': 'Base 020 Variable generation 0.6'},
                 'var05base020' : 
       {'file': 'ENS', 'dir' : 'today/var05base020/', 'title': 'Base 020 Variable generation 0.5'},
                 'var04base020' : 
       {'file': 'ENS', 'dir' : 'today/var04base020/', 'title': 'Base 020 Variable generation 0.4'},
                 'var03base020' : 
       {'file': 'ENS', 'dir' : 'today/var03base020/', 'title': 'Base 020 Variable generation 0.3'},
                 'var02base020' : 
       {'file': 'ENS', 'dir' : 'today/var02base020/', 'title': 'Base 020 Variable generation 0.2'},
                 'var01base020' : 
       {'file': 'ENS', 'dir' : 'today/var01base020/', 'title': 'Base 020 Variable generation 0.1'},
                 'var00base020' : 
       {'file': 'ENS', 'dir' : 'today/var00base020/', 'title': 'Base 020 Variable generation 0.0'} 
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
    scenario_title = 'for different store staring sizes'
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
    scenario_title = 'Fig (8) Fragaki et. al. with model and data changes'
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'new_fig8S75/', 'title': 'Existing heating 0.75 Store 70% at start'},
                 'new' : 
       {'file': 'ENS', 'dir' : 'new_fig8S85/', 'title': 'Existing heating 0.85 Store 70% at start'}    }
if args.scenario == 'newfig8min':
    scenario_title = 'Fig (8) Fragaki et. al. with model and data changes'
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : '30dayminS75/', 'title': '0.75 30 day storage min energy'},
                 'new' :
       {'file': 'ENS', 'dir' : '30dayminS85/', 'title': '0.85 30 day storage min energy'}    }
if args.scenario == 'newfig8hg':
    scenario_title = 'Fig (8) Fragaki et. al. with model and data changes'
    scenarios = {'old' :
       {'file': 'ENS', 'dir' : 'new_fig8hgS75/', 'title': 'Existing heating 0.75 Store 70% at start'},
                 'new' : 
       {'file': 'ENS', 'dir' : 'new_fig8hgS85/', 'title': 'Existing heating 0.85 Store 70% at start'}    }
if args.scenario == 'fig8oldvnew':
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
    scenario_title = 'using this model from this study.'
    scenarios = {'allS75' :
       {'file': 'ENS', 'dir' : 'allS75/', 'title': 'Existing heating 0.75'},
                 'allS85' : 
       {'file': 'ENS', 'dir' : 'allS85/', 'title': 'Existing heating 0.85'}    }
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
    scenario_title = '4 distinct decades'
    scenarios = {'decade1' :
       {'file': 'ENS', 'dir' : 'decade1/', 'title': 'Existing heating 1980 - 1989'},
                 'decade2' : 
       {'file': 'ENS', 'dir' : 'decade2/', 'title': 'Existing heating 1990 - 1999'},
                 'decade3' : 
       {'file': 'ENS', 'dir' : 'decade3/', 'title': 'Existing heating 2000 - 2019'},
                 'decade4' : 
       {'file': 'ENS', 'dir' : 'decade4/', 'title': 'Existing heating 2010 - 2019'}    }
if args.scenario == 'decadesnew':
    scenario_title = '4 distinct decades'
    scenarios = {'decade1' :
       {'file': 'ENS', 'dir' : 'new_model/decade1/', 'title': 'Existing heating 1980 - 1989'},
                 'decade2' : 
       {'file': 'ENS', 'dir' : 'new_model/decade2/', 'title': 'Existing heating 1990 - 1999'},
                 'decade3' : 
       {'file': 'ENS', 'dir' : 'new_model/decade3/', 'title': 'Existing heating 2000 - 2019'},
                 'decade4' : 
       {'file': 'ENS', 'dir' : 'new_model/decade4/', 'title': 'Existing heating 2010 - 2019'}    }
if args.scenario == 'historic':
    scenario_title = 'for different methods of generating electricity demand time series'
    scenarios = {'historic' :
       {'file': 'ENH', 'dir' : ninja85, 'title': 'Historic time series'},
                 'synthetic' : 
       {'file': 'ENS', 'dir' : ninja85, 'title': 'Synthetic time series'}    }
if args.scenario == 'generation':
    scenario_title = 'for onshore wind from Ninja compared to Fragaik et. al.'
    scenarios = {'ninja' :
       {'file': 'ENS', 'dir' : 'ninjaOnshore/', 'title': 'Wind Generation from Renewables Ninja (onshore)'},
                 'kf' : 
       {'file': 'ENS', 'dir' : 'kfwind/', 'title': 'Wind Generation from Fragaki et. al. '} }
if args.scenario == 'ninjapv':
    scenario_title = 'PV MIDAS Stations vs Ninja'
    scenarios = {'ninja' :
       {'file': 'ENS', 'dir' : 'ninjaOnshore/', 'title': 'PV Generation from Renewables Ninja'},
                 'kfs' : 
       {'file': 'ENS', 'dir' : 'kfpvs/', 'title': 'PV Generation from Fragaki et. al. scaled to Ninja CF'},
                 'kf' : 
       {'file': 'ENH', 'dir' : 'kfpv/', 'title': 'PV Generation from Fragaki et. al. '} }
if args.scenario == 'shore':
    scenario_title = 'Onshore vs Offshore'
    scenarios = {'off' :
       {'file': 'ENS', 'dir' : 'ninjaOffshore/', 'title': 'Ninja (offshore)'},
                 'on' : 
       {'file': 'ENS', 'dir' : 'ninjaOnshore/', 'title': 'Ninja (onshore)'}    }
if args.scenario == 'pattern':
    scenario_title = 'Impact of wind generation pattern - efficiency 80%'
    scenarios = {'off' :
       {'file': 'ENS', 'dir' : 'ninjaOffshore/', 'title': 'Ninja (offshore near future)'},
                 'ons' : 
       {'file': 'ENS', 'dir' : 'ninjaOnShoreScaled/', 'title': 'Ninja (onshore scaled to offshore cf)'},
                 'current' : 
       {'file': 'ENS', 'dir' : 'ninjaOffshoreCurrent/', 'title': 'Ninja (offshore current)'},
                 'kf' : 
       {'file': 'ENS', 'dir' : 'kfwindScaled/', 'title': 'Fragaki. et. al. (onshore scaled to ninja offshore cf )'},
    }
if args.scenario == 'patternh':
    scenario_title = 'Impact of wind generation pattern - efficiency 50%'
    scenarios = {'off' :
       {'file': 'ENS', 'dir' : 'hydrogen/ninjaOffshore/', 'title': 'Ninja (offshore near future)'},
                 'ons' : 
       {'file': 'ENS', 'dir' : 'hydrogen/ninjaOnShoreScaled/', 'title': 'Ninja (onshore scaled to offshore cf)'},
#                'current' : 
#      {'file': 'ENS', 'dir' : 'ninjaOffshoreCurrent/', 'title': 'Ninja (offshore current)'},
#                'kf' : 
#      {'file': 'ENS', 'dir' : 'kfwindScaled/', 'title': 'Fragaki. et. al. (onshore scaled to ninja offshore cf )'},
    }
if args.scenario == 'shores':
    scenario_title = 'Onshore vs Offshore'
    scenarios = {'off' :
       {'file': 'ENS', 'dir' : 'ninjaOffshore/', 'title': 'Ninja (offshore)'},
                 'on' : 
       {'file': 'ENS', 'dir' : 'ninjaOnshore/', 'title': 'Ninja (onshore)'},
                 'ons' : 
       {'file': 'ENS', 'dir' : 'ninjaOnShoreScaled/', 'title': 'Ninja (onshore scaled to offshore cf)'}    }
if args.scenario == 'soc':
    scenario_title = 'Onshore vs Offshore'
    scenarios = {'off' :
       {'file': 'ENS', 'dir' : 'experiments/offshore/', 'title': 'Ninja (offshore)'},
                 'on' : 
       {'file': 'ENS', 'dir' : 'experiments/onshore/', 'title': 'Ninja (onshore)'},
                 'ons' : 
       {'file': 'ENS', 'dir' : 'experiments/onshores/', 'title': 'Ninja (onshore scaled to offshore cf)'}    }
if args.scenario == 'soc0':
    scenario_title = 'Onshore vs Offshore'
    scenarios = {'off' :
       {'file': 'ENS', 'dir' : 'experiments/zeropv/offshore/', 'title': 'Ninja (offshore)'},
                 'on' : 
       {'file': 'ENS', 'dir' : 'experiments/zeropv/onshore/', 'title': 'Ninja (onshore)'},
                 'ons' : 
       {'file': 'ENS', 'dir' : 'experiments/zeropv/onshores/', 'title': 'Ninja (onshore scaled to offshore cf)'}    }
if args.scenario == 'kfig8':
    scenario_title = 'using model, generation and demand data from Fragaki et. al.'
    scenarios = {'S75' :
       {'file': 'NNH', 'dir' : 'kfig8S75/', 'title': 'Efficiency 75%'},
                 'S85' : 
       {'file': 'NNH', 'dir' : 'kfig8S85/', 'title': 'Efficiency 85%'}    }
if args.scenario == 'models':
    scenarios = {'HNSh' :
       {'file': 'PNS', 'dir' : sm, 'title': 'All heat pumps, mp storage model'},
                 'HNSy' : 
       {'file': 'PNS', 'dir' : y40, 'title': 'All heat pumps, kf storage model'}    }
if args.scenario == 'eheat':
    scenarios = {'NNS' :
       {'file': 'NNS', 'dir' : 'heatpaper', 'title': '2018 with heating electricity removed'},
                 'PNS' :
       {'file': 'PNS', 'dir' : 'heatpaper', 'title': 'All heating is provided by heat pumps'},
    }
if args.scenario == 'eheat2':
    scenarios = {'BBB' :
       {'file': 'GNS', 'dir' : 'heatpaper', 'title': '41% heating is provided by heat pumps'},
                 'AAA' :
       {'file': 'ENS', 'dir' : 'heatpaper', 'title': '2018 with existing heating electricity'}
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
if args.scenario == 'cost_comp':
    scenario_title = ' cost model paper comparison'
    scenarios = {'baseline' :
       {'file': 'ENS', 'dir' : 'cost', 'title': 'Synthetic Demand with baseline'},
                 'cost' :
       {'file': 'ENM', 'dir' : 'cost', 'title': 'Scaled demand as Cardenas et. al.'}
    }
if args.scenario == 'cost_comph':
    scenario_title = 'Comparison to Cardenas et. al. '
    scenarios = {'ninja_baseline' :
       {'file': 'ENS', 'dir' : 'cost_hourly', 'title': 'Baseline Demand, Ninja Wind'},
                 'ngrid_baseline' :
       {'file': 'ENS', 'dir' : 'cost_ngrid', 'title': 'Baseline Demand , National Grid Wind.'},
                 'ngrid_scale' :
       {'file': 'ENM', 'dir' : 'cost_ngrid', 'title': 'Scaled demand, National Grid Wind.'},
                 'ninja_scale' :
       {'file': 'ENM', 'dir' : 'cost_hourly', 'title': 'Scaled demand, Ninja Wind'}
    }
if args.scenario == 'cardenas':
    scenario_title = ' CAES 35 days storage'
    scenarios = {'cost' :
       {'file': 'ENM', 'dir' : 'individuals/cardenas_caes_storage', 'title': 'Nine years from cost paper with existing heat'},
    }
if args.scenario == 'cost':
    scenario_title = ' cost model paper comparison'
    scenarios = {'cost' :
       {'file': 'ENM', 'dir' : 'cost', 'title': 'Nine years from cost paper with existing heat'},
    }
if args.scenario == 'cost_hydrogen':
    scenario_title = ' Reproduction of Cardenas et. al. hydrogen cost'
    scenarios = {'cost' :
       {'file': 'ENM', 'dir' : 'cost_ngrid_hydrogen', 'title': 'Round trip 45% scaled as per Cardenas et. al. '},
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

# variables and axis labels
axis_labels = {
    'f_pv': 'Solar PV ( generation capacity in proportion to normalised demand)',
    'f_wind': 'Wind ( generation capacity in proportion to nomarlised demand)',
    'energy' : 'Energy generated ( normalised to demand )',
    'fraction' : 'Wind energy fraction',
    'wind_energy' : 'Wind energy ( normalised to demand )',
    'pv_energy' : 'PV energy ( normalised to demand )',
    'storage' : 'Amount of energy storage (days)',
    'cost' : 'cost ( £/Kwh )',
    'last' : 'store level % fill at end',
}

# variables and axis labels
units = {
    'f_pv': 'days',
    'f_wind': 'days',
    'energy' : 'days',
    'fraction' : '%',
    'wind_energy' : 'days',
    'pv_energy' : 'days',
    'storage' : 'days',
    'cost' : '£/Kwh',
    'last' : '%',
}

# load the demands
demands = {}
capacities = {}
settings = {}
max_sl = 0
print('scenario     number of annual  capacity')
print('             days      demand  to supply load')
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
        if not 'normalise' in setting:
            setting['normalise'] = 818387.7082191781
        if not 'hourly' in setting:
            setting['hourly'] = False
        if not 'baseload' in setting:
            setting['baseload'] = 0.0
    else:
        setting = {'storage' : 'kf', 'baseload' : '0.0', 'start' : 1980, 'end': 2019, 'hourly': False, 'normalise' : 818387.7082191781 }
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
    for col in ['base', 'variable', 'wind_energy', 'pv_energy', 'charge_rate', 'discharge_rate', 'variable_energy', 'yearly_store_min', 'yearly_store_max']:
        if col not in df.columns:
            print('Warning {} missing, setting to zero'.format(col))
            df[col] = 0.0
#   print(df)

    wind_at_1 = df[df['f_wind']==1.0]
    if len(wind_at_1.index) == 0 or 'gw_wind' not in wind_at_1.columns:
        gen_cap[key] = 30
    else:
        gen_cap[key] = wind_at_1['gw_wind'].values[0]
  
    # calculate cost and energy
    n_years = int(settings[key]['end']) - int(settings[key]['start']) + 1
    if args.stype == 'none':
        df['cost'] = 0.0
    else:
        storage.generation_cost(df, args.stype, float(settings[key]['normalise']), n_years, settings[key]['hourly']=='True', args.shore, args.costmodel  )

    # calculate energy
    df['energy'] = df['wind_energy'] + df['pv_energy']
    # calculate wind energy fraction
    df['fraction'] = df['wind_energy'] / df['energy']

    if args.last == 'full':
        viable = df[df['last']==100.0]
    else:
        if args.last == 'p3':
            viable = df[df['last']>97.0]
        else:
            viable = df
    zero = df[df['storage']==0.0]
    dfs[key] = viable
    store_max = viable['storage'].max()
    store_min = viable['storage'].min()
    print('{: <12}  {}  {}    {}   {:.2f}    {:.2f} '.format(key, len(zero), len(viable), len(df), store_max, store_min ) )

    # TODO this merges two together and then the 3rd into it so it gets a wierd result!
    if len(last_viable)>0:
        last_viable = pd.merge(last_viable, viable, how='inner', on=['f_pv', 'f_wind'])
        if args.heatdiff:
            diff_variable = 'storage'
            if args.heat:
                diff_variable = args.heat
            diff_label = '{} difference'.format(diff_variable)
            last_viable['heat_diff'] = last_viable[diff_variable + '_x'] - last_viable[diff_variable + '_y']
            scatterHeat(last_viable, 'heat_diff', diff_label, 'between {} and {}'.format(label, last_label), args.annotate)
    else:
        last_viable = viable
    last_label = label


# output comparison values


outputs=[]
excess_wind=None
excess_pv=None

if args.compare:
    print('COMPARISONS ****')
    print_titles(max_sl)
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        # Minimum storage point for 50% excess energy from first scenario
        edf = df[df['energy']<1.0 + args.excess]
        if len(edf) == 0:
            print('ERROR: {} excess got no points for {}, try {}'.format(args.excess, label, df['energy'].min() + 1 ) )
            quit()
        min_excess = storage.min_point(edf, 'storage', 'f_wind', 'f_pv')
        output = print_min(min_excess, '{} excess this '.format(args.excess), label, max_sl)
        outputs.append(output)
        if not excess_wind:
            excess_wind = min_excess['f_wind']
            excess_pv = min_excess['f_pv']
#       print('EXCESS ENERGY compare point: wind {} pv {}'.format(excess_wind, excess_pv) )
        excess_point = storage.get_point(df, excess_wind, excess_pv, 'f_wind', 'f_pv')
        output = print_min(excess_point, '{} excess first'.format(args.excess), label, max_sl)
        outputs.append(output)
        # print the minimum cost point of the scenario
        min_energy = storage.min_point(df, 'cost', 'f_wind', 'f_pv')
        output = print_min(min_energy, 'minimum cost    ', label, max_sl)
        outputs.append(output)
        # print the minimum storage point of the scenario
        min_storage = storage.min_point(df, 'storage', 'f_wind', 'f_pv')
        output = print_min(min_storage, 'min storage     ', label, max_sl)
        outputs.append(output)

# Plot storage heat maps

if args.heat:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, args.heat, args.heat, label, args.annotate)

# Plot surface
if args.surface:
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))
    pcount=0
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        pcount+=1
        ax = fig.add_subplot(1, 2, pcount, projection='3d')
        scatter3d(ax, df, args.surface, scenario_title, label)
    plt.show()

if args.plot:
    # Plot viable solutions
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        scatterHeat(df, 'last', 'Store % remaining at end', label)

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


# Plot constant storage lines

first = True
markers = ['o', 'v', '+', '<', 'x', 'D', '*', 'X','o', 'v', '+', '<', 'x', 'D', '*', 'X']
styles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted', 'dashed', 'solid', 'dotted', 'dashed', 'dashdot', 'dashdot', 'solid', 'dotted', 'dashed' ]
colours = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'yellow', 'black', 'salmon' ]
scount=0
odd=0
min_vars=[]
if args.pmin:
    min_vars = args.pmin.split(',')

if not args.nolines:
    pmarkers = { 'energy' : '*', 'cost' : 'X', 'storage' : 'p' }
    fig, ax = plt.subplots(constrained_layout=False)
    for key, scenario in scenarios.items():
        df = dfs[key].copy()
        filename = scenario['file']
        label = scenario['title']
        # get the generation capacity
        generation_capacity = gen_cap[key]
    #   testx = wind2gw(2)
    #   wind_parm = 'f_wind'
    #   pv_parm = 'f_pv'
        # calculate constant storage line for 40 days
        # or the value specified
        storage_model = settings[key]['storage']
        baseload = float(settings[key]['baseload'])
        dcount = 0
        for days in day_list:
            if odd==1:
              odd=0
            else:
              odd=1
            storage_line = get_storage_line(df, storage_model, days, args.cx, args.cy, args.cvariable)
            if len(storage_line) == 0:
                print('Skipping line {: <12} {} {} '.format(key, days, len(storage_line) ))
                continue

            # print the minimum energy point in the contour
            min_energy = storage.min_point(storage_line, 'energy')
            output = print_min(min_energy, '{:.1f} days energy'.format(days), label, max_sl)
            outputs.append(output)
            # print the minimum cost point in the contour
            min_cost = storage.min_point(storage_line, 'cost')
            output = print_min(min_cost, '{:.1f} days cost  '.format(days), label, max_sl)
            outputs.append(output)
            # print the minimum storage point in the contour
            min_storage = storage.min_point(storage_line, 'storage')
            output = print_min(min_storage, '{:.1f} days storag'.format(days), label, max_sl)
            outputs.append(output)
            # get the colour
            if args.dcolour:
                line_colour = colours[dcount]
                if len(day_list) ==1:
                    line_colour = colours[odd]
                line_style = linestyle=styles[scount]
                marker_type = markers[scount]
            else:
                line_colour = colours[scount]
                line_style = linestyle=styles[dcount]
                marker_type = markers[dcount]

            # format the line label
            label_string = '{} {:.' + str(args.decimals) + 'f} ({}). {}'
            label_formated = label_string.format(args.cvariable, days, units[args.cvariable], label)
            # plot the storage line
            ax.plot(storage_line[args.sx],storage_line[args.sy],label=label_formated, marker=marker_type, linestyle=line_style, color=line_colour, markevery=args.markevery)
            # Plot minimum point if requested
            last = dcount==len(day_list)-1 and scount==len(scenarios)-1
            for mvar in min_vars:
#               print('VAR {} last {} label {} '.format(mvar, last, label))
                min_ppoint = min_storage
                if mvar == 'energy':
                    min_ppoint = min_energy
                if mvar == 'cost':
                    min_ppoint = min_cost
                # determine the marker colour
                if args.mblack:
                    marker_colour = 'black'
                else:
                    marker_colour = line_colour
                # only include the label on the last one so it comes last
                # in the legend
                if last:
                    ax.plot(min_ppoint[args.sx], min_ppoint[args.sy], label='minimum {}'.format(mvar), marker=pmarkers[mvar], color=marker_colour, ms=14, linestyle='None')
                else:
                    ax.plot(min_ppoint[args.sx], min_ppoint[args.sy], marker=pmarkers[mvar], color=marker_colour, ms=14)

            # day counter
            dcount+=1
        # scenario counter
        scount+=1

        # plot energy generation of 1.0 line 
        if args.min:
            min_days = 1.0 - (baseload * 0.6)
            energy_line = get_storage_line(df, storage_model, min_days, 'f_wind', 'f_pv', 'energy')
            energy_line.plot(x='f_wind',y='f_pv',ax=ax,label='energy {:.2f} days. {}'.format(min_days,label), marker=markers[scount])
            scount+=1
            min_gen_line(ax, min_days, markers[scount])
            scount+=1

    plt.title('Constant {} lines {}'.format(args.cvariable, scenario_title) )
    plt.xlabel(axis_labels[args.sx])
    plt.ylabel(axis_labels[args.sy])
    # 2nd axis#
    if not first and args.sx == 'f_wind' and args.sy=='f_pv':
        axx = ax.secondary_xaxis('top', functions=(cf2gw, gw2cf))
        axx.set_xlabel('Capacity GW')
        axy = ax.secondary_yaxis('right', functions=(cf2gw, gw2cf))
        axy.set_ylabel('Capacity GW')


    plt.legend(loc='best', fontsize=10)
    plt.show()

keys = scenarios.keys()

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
        ax = plt.scatter(yearly_demand.index, yearly_demand, s=12, marker=markers[count])
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
    plt.xlabel('weather year', fontsize=15)
    plt.ylabel('Annual Electricity Demand (TWh)', fontsize=15)
    plt.legend(loc='center left', fontsize=15)

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
    plt.xlabel('weather year', fontsize=15)
    plt.ylabel('Eelectricity Demand (MWh)', fontsize=15)
    plt.legend(loc='upper center', fontsize=15)
    plt.show()

# plot the hydrogen demand

    for key, scenario in scenarios.items():
        folder = scenario['dir']
        label = scenario['title']
        filename = scenario['file']
        path = '{}/{}/hydrogen{}.csv'.format(output_dir, folder, filename)
        demand = pd.read_csv(path, header=0, index_col=0, squeeze=True)
        demand.index = pd.DatetimeIndex(pd.to_datetime(demand.index).date)
        ndays = len(demand)
        print('Hydrogen demand {} for {}'.format(demand.sum() * 365 / ndays, label))

        demand.plot(label='Hydrogen Demand {}'.format(label) )

    plt.title('Daily Hydrogen demand')
    plt.xlabel('weather year', fontsize=15)
    plt.ylabel('Hydrogen Demand (MWh)', fontsize=15)
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

if args.pwind != None or args.ppv != None:
    for key, scenario in scenarios.items():
        label = scenario['title']
        df = dfs[key]
        if args.pwind != None:
            df = df[df['f_wind'] == args.pwind]
        if args.ppv != None:
            df = df[df['f_pv'] == args.ppv]
        print('POINTS : {}'.format(label))
        print(df[['f_wind','f_pv','cost','fraction','storage']])

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
        if args.rolling >0:
            store = store.rolling(args.rolling, min_periods=1).mean()
        store.plot(label='Store size: {}'.format(label) )
        durations[key] = storage.storage_duration(store)

    plt.xlabel('Time')
    plt.ylabel('Storage days')
    plt.title('Store history ')
    plt.legend(loc='lower center', fontsize=15)
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

    plt.xlabel('State Of Charge (stored energy) in days')
    plt.ylabel('Time in days the store contained more energy than this')
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
