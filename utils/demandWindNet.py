# python script to create a net demand time series for a specified wind capacity
# and wind series. Then to calculate the:
# - area under the net demand curve
# - variance of the curve

# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np

# custom code
import stats
import readers

# plot net demand
def plot_net_demand(wind, pv, fraction, demand):
    supply = fraction * wind + (1-fraction) * pv
    net = demand - supply
    zero = net * 0.0
    net.plot(color='blue')
    zero.plot(color='red')
    plt.title('Net demand for wind fraction {}'.format(fraction))
    plt.xlabel('Day of the year', fontsize=15)
    plt.ylabel('Normalised net demand', fontsize=15)
#   plt.legend(loc='upper center')
    plt.show()

# normalize a series
def normalize(s):
    max_value = s.max()
    min_value = s.min()
    n = (s - min_value) / (max_value - min_value)
    return n

# area under curve
def curve_area(s):
    area = np.trapz(s.clip(0.0).values)
    return area

# Pearsons correlation coefficient
def correlation(s1, s2):
    corr = s1.corr(s2)
    return corr


def normalize(df):
    normalized_df=(df-df.min())/(df.max()-df.min())
    return normalized_df

def read_demand(filename, ninja_start, ninja_end):
    demand_dir = '/home/malcolm/uclan/output/hourly/gbase04/'
    demand = pd.read_csv(demand_dir+filename, header=0, squeeze=True, parse_dates=[0], index_col=0)
#   demand.index = pd.DatetimeIndex(demand['time'])
    demand = demand[ninja_start : ninja_end]
    return demand

# process command line
parser = argparse.ArgumentParser(description='Compare demand profile with generation')
parser.add_argument('--wind', action="store", dest="wind", help='Source of the series' , default='near')
parser.add_argument('--shore', action="store", dest="shore", help='Onshore or off' , default='onshore')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--capacity', action="store", dest="capacity", help='Capacity if wind', default=1, type=float)
parser.add_argument('--scale', action="store", dest="scale", help='Scale factor for onshore', default=0.3927, type=float)
args = parser.parse_args()

# load ninja generation data and normalise
ninja_start = '1980-01-01 00:00:00'
ninja_end = '2019-12-31 23:00:00'
# Ninja capacity factors for wind
if args.wind == 'near':
    ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_near-termfuture-merra-2_corrected.csv'
else:
    if args.wind == 'future':
        # ninja future
        ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_long-termfuture-merra-2_corrected.csv'
    else:
        if args.wind == 'current':
        # ninja current
            ninja_filename_wind = '/home/malcolm/uclan/data/ninja/ninja_wind_country_GB_current-merra-2_corrected.csv'
        else:
            print('Invalide wind')
            quit()

wind_hourly = readers.read_ninja_country(ninja_filename_wind)

ninja_wind = wind_hourly[ninja_start : ninja_end]
ninja_wind = ninja_wind[args.shore]

if args.shore == 'onshore':
    ninja_wind = ninja_wind * args.scale / ninja_wind.mean()

# load demand with existing heat
demand = read_demand('demandENS.csv', ninja_start, ninja_end)

# net demand
net = demand - args.capacity * ninja_wind

print('#### Capacity Factors')
print('Series mean    min      max    length')
print('Wind   {:.4f}  {:.4f}  {:.4f}  {}'.format(ninja_wind.mean(), ninja_wind.min(), ninja_wind.max(), len(ninja_wind) ) )
print('Demand {:.4f}  {:.4f}  {:.4f}  {}'.format(demand.mean(), demand.min(), demand.max(), len(demand) ) )
print('Net    {:.4f}  {:.4f}  {:.4f}  {}'.format(net.mean(), net.min(), net.max(), len(net) ) )
print(' ---------------- ')

# plot 

if args.plot:
    net.plot(label='Net demand')
    plt.title('Demand net of renewables')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.show()

area = curve_area(net)
variance = net.var()

print(' area {} variance {} max {}'.format(area, variance, net.max()))
