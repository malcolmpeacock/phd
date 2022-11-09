# Factor the wind generation from espini (ie national grid and elexon) so that
# the increase in wind capacity is accounted for and convert it to 
# a time series of capacity factors.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pytz
import argparse
import calendar

# custom code
import stats
import readers
import stats

def get_wind(start_year, end_year):
    start = '{}-01-01 00:00:00'.format(start_year)
    end = '{}-12-31 23:00:00'.format(end_year)
    demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
    data = readers.read_espeni(demand_filename, None, ['ELEXM_utc', 'POWER_ESPENI_MW', 'POWER_ELEXM_WIND_MW', 'POWER_NGEM_EMBEDDED_WIND_GENERATION_MW'])
    data = data[ start : end ]
#   print(data)
    wind = data['POWER_ELEXM_WIND_MW'] + data['POWER_NGEM_EMBEDDED_WIND_GENERATION_MW']
    return wind

# process command line
parser = argparse.ArgumentParser(description='Convert national grid wind generation to capacity factors')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

start_year = 2011
end_year = 2019

# get wind
wind = get_wind(str(start_year), str(end_year))
print(wind)

if args.plot:
    wind.plot(color='blue', label='National grid wind Generation')
    plt.title('National grid wind generation')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

# get capacity
capacity_file = '/home/malcolm/uclan/data/electricity/monthlyCapacity.csv'
capacity = pd.read_csv(capacity_file, header=0) 
# print(capacity)

quarters =[]
# get capacities for each quarter
for year in range(start_year, end_year+1):
#   print(year)
    for quarter in range(0,4):
        heading = '{} Q{}'.format(str(year), str(quarter+1) )
#       print(heading)
        cap_quarter = capacity[heading]
#       print(cap_quarter)
#       Sum of the onshore and offshore capacity
        cap_wind = float(cap_quarter[0].replace(',','')) + float(cap_quarter[1].replace(',',''))
        #
        quarter_start = '{}-{:02d}-01 00:00:00+00:00'.format(year, 1+quarter*3)
        month = 3+quarter*3
        day,ndays = calendar.monthrange(year, month)
        quarter_end = '{}-{:02d}-{} 23:00:00+00:00'.format(year,month,ndays)
#       print(quarter_start, quarter_end, cap_wind, year, month)
        quarter_wind = wind.loc[quarter_start : quarter_end]
        quarter_wind = quarter_wind / cap_wind
#       print(quarter_wind)
        quarters.append(quarter_wind)

all_wind = pd.concat(quarters[n] for n in range(len(quarters)) )
print(all_wind)
if args.plot:
    all_wind.plot(color='blue', label='National grid wind capacity factors')
    plt.title('National grid wind capacity factors')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

print('Capacify Factor min {} mean {} max {}'.format(all_wind.min(), all_wind.mean(), all_wind.max() ) )

all_wind.to_csv('/home/malcolm/uclan/data/electricity/wind_national_grid.csv', header=False)
