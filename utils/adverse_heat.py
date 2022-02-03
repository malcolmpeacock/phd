# python script to validate the heat demand methods against gas meter data

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
# custom code
import stats

# main program
parser = argparse.ArgumentParser(description='Calculate annual heat demand and mean temp from adverse heat demand files')
parser.add_argument('--adverse', action="store", dest="adverse", help='Adverse file mnemonic', default='a5s2')
args = parser.parse_args()

# make up full file name from the abreviation
etypes = { 'd' : 'duration', 's': 'severity' }
warmings = { 'a' : '12-3', 'b' : '12-4', 'c': '4' }
warming = args.adverse[0:1]
p_end = len(args.adverse)-2
period = args.adverse[1:p_end]
etype = args.adverse[p_end:p_end+1]
eno = args.adverse[p_end+1:p_end+2]
filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/adv/winter_wind_drought_uk_return_period_1_in_{}_years_{}_gwl{}degC_event{}.csv'.format(period, etypes[etype], warmings[warming], eno)

# read in the data
#output_dir = "/home/malcolm/uclan/tools/python/scripts/heat/output/adv/"
#filename = output_dir + 'winter_wind_drought_uk_return_period_1_in_5_years_severity_gwl12-3degC_event2.csv'
df = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

heat = df['heat'].resample('Y').sum() * 1e-6
temp = df['temperature'].resample('Y').mean()

print('Warming  Heat Demand   Mean Temp')
print(' {}      {}            {}       '.format(warming, heat.values[0], temp.values[0]) )
print(' {}      {}            {}       '.format(warming, heat.values[1], temp.values[1]) )
print(' {}      {}            {}       '.format(warming, heat.mean(), temp.mean()) )
