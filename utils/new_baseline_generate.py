# Program to generate baseline electricity demand for a given reference year.

# library stuff
import sys
import os
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import calendar
import numpy as np
from scipy.stats import wasserstein_distance

# custom code
import stats
import readers
import storage
import math

start_time = datetime.now()

# main program

# process command line
parser = argparse.ArgumentParser(description='Generate baseline electricity demand without heat')
parser.add_argument('--reference', action="store", dest="reference", help='Reference Year', default='2018' )
parser.add_argument('--filename', action="store", dest="filename", help='Output filename', default='base' )
parser.add_argument('--debug', action="store_true", dest="debug", help='Output debug info', default=False)
parser.add_argument('--heat', action="store_true", dest="heat", help='Use the heat demand assuming 100 percent efficiency. Alterative is to use the resistive heat outpput file based on 2018 heat pumps proportions', default=False)
parser.add_argument('--heat_electric', action="store", dest="heat_electric", help='Proportion of heat in electricity demand, default=0.06', type=float, default=0.06)

args = parser.parse_args()

output_dir = "/home/malcolm/uclan/output/new/baseline/"
if not os.path.isdir(output_dir):
    print('Error output file {} does not exist'.format(output_file))
    quit()
    
if args.reference < '2009' or args.reference > '2020':
    print('Error reference year {} out of range'.format(args.reference))
    quit()

# print arguments
print('Reference year {} filename {} heat {}'.format(args.reference, args.filename, args.heat) )

# read historical electricity demand for reference year
# (power in MW for each hour of the year)

demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
original_demand = readers.read_espeni(demand_filename, args.reference)

# convert to TWh
original_demand = original_demand * 1e-6

# input assumptions for reference year
heat_that_is_electric = args.heat_electric

# read reference year electric heat series based on purely resistive heating
# so that it can be removed from the reference year series. 
 
# use heat demand to create baseline
if args.heat:
    # file name contructed from:
    # B    - the BDEW method
    # rhpp - the rhpp hourly (heat pump) profile
    file_base = 'Brhpp'
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(args.reference, args.reference, file_base)
    demand = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
    ref_resistive_heat = demand['heat']
# use resistive electricity demand
else:
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{0:}Weather{0:}I-Bbdew_resistive.csv'.format(args.reference) 
    ref_resistive = readers.read_copheat(demand_filename, ['electricity', 'temperature'])
    ref_resistive_heat = ref_resistive['electricity']
#   ref_temperature = ref_resistive['temperature']

# convert to TWh
ref_resistive_heat = ref_resistive_heat * 1e-6

#  To remove existing space and water heating from the electricity demand time 
#  series for the reference year - subtract the resistive heat series
baseline_demand = original_demand - (ref_resistive_heat * heat_that_is_electric)

original_annual_demand = original_demand.sum()
baseline_annual_demand = baseline_demand.sum()
print('Annual Demand: original {:.2f} TWh baseline {:.2f} Twh '.format(original_annual_demand, baseline_annual_demand))

# output the file
output_file = output_dir + args.filename + 'Y' + args.reference + '.csv'
baseline_demand.to_csv(output_file, index_label='time', header=['demand_twh'])

# output settings file
settings = {
    'heat'           : args.heat,
    'heat_electric'  : args.heat_electric,
    'reference'      : args.reference
}
settings_df = pd.DataFrame.from_dict(data=settings, orient='index')
output_file = output_dir + args.filename + 'Y' + args.reference + '.settings.csv'
settings_df.to_csv(output_file, header=False)
