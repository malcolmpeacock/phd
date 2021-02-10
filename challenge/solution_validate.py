# python script to validate solution file for the pod challenge.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math

# custom
import utils

# main program

solution = sys.argv[1]

# read in the data
input_dir = "/home/malcolm/uclan/challenge/output/"
filename = input_dir + solution
print('***Validating File {}'.format(filename) )
s = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

#print(s)

# no NaN's
nans = s.isna().sum()
if nans >0:
    print('FAIL {} nans'.format(nans) )

# 1 weeks worth of values
if len(s) != 48 * 7:
    print('FAIL wrong number of values')

count=0
store=0.0
errors=0

# For each day ...
days = s.resample('D', axis=0).mean().index
for day in days:
#   print(day)
    s_day = s[day : day + pd.Timedelta(hours=23,minutes=30)]
#   print(s_day)

    # for each period of the day ...
    for index, value in s_day.items():
        count+=1
        # -2.5 <= B <= 2.5
        if value<-2.5 or value>2.5:
            print('Value {} out of range at line {}'.format(value, count) )
            errors+=1
        # period
        k = utils.index2k(index)
        # B <= 0 if period >= 32
        if k>31 and k<43 and value >0:
            print('Charging at the wrong time: Value {} period {} at line {}'.format(value, k, count) )
            errors+=1
        # B >= 0 if period <= 31
        if k<32 and value <0:
            print('Discharging at the wrong time: Value {} period {} at line {}'.format(value, k, count) )
            errors+=1
        # battery storage
        store = store + (0.5 * value)
        if store <0 or store >6:
            print('Store out of range: store {} day {} period {} at line {}'.format(store, day, k, count) )
            errors+=1
            quit()

if errors==0:
    print("PASSED")
else:
    print("Failed {} errors".format(errors) )
