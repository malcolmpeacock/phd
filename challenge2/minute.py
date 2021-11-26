# read minutely data .

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MaxAbsScaler
import matplotlib

# read in the data
input_dir = "/home/malcolm/uclan/challenge2/data/"
output_dir = "/home/malcolm/uclan/challenge2/output/"

# read minute data.
minute_filename = 'MW_Staplegrove_CB905_MW_minute_real_power_MW_pre_august.csv'
df = pd.read_csv(input_dir+minute_filename, header=0, sep=',', parse_dates=[1], index_col=1, usecols=['quality', 'time', 'value'], squeeze=True)
#df = pd.read_csv(input_dir+minute_filename, header=0, sep=',', parse_dates=[0], index_col=0, usecols=['time', 'value'], squeeze=True)
print(df)
# TODO this does not work
#df = df[df['quality'] == 'Good']
#print(df)
df = df.drop('quality', 1)
df = df.interpolate()
print(df)
df = df['value']
print(df)

df = df.resample('30T').agg(['min', 'max', 'std', 'var', 'sem'])
# sanity check
print(df.columns)
for col in df.columns:
    if df[col].isna().sum() >0:
        print("ERROR nans in {}".format(col))
        print(df[df[col].isnull()])
        quit()

df.columns = ['max_demand', 'min_demand', 'std', 'var', 'sem']
print(df)
# output correlation values
output_filename = 'minutely.csv'
df.to_csv(output_dir+output_filename, float_format='%.8f')
