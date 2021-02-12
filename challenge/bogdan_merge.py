# python script to merge bogdan solution file into mine

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

# process command line

parser = argparse.ArgumentParser(description='Merge soution files.')
parser.add_argument('bogdan', help='bogdan solution file name')
parser.add_argument('mine', help='my solution file name')

args = parser.parse_args()

# read in the data
input_dir = "/home/malcolm/uclan/challenge/output/"
bodgan_filename = input_dir + args.bogdan
my_filename = input_dir + args.mine

print('Merging bogdan {} with mine {}'.format(bodgan_filename, my_filename) )

# read bogdan file
bogdan = pd.read_csv(bodgan_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(bogdan)

# read my file
mine = pd.read_csv(my_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(mine)
for index, value in bogdan.iteritems():
    k = utils.index2k(index)
    if k>31:
        mine[index] = bogdan[index]

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = output_dir + 'merged_solution.csv'

#mine.index.rename('datetime', inplace=True)
mine.to_csv(output_filename, float_format='%g')
