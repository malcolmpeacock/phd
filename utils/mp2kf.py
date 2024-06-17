# python script to add columns for year, month, day, hour

# library stuff
import sys
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description='Convert csv to Katerina format.')
parser.add_argument('filename', help='filename')
args = parser.parse_args()

# main program

input_filename = '/home/malcolm/uclan/output/timeseries2/mp_files/{}.csv'.format(args.filename)
output_filename = '/home/malcolm/uclan/output/timeseries2/kf_files/{}.csv'.format(args.filename)
df = pd.read_csv(input_filename, header=0, parse_dates=[0], index_col=0 )
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour
df.to_csv(output_filename)

