# python script to compare 2 csv time series files 
#   usage:  python comparecsv.py file1 file2
import sys
from pandas import read_csv
import statsmodels.api as sm
import stats

# get command line.

file1 = sys.argv[1]
file2 = sys.argv[2]
field = sys.argv[3]

print(file1, file2)

data1 = read_csv(file1, header=0, sep=',', decimal='.', parse_dates=[0], index_col=0, squeeze=True, usecols=['time','space','heat',field] )
data2 = read_csv(file2, header=0, sep=',', decimal='.', parse_dates=[0], index_col=0, squeeze=True, usecols=['time','space','heat',field] )

print(data1.head(7))
print(data2.head(7))

stats.print_stats_header()
stats.print_stats(data1[field], data2[field], 'oldVnew')
