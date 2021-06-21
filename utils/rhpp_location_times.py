# python script to: read in the rhpp stats.csv and plot number of houses
#   against day

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import glob
import os.path

# custom code
import stats
import readers

# main program

output_dir = "/home/malcolm/uclan/data/rhpp-heatpump/testing/"
#df = pd.read_csv(output_dir + 'stats.csv')
df = pd.read_csv(output_dir + 'stats.csv', index_col=1, parse_dates=[2,3], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
print(df)
start = df['start'].min()
end = df['end'].max()
whole_period = pd.date_range(start = start, end = end, freq='D' )
print(whole_period)
counts=[]
for period in whole_period:
    started = df[df['start'] < period]
#   print(started)
    going = started['end'] > period
#   print(going)
    counts.append(going.values.sum())
#print(counts)
s = pd.Series(counts, index=whole_period)
print(s)

# plot
s.plot()
plt.title('RHPP Heat pump trial')
plt.ylabel('Monitored properties', fontsize=15)
plt.xlabel('Day of the year')
plt.show()

# plot a single line showing when each house was monitored
df.sort_values(by='start', ascending=True, inplace=True, axis=0)
df.reset_index(drop=True, inplace=True)
print(df)
for index, row in df.iterrows():
    x = [ row['start'], row['end'] ]
    y = [ index, index ]
    plt.plot(x,y)
plt.ylabel('House number', fontsize=15)
plt.xlabel('Day of the year')
plt.title('RHPP Heat pump trial - monitoring periods')
plt.show()
