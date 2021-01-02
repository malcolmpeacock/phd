# python script to read midas marine data.
# collect records for a particular latitude and longitude
# and create hourly windspeed data

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

# custom code
import stats
import readers

# main program
midas_filename = '/home/malcolm/uclan/data/midas/marine/midas_marine-obs-lon-band-f_201801-201812.txt'

midas = readers.read_midas_marine(midas_filename)
east = midas['longitude'] > 1.5
filtered = midas[east]
print(filtered)
north = filtered['latitude'] > 54.4
filtered = filtered[north]
print(filtered)
west = filtered['longitude'] < 2.0
filtered = filtered[west]
south = filtered['latitude'] < 59.0
filtered = filtered[south]
print(filtered)
# point1 = filtered.loc['2018-01-01 00:00:00']
# lats = point1['latitude']
# print(lats.nsmallest())
# location = filtered['latitude'] == 57.0 and filtered['longitude'] == 1.8
#
location_lat = 57.0
location_lon = 1.8
location = filtered['latitude'] == location_lat
ldata = filtered[location]
location = ldata['longitude'] == location_lon
ldata = ldata[location]
wind = ldata['windspeed']
# create an index with the missing hours
year = '2018'
d = pd.date_range(start = year + '-01-01 00:00:00', end = year + '-12-31 23:00:00', freq='H' ).difference(wind.index)
newdata={}
for dt in d:
    newdata[dt] = [float("NaN") for i in range(9)]
newseries = pd.DataFrame.from_dict(data=newdata,orient='index',columns=ldata.keys())
# newseries = pd.Series(data=newdata,name='windspeed')
newseries.index=d
print('newseries')
print(newseries)
# df = pd.concat([wind,newseries])
df = pd.concat([ldata,newseries])
df.sort_index(inplace=True)

# interpolate missing values (replace the NaNs)
#   print(df)
df = df.interpolate()
index = pd.DatetimeIndex(data=df.index,name='time')
df.index = index.strftime('%Y-%m-%dT%H:%M:%SZ')
print(df)
print(df.dtypes)
# drop sea_temperature because its not set.
df.drop(columns=['latitude','longitude','version','sea_temperature'],inplace=True)
print(df)
output_dir = '/home/malcolm/uclan/output/wind/'
output_file = '{}marine_lat{:.1f}_long{:.1f}_{}.csv'.format(output_dir, location_lat, location_lon, year)
df.to_csv(output_file, sep=',', decimal='.', float_format='%g')
