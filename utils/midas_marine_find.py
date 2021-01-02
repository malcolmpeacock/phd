# python script to read midas marine data file
# and filter it to contain locations with more than 8000 hours
# and find the records closest to a given latitude and longitude

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import math

# custom code
import stats
import readers

# main program
# midas_filename = '/home/malcolm/uclan/data/midas/marine/midas_marine-obs-lon-band-f_201801-201812.txt'
midas_filename = '/home/malcolm/uclan/data/midas/marine/midas_marine-obs-lon-band-f_200901-200912.txt'
#midas_filename = '/home/malcolm/uclan/data/midas/marine/midas_marine-obs-lon-band-f_198501-198512.txt'
# Amsterdam offshore mast
# longitude = 4.39
# latitude = 52.61
# my offshore location from midas data in 2018
#longitude = 1.8
#latitude = 57.0
# london array
longitude = 1.39
latitude = 51.59

closest = 100000000.0
closest_key = (9999999999999.0, 9999999999999.0)

print('Searching for lat {} lon {} in {}'.format(latitude,longitude,midas_filename) )

midas = readers.read_midas_marine(midas_filename)
#print(midas)
# stationary
# midas = midas[midas['shipdir'] == '0']
groups = midas.groupby(['longitude', 'latitude']).groups
#print(groups.keys())
for key in groups.keys():
    if len(groups[key]) > 8000:
        distance = math.sqrt( (key[0] - longitude)**2 + (key[1] - latitude)**2)
        if distance<closest:
            closest = distance
            closest_key = key
            print('Latitude {} Longitude {} Distance {}'.format(key[1], key[0], distance) )
print('Latitude {} Longitude {} Distance {}'.format(closest_key[1], closest_key[0], closest) )
# find those with 8000 hours or more
quit()
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
quit()
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
