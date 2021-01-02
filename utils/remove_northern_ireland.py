# remove points for Northern Ireland from the population mapped onto the 
# weather grid.

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

def not_in_ni(point):
    if point.x < -5.18 and point.y < 55.2 and point.y > 51.0:
        return False
    else:
        return True

file = '/home/malcolm/uclan/tools/python/interim/population_GB'
newfile = '/home/malcolm/uclan/tools/python/interim/population_GB_NO_NI'
polygon = Polygon([(-5.18, 51.0), (-5.18, 55.2), (-12, 55.2), (-12, 51.0), (-5.18, 51.0)])
# poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs={'init': 'epsg:4326'})


s = pd.read_pickle(file)
print(s)
mainland=[]
count=0
for items in s.iteritems():
#   print(items)
    latitude=items[0][0]
    longitude=items[0][1]
#   print(latitude, longitude)
    if longitude < -5.18 and latitude < 55.2 and latitude > 51.0:
        print('In NI {} {}'.format(longitude, latitude) )
    else:
#       mainland.append(count)
        mainland.append(items[0])
    count+=1
# print(mainland)
gb = s.loc[mainland]
# print(s.index)
# new_ind = filter(not_in_ni,s.index)
# print(new_ind)
# result = filter(lambda x: not( x[1] < -5.18 and x[0] < 55.2 and x[0] > 51.0) , s.index) 
# print(list(result)) 
# print(type(s.index))
# ni = s.loc[list(result)]
# g = gpd.GeoDataFrame(geometry=[s.index])
# ni = g.within(polygon)
# ni = s.loc[s.index[1] < -5.18 and s.index[0] < 55.2 and s.index[0] > 51.0]
# ni = s.index.x < -5.18 & s.index.y < 55.2 & s.index.y > 51.0
print(gb)
# print(s.head())
gb.to_pickle(newfile)
