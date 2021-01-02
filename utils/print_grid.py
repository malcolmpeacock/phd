# Print out points from a pickle file of the weather grid.

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

file = '/home/malcolm/uclan/tools/python/scripts/heat/interim/population_GB'

s = pd.read_pickle(file)
print(s)
count=0
for items in s.iteritems():
    latitude=items[0][0]
    longitude=items[0][1]
    print(latitude, longitude)
