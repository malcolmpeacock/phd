# plot the population mapped onto the weather grid.

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# from mpl_toolkits.basemap import Basemap
# import cartopy.crs as ccrs
import matplotlib.pyplot as plt

file = '/home/malcolm/uclan/tools/python/interim/population_GB'
file = '/home/malcolm/uclan/tools/python/interim/population_GB_NO_NI'

s = pd.read_pickle(file)
s.to_csv('population_GB.csv')
gdf = gpd.GeoDataFrame(s, columns=['TOT_P'])
gdf['geometry'] = gdf.index.map(lambda i: Point(reversed(i)))
gdf.plot(column='TOT_P', legend=True)
# plt.legend(loc='upper right')
plt.show()
