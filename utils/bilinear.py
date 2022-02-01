# Bilinear interpolation
# For getting the value of a different point from the ERA5 grid

# Python modules
import math
from scipy import interpolate
import pandas as pd

# If one of the points contains NaNs as happens with the Adverse
# weather files because we sometimes need a grid point in the sea which
# does not have PV values then set to one which is OK
def replace_nans(df_in, dfs):
    if df_in.isna().sum() > 0:
        for df in dfs:
            if df.isna().sum() == 0:
                print("Replace NANS in")
                print(df_in)
                print("WITH")
                print(df)
                return df.copy()
    
    return df_in
    

def distance(x1,y1,x2,y2):
    d = math.sqrt( ( x1-x2 )**2 + ( y1-y2)**2 )
    return d

def bilinear(location_latitude, location_longitude, df):

    # Get axis 1 which is a multilevel index
    points = df.axes[1]
    latitudes = points.get_level_values('latitude').unique().values
    print(latitudes)
    # get latitude of grid point below location ( box_lat_min )
    box_lat_min = latitudes[latitudes < location_latitude].max()
    # get latitude of grid point above location ( box_lat_max )
    box_lat_max = latitudes[latitudes > location_latitude].min()
    print(box_lat_min,box_lat_max)
    longitudes = points.get_level_values('longitude').unique().values
    print(longitudes)
    # get longitude of grid point west of location ( box_long_min )
    box_long_min = longitudes[longitudes < location_longitude].max()
    # get longitude of grid point east of location ( box_long_max )
    box_long_max = longitudes[longitudes > location_longitude].min()
    print(box_long_min,box_long_max)

    t_min_min = df[(box_lat_min,box_long_min)]
    t_min_max = df[(box_lat_min,box_long_max)]
    t_max_min = df[(box_lat_max,box_long_min)]
    t_max_max = df[(box_lat_max,box_long_max)]

    t_min_min = replace_nans(t_min_min, [t_min_max, t_max_min, t_max_max])
    t_min_max = replace_nans(t_min_max, [t_min_min, t_max_min, t_max_max])
    t_max_min = replace_nans(t_max_min, [t_min_max, t_min_min, t_max_max])
    t_max_max = replace_nans(t_max_max, [t_min_max, t_max_min, t_min_min])

    y = [box_lat_min,box_lat_max,box_lat_min,box_lat_max,location_latitude]
    x = [box_long_min,box_long_max,box_long_max,box_long_min,location_longitude]

    d_min_min = distance(box_lat_min, box_long_min, location_latitude, location_longitude) 
    d_min_max = distance(box_lat_min, box_long_max, location_latitude, location_longitude) 
    d_max_min = distance(box_lat_max, box_long_min, location_latitude, location_longitude) 
    d_max_max = distance(box_lat_max, box_long_max, location_latitude, location_longitude) 
    d_total = d_min_min + d_min_max + d_max_min + d_max_max
    f_min_min = d_min_min / d_total
    f_min_max = d_min_max / d_total
    f_max_min = d_max_min / d_total
    f_max_max = d_max_max / d_total
    t_weighted = t_min_min * f_min_min + t_min_max * f_min_max + t_max_min * f_max_min + t_max_max * f_max_max
    print(t_min_min)
    print(t_weighted)
    t_min_min.rename('t_min_min', inplace=True)
    t_min_max.rename('t_min_max', inplace=True)
    t_max_min.rename('t_max_min', inplace=True)
    t_max_max.rename('t_max_max', inplace=True)
    t_weighted.rename('t_weighted', inplace=True)
    # blinear interpolation
    f_min_min = (box_lat_max - location_latitude) * (box_long_max - location_longitude)
    f_max_min = (location_latitude - box_lat_min) * (box_long_max - location_longitude)
    f_min_max = (box_lat_max - location_latitude) * (location_longitude - box_long_min)
    f_max_max = (location_latitude - box_lat_min) * (location_longitude - box_long_min)
    t_interp = (t_min_min * f_min_min + t_max_min * f_max_min + t_min_max * f_min_max + t_max_max * f_max_max ) / ( (box_lat_max - box_lat_min) * (box_long_max - box_long_min) + 0.0 )
    t_interp.rename('t_interp', inplace=True)
#   print(t_interp)
    df = pd.concat([t_min_min, t_min_max, t_max_min, t_max_max, t_weighted, t_interp], axis=1, names=['t_min_min', 't_min_max', 't_max_min', 't_max_max', 't_weighted', 't_interp'])

#   print(df)
    return df, x, y
