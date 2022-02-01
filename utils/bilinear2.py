# Bilinear interpolation
# For getting the value of a different point from a wind , pv grid

# Python modules
import math
from scipy import interpolate
import pandas as pd

# If one of the points contains NaNs as happens with the Adverse
# weather files because we sometimes need a grid point in the sea which
# does not have PV values then set to one which is OK
def replace_nans(df_in, dfs):
    print(df_in)
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

def bilinear(request_wind, request_pv, df, variable):

    # Get unique wind and pv values
    wind_values = df['f_wind'].unique()
    pv_values = df['f_pv'].unique()
#   print(wind_values)
#   print(pv_values)
  
    # check value is within range
    if request_pv < pv_values.min() or request_pv > pv_values.max() or request_wind < wind_values.min() or request_wind > wind_values.max():
        print('Out of range wind {} pv {}'.format(request_wind, request_pv))
        return float("NaN")

    # get latitude of grid point below location ( box_wind_min )
    box_wind_min = wind_values[wind_values <= request_wind].max()
    # get latitude of grid point above location ( box_wind_max )
    box_wind_max = wind_values[wind_values >= request_wind].min()
#   print('Wind min {} max {}'.format(box_wind_min,box_wind_max))
    # get longitude of grid point west of location ( box_pv_min )
    box_pv_min = pv_values[pv_values <= request_pv].max()
    # get longitude of grid point east of location ( box_pv_max )
    box_pv_max = pv_values[pv_values >= request_pv].min()
#   print('PV min {} max {}'.format(box_pv_min,box_pv_max))

    t_min_min = df[(df['f_wind']==box_wind_min) & (df['f_pv']==box_pv_min)]
#   print("t_min_min")
#   print(t_min_min)
    t_min_max = df[(df['f_wind']==box_wind_min) & (df['f_pv']==box_pv_max)]
#   print("t_min_max")
#   print(t_min_max)
    t_max_min = df[(df['f_wind']==box_wind_max) & (df['f_pv']==box_pv_min)]
#   print("t_max_min")
#   print(t_max_min)
    t_max_max = df[(df['f_wind']==box_wind_max) & (df['f_pv']==box_pv_max)]
#   print("t_max_max")
#   print(t_max_max)

    # check there are values
    if len(t_min_min)==0 or len(t_min_max)==0 or len(t_max_min)==0 or len(t_max_max)==0:
        print('No values found wind {} pv {}'.format(request_wind, request_pv))
        return float("NaN")

    v_min_min = t_min_min[variable].values[0]
    v_min_max = t_min_max[variable].values[0]
    v_max_min = t_max_min[variable].values[0]
    v_max_max = t_max_max[variable].values[0]
#   print(v_min_min, v_min_max, v_max_min, v_max_max)

#   t_min_min = replace_nans(t_min_min, [t_min_max, t_max_min, t_max_max])
#   t_min_max = replace_nans(t_min_max, [t_min_min, t_max_min, t_max_max])
#   t_max_min = replace_nans(t_max_min, [t_min_max, t_min_min, t_max_max])
#   t_max_max = replace_nans(t_max_max, [t_min_max, t_max_min, t_min_min])

    y = [box_wind_min,box_wind_max,box_wind_min,box_wind_max,request_wind]
    x = [box_pv_min,box_pv_max,box_pv_max,box_pv_min,request_pv]

    d_min_min = distance(box_wind_min, box_pv_min, request_wind, request_pv) 
    d_min_max = distance(box_wind_min, box_pv_max, request_wind, request_pv) 
    d_max_min = distance(box_wind_max, box_pv_min, request_wind, request_pv) 
    d_max_max = distance(box_wind_max, box_pv_max, request_wind, request_pv) 
    d_total = d_min_min + d_min_max + d_max_min + d_max_max
    f_min_min = d_min_min / d_total
    f_min_max = d_min_max / d_total
    f_max_min = d_max_min / d_total
    f_max_max = d_max_max / d_total
#   t_weighted = t_min_min * f_min_min + t_min_max * f_min_max + t_max_min * f_max_min + t_max_max * f_max_max
#   print('t_min_min')
#   print(t_min_min)
#   print('t_weighted')
#   print(t_weighted)
#   t_min_min.rename('t_min_min', inplace=True)
#   t_min_max.rename('t_min_max', inplace=True)
#   t_max_min.rename('t_max_min', inplace=True)
#   t_max_max.rename('t_max_max', inplace=True)
#   t_weighted.rename('t_weighted', inplace=True)
    # blinear interpolation
    f_min_min = (box_wind_max - request_wind) * (box_pv_max - request_pv)
    f_max_min = (request_wind - box_wind_min) * (box_pv_max - request_pv)
    f_min_max = (box_wind_max - request_wind) * (request_pv - box_pv_min)
    f_max_max = (request_wind - box_wind_min) * (request_pv - box_pv_min)
    t_interp = (v_min_min * f_min_min + v_max_min * f_max_min + v_min_max * f_min_max + v_max_max * f_max_max ) / ( (box_wind_max - box_wind_min) * (box_pv_max - box_pv_min) + 0.0 )
#   t_interp.rename('t_interp', inplace=True)
#   print(t_interp)
#   df = pd.concat([t_min_min, t_min_max, t_max_min, t_max_max, t_interp], axis=1, names=['t_min_min', 't_min_max', 't_max_min', 't_max_max', 't_interp'])

#   print(df)
#   return t_interp, x, y
    return t_interp
