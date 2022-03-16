import xarray as xr
import pandas as pd

#filename = "/home/malcolm/uclan/output/correlation/weather/ERA1I_weather_2018.nc"
#filename = "/home/malcolm/uclan/data/ukcp18/tas_gwl4_land-derived_uk_n216_01_mon_30010101-30501230.nc"
filename = "/home/malcolm/uclan/data/ukcp18/2022-03-15T15-45-23.nc"

ds = xr.open_dataset(filename)
df = ds.to_dataframe()
print(df.columns)
#print(df)
wanted = ['tas', 'latitude', 'longitude', 'yyyymm']
smaller = df[wanted]
print(smaller['yyyymm'])

print(smaller[ smaller['yyyymm']=='198004'])
