import xarray as xr
import pandas as pd

filename = "/home/malcolm/uclan/output/correlation/weather/ERA1I_weather_2018.nc"

ds = xr.open_dataset(filename)
df = ds.to_dataframe()
print(df)
#for col in df.columns: 
#    print(col) 
print(df.columns)
