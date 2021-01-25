# combine clnr files into 1 useful one
import glob
import os.path
import pandas as pd
from datetime import datetime

def clnr_read(filename,variable):
    clnr = pd.read_csv(filename, sep=',', header=0, parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M:%S'),index_col=0, squeeze=True)
#   print(clnr)
    if variable == 'home_power' or variable == 'pump_power':
        clnr = clnr.resample('H').sum()
    else:
        clnr = clnr.resample('H').mean()
    return clnr.rename(variable)

clnr_filename = "TemperatureData.csv"
clnr_dir = "/home/malcolm/uclan/data/CLNR-L082-TC12-dataset-March-15/TC3"
clnr_output = clnr_dir + '/processed'

locations={}


# glob to locations list
for name in glob.glob(clnr_dir + '/files/*'):
    filename = os.path.basename(name)
    parts = filename.split('_')
    location = parts[0]
    variable = parts[1][:-4]
    print(filename, location, variable)
    if location in locations:
        locations[location][variable] = name
    else:
        locations[location] = { variable : name }


print(locations)
# for each loation ...
for location in locations:
    # if all the files exist 
#   if 'pump' in locations[location] and 'home' in locations[location] and 'tempout' in locations[location] and 'tempin' in locations[location] :
    if 'pump' in locations[location] and 'home' in locations[location]:
        print('Location {} all'.format(location) )
        df1 = clnr_read(locations[location]['pump'],'pump_power')
        df2 = clnr_read(locations[location]['home'],'home_power')
#       df3 = clnr_read(locations[location]['tempout'],'temperature_outside')
#       df4 = clnr_read(locations[location]['tempin'],'temperature_inside')
#       df = pd.concat([df1, df2, df3, df4],axis=1)
        df = pd.concat([df1, df2],axis=1)
        print(df)
        nan_pump_power = df['pump_power'].isna().sum()
        nan_home_power = df['home_power'].isna().sum()
#       nan_temperature_outside = df['temperature_outside'].isna().sum()
#       nan_temperature_inside = df['temperature_inside'].isna().sum()
#       print('NaN pump_power {} home_power {} temperature_outside {} temperature_inside {}'.format(nan_pump_power,nan_home_power, nan_temperature_outside, nan_temperature_inside) )
        print('NaN pump_power {} home_power {} '.format(nan_pump_power,nan_home_power ) )
        df = df.interpolate()
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        print('NaN pump_power {} home_power {} '.format(nan_pump_power,nan_home_power ) )
        df.to_csv(clnr_output + '/' + location + '.csv')
    else:
        print('Location {} *missing some'.format(location) )

#   read in, convert to hourly, concatonate to make a df
#    and store in a list of dfs to be concatonated at the end

# output to csv
