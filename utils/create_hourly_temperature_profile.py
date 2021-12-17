# Use the 2018 hourly temperature to create a temperature profile from the
# mean daily temperature based on 5 degree ranges.

import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse

# custom code
import stats
import readers

def get_lower(x):
    return x.right

# process command line
parser = argparse.ArgumentParser(description='Create an hourly profile for UK temperature')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--one', action="store_true", dest="one", help='Use one bin only', default=False)
parser.add_argument('--diff', action="store_true", dest="diff", help='Use temperature difference', default=False)
args = parser.parse_args()

# read hourly weather data
year = '2018'
weather_filename = '/home/malcolm/uclan/output/wparms/weather_parms{}.csv'.format(year)
weather = pd.read_csv(weather_filename, header=0, parse_dates=[0], index_col=0 )
print(weather)

# read the wind and temperature columns, but then overwrite the wind with the
# mean daily temperature of the day
ref_temperature = weather[['wind', 'temp']].copy()
ref_temperature['temp'] = ref_temperature['temp'] - 273.15
ref_temperature.columns = ['dailytemp', 'temperature']
print(ref_temperature)
daily_temp = ref_temperature['temperature'].resample('D', axis=0).mean()
days = pd.Series(ref_temperature.index.date).unique()
for day in days:
    day_str = day.strftime('%Y-%m-%d')

    # daily temperature
    ref_temperature.loc[day_str+' 00:00:00' : day_str+' 23:30:00','dailytemp'] = daily_temp.loc[day_str+' 00:00:00']

# add hour of the day and remove the index
ref_temperature['hour'] = ref_temperature.index.hour
ref_temperature.reset_index(drop=True, inplace=True)

print(ref_temperature)

# split the data into a range of temperature bins at 5 degree intrevals
range_start = math.floor(ref_temperature['dailytemp'].min())
range_end = math.ceil(ref_temperature['dailytemp'].max())
bins = pd.cut(ref_temperature['dailytemp'], range(range_start,range_end,5) )
print(bins)
profiles = ref_temperature.groupby([bins, 'hour'])['temperature'].mean()

# plot 
if args.plot:
    curves = profiles.index.unique(level = 'dailytemp')
    for curve in curves:
        print(curve)
#       n_factor = profiles[curve].max() - profiles[curve].min()
#       normal_curve = (profiles[curve] - profiles[curve].min() ) / n_factor
#       print(normal_curve)
        normal_curve = profiles[curve]
        plt.plot(normal_curve.index, normal_curve.values, label=curve)

    plt.title('Hourly Temperature profiles')
    plt.xlabel('hour of the day')
    plt.ylabel('temperature ')
    plt.legend(loc='upper right', fontsize=15)
    plt.show()


# output to csv
profiles = profiles.reset_index()
profiles['dailytemp'] = profiles['dailytemp'].apply(get_lower)
profiles.reset_index(drop=True, inplace=True)
#profiles.drop(index=0, inplace=True)
#rofiles.drop_index(inplace=True)
profiles.set_index('dailytemp', inplace=True)

if args.diff:
    ref_temperature['diff'] = ref_temperature['temperature'] - ref_temperature['dailytemp']
    diffs = ref_temperature[['hour','diff']]
    print(diffs)
    profiles = diffs.groupby('hour')['diff'].mean()
    print(profiles)
    if args.plot:
        profiles.plot()
        plt.title('Hourly Temperature profile based on population weighted 2018 UK temperature')
        plt.xlabel('hour of the day')
        plt.ylabel('temperature difference from mean')
        plt.show()

# If using one bin, combine the profiles
if args.one:
    profiles = profiles.groupby('hour')['temperature'].mean()
    # center and normalize
    n_factor = profiles.max() - profiles.min()
    print('Max {} Min {} factor {}'.format( profiles.max(), profiles.min(), n_factor ) )
    profiles = (profiles - profiles.min() ) / n_factor
    print('Max {} Min {} sum {}'.format( profiles.max(), profiles.min(), profiles.sum()) )
    # divide by sum because of 24 hours so they will add up to 1.
    profiles = profiles / profiles.sum()
    print('Sum of temperature profile: {}'.format(profiles.sum() ) )

    if args.plot:
        profiles.plot()
        plt.title('Hourly Temperature profile based on population weighted 2018 UK temperature')
        plt.xlabel('hour of the day')
        plt.ylabel('temperature %')
        plt.show()
    
print(profiles)
#print(profiles.index)
profiles.to_csv('/home/malcolm/uclan/tools/python/scripts/heat/output/adv/temp_profile.csv')
#profiles['lower'] = profiles.index
#print(curves)

#for curve in curves:
