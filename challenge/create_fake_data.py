# python script to create fake input data for the data challenge.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math

# custom code
import utils

# main program

# data
irradiance = []
power=[]
temp=[]
for day in range(14):
    for k in range(48):
        p=0.5
        if k>30:
            p=0.0
        ir = k*48.0 - k*k + 10.0 -20.0*p
        irradiance.append(ir)
        power.append(ir*0.5)
        temp.append(ir*0.2)

data = {'irradiance_Wm-2' : irradiance, 'pv_power_mw': power, 'panel_temp_C': temp }
pv = pd.DataFrame(data)
    
start_time = '2018-07-09 00:00:00'
end_time = '2018-07-22 23:30:00'

pv.index = pd.date_range( start_time, end_time, freq='30min')
pv.index.name = 'datetime'

print(pv)

pv['irradiance_Wm-2'].plot(label='irradiance', color='blue')
pv['pv_power_mw'].plot(label='power', color='red')
pv['panel_temp_C'].plot(label='temp', color='green')
plt.title('Fake data')
plt.xlabel('Hour of the year', fontsize=15)
plt.ylabel('fake data', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

input_dir = "/home/malcolm/uclan/challenge/input/"
output_filename = input_dir + 'pv_train_fake.csv'
pv.to_csv(output_filename, float_format='%.2f')

for location in range(6):
    temp_name = 'temp_location' + str(location+1)
    solar_name = 'solar_location' + str(location+1)
    pv[temp_name] = pv['panel_temp_C'] * 0.5
    pv[solar_name] = pv['irradiance_Wm-2']

weather = pv.drop(['irradiance_Wm-2', 'pv_power_mw', 'panel_temp_C'], axis=1)
# weather is hourly
weather = weather.resample('H', axis=0).mean()
# copy the first week
start_time = '2018-07-09 00:00:00'
end_time =   '2018-07-15 23:00:00'
weather_next_week = weather[start_time : end_time]
# relabel it to the week after
start_time = '2018-07-23 00:00:00'
end_time =   '2018-07-29 23:00:00'
weather_next_week.index = pd.date_range( start_time, end_time, freq='H')
weather_next_week.index.name = 'datetime'
weather = weather.append(weather_next_week)
print(weather)
output_filename = input_dir + 'weather_train_fake.csv'
weather.to_csv(output_filename, float_format='%.2f')
