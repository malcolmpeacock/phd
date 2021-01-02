# crude thermal model of uk heat demand using thermal loss derived
# from my spreadsheet.
import sys
import pandas as pd
from datetime import datetime
import pytz

# read the temperatures. 
def read_gas_temp(filename):
    temp = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%d/%m/%Y').replace(tzinfo=pytz.timezone('Europe/London')), index_col=0, squeeze=True, usecols=[1,3] )
    temp = temp.astype('float')
    # reverse it (december was first! )
    temp = temp.iloc[::-1]
    # get rid of time so we just have a date
    temp.index = pd.DatetimeIndex(pd.to_datetime(temp.index).date)
    return temp

def read_temp(filename):
    temp = pd.read_csv(filename, header=0, sep=',', parse_dates=[0], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d').replace(tzinfo=pytz.timezone('Europe/London')), index_col=0, squeeze=True )
    temp = temp.astype('float')
    print(temp.head())
    # get rid of time so we just have a date
#   temp.index = pd.DatetimeIndex(pd.to_datetime(temp.index).date)
    return temp

# main program
year = '2018'
# base_temp = 15.5
base_temp = 19.0
# base_temp = 12.8
#temp_filename = '/home/malcolm/uclan/data/DailyTempGasExplorer' + year + '.csv'
temp_filename = '/home/malcolm/uclan/tools/python/output/avtemp/' + year + '.csv'
output_file = '/home/malcolm/uclan/tools/python/output/thermal/test' + year + '.csv'
# heat loss for uk housing stock (watts) from spreadsheet.
heat_loss = 8656802139.0
# number of hours of heating
heat_hours = 8.0
# multiply by hours to convert watts to watt hours
heat_loss = heat_loss * heat_hours   

# read temperature data
temp = read_temp(temp_filename)
print(temp)
# TODO this is just domestic space heating - need water
demand = base_temp - temp
demand = demand.clip(lower=0) * heat_loss / 1000000

print(demand)
total = demand.sum()
print(' Total Demand {}'.format(total) )
# Timestamp
#   index = pd.DatetimeIndex(demand.index)
#   demand.index = index.strftime('%Y-%m-%dT%H:%M:%SZ')

demand.to_csv(output_file, sep=',', decimal='.', float_format='%g')
