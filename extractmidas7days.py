# Program to extract 7 days of hourly midas irradiance data for an exercise.

import pandas as pd
import pytz
import matplotlib.pyplot as plt
from utils.readers import read_midas_irradiance

# read midas
midas_file = "/home/malcolm/uclan/data/midas/midas-open_uk-radiation-obs_dv-201908_cornwall_01395_camborne_qcv-1_2018.csv"
pv_hourly = read_midas_irradiance(midas_file,['glbl_irad_amt'])
# convert from kJ/m2 to wh/m2
# pv_hourly = pv_hourly * 0.2777777777
pv_day = pv_hourly['2018-06-01 00:00:00' : '2018-06-07 23:00:00']
print(pv_day)
df = pv_day
output_file = "/home/malcolm/uclan/output/pv/cambourne.csv";
df.to_csv(output_file, sep=',', decimal='.', float_format='%g', date_format='%Y-%m-%d %H:%M:%S')
