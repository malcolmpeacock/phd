# compare my rhpp profile put into the watson 3 degree bands with mine

import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math

# read in my profile in watson bands
filename = "/home/malcolm/uclan/data/rhpp-heatpump/testing/hourly_factors_w.csv"
mine = pd.read_csv(filename, header=0, sep=',', index_col=0, parse_dates=[0], squeeze=True)
print(mine)

# read in watson 
filename = "/home/malcolm/uclan/data/watson2020/ashp.csv"
watson_ashp = pd.read_csv(filename, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=lambda x: datetime.strptime(x, '%H:%M:%S'))
# upscale watson to hourly
watson_ashp = watson_ashp.resample('H').sum()
print(watson_ashp)
filename = "/home/malcolm/uclan/data/watson2020/gshp.csv"
watson_gshp = pd.read_csv(filename, header=0, index_col=0, parse_dates=[0], squeeze=True, date_parser=lambda x: datetime.strptime(x, '%H:%M:%S'))
watson_gshp = watson_gshp.resample('H').sum()
print(watson_gshp)

# combine ashp and gshp in uk proportions
watson = watson_ashp * 0.9 + watson_gshp * 0.1
watson.index = mine.index
print(watson)

for column in watson.columns:
    watson[column].plot(label='watson')
    mine[column].plot(label='mine')

    plt.title('Hourly profile {}'.format(column))
    plt.xlabel('Hour of the day', fontsize=15)
    plt.ylabel('Proportion of heat demand ', fontsize=15)
    plt.legend(loc='upper right')
    plt.show()

# plot to compare

# rsquared for each band.
