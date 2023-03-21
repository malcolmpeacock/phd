# python script to plot COP vs DELTAt curves.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
# custom code
import stats

# main program

# read in the data
filename = "/home/malcolm/uclan/data/heatpumps/COPvsTemp.csv"
data = pd.read_csv(filename, header=0, sep=',', index_col=0)

# output plots

data['ASHP_S'].plot(label='ASHP Staffell', color='blue')
data['GSHP_S'].plot(label='GSHP Staffell', color='blue', linestyle='dotted')
data['ASHP_R'].plot(label='ASHP Ruhnau', color='red')
data['GSHP_R'].plot(label='GSHP Ruhnau', color='red', linestyle='dotted')
data['ASHP_F'].plot(label='ASHP Fischer', color='green')
data['GSHP_F'].plot(label='GSHP Fischer', color='green', linestyle='dotted')
data['ASHP_P'].plot(label='ASHP RHPP', color='yellow')
data['GSHP_P'].plot(label='GSHP RHPP', color='yellow', linestyle='dotted')
data['COP'].plot(label='ASHP Kelly', color='purple')
plt.title('Heat Pump COP vs Delta T')
plt.xlabel('Temperature Difference (degrees C)', fontsize=15)
plt.ylabel('Coefficient of Performance', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()


