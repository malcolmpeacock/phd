
# contrib code
import sys
import pandas as pd
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

# main program

# process command line
parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Plots ', default=False)
args = parser.parse_args()

# read in the data
output_dir = "/home/malcolm/uclan/challenge2/output/"
# merged data file ( demand, weather, augmented variables )
merged_filename = '{}merged_pre_august.csv'.format(output_dir)
df_in = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(df_in)
# merged data file ( demand, weather, augmented variables )
maxmin_filename = '{}maxmin_pre_august.csv'.format(output_dir)
df_out = pd.read_csv(maxmin_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

mmdiff = (df_out['max_demand'] + df_out['min_demand']) / 2

if args.plot:
    mmdiff.plot(label='Average of max and min demand')
    df_in['demand'].plot(label='Half hour demand')
    plt.title('Demand and max min diff')
    plt.xlabel('Half Hour of the month', fontsize=15)
    plt.ylabel('Demand', fontsize=15)
    plt.legend(loc='lower left', fontsize=15)
    plt.show()


sqrs = (df_in['demand'] - mmdiff).pow(2)
rmse = math.sqrt(sqrs.sum() )

# Regression Rsquared.
model = sm.OLS(df_in['demand'].to_numpy(), mmdiff.to_numpy())
results = model.fit()
p = results.params
gradient = p[0]
rsquared = results.rsquared

print('RMSE {} Rsquared {}'.format(rmse, rsquared))

