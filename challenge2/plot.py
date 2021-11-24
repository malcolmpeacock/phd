
# contrib code
import sys
import pandas as pd
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def regression(X, y):
    estimator = LinearRegression()
#   model = make_pipeline(PolynomialFeatures(1),estimator)
#   fit = model.fit(X, y)
    poly = PolynomialFeatures(1)
    pf = poly.fit_transform(X)
    fit = estimator.fit(pf, y)
    coeffs = estimator.coef_
    print(fit.score(pf,y))
    print(estimator.coef_)
    print(estimator.intercept_)
#   p = fit.predict(Xp)
    return coeffs[1], coeffs[2]

# main program

# process command line
parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('--split', action="store_true", dest="split", help='Do a regression to split power sources', default=False)
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

# plot the diffs demand and max min
ax1 = df_in['solar_irradiance1'].plot(color='red',label='irradiance')
plt.ylabel('Solar Irradiance', fontsize=15, color='red')
# 2nd axis
ax2 = ax1.twinx()
ax2.set_ylabel('Demand (MWh)',color='black', fontsize=15)
max_diff = df_out['max_demand'] - df_in['demand']
min_diff = df_in['demand'] - df_out['min_demand']
max_diff.plot(ax=ax2, label='max_demand diff', color='green')
min_diff.plot(ax=ax2, label='min_demand diff', color='blue')
plt.title('min and max demand diffs')
plt.xlabel('Half Hour of the month', fontsize=15)
plt.ylabel('Demand (MW)', fontsize=15)
plt.legend(loc='lower left', fontsize=15)
plt.show()

# plot the demand and max min
df_out['max_demand'].plot(label='max_demand')
df_out['min_demand'].plot(label='min_demand')
df_in['demand'].plot(label='half hourly demand')
plt.title('min and max demand')
plt.xlabel('Half Hour of the month', fontsize=15)
plt.ylabel('Demand (MW)', fontsize=15)
plt.legend(loc='lower left', fontsize=15)
plt.show()

if args.split:
    c1, c2 = regression(df_in[['solar_irradiance1', 'windspeed1']], df_in['demand'])
    pv = df_in['solar_irradiance1'] * c1 * -1.0
    wind =  df_in['windspeed1'] * c2 * -1.0

    base = df_in['demand'] + pv + wind

    ax1 = df_in['solar_irradiance1'].plot(color='red',label='irradiance')
    plt.ylabel('Solar Irradiance', fontsize=15, color='red')
    # 2nd axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Demand (MWh)',color='black', fontsize=15)

#   df_in['demand'].plot(ax=ax2,color='blue',label='half hourly demand')
    base.plot(ax=ax2,color='blue',label='half hourly demand')
    pv.plot(ax=ax2,color='green',label='pv')
    wind.plot(ax=ax2,color='yellow',label='wind')
    plt.title('Seperate out the PV')
    plt.xlabel('Half Hour of the month', fontsize=15)
    plt.ylabel('Demand (MW)', fontsize=15)
    plt.legend(loc='lower left', fontsize=15)
    plt.show()
