import argparse
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from datetime import datetime
from datetime import timedelta
from datetime import date
import numpy as np
import matplotlib.pyplot as plt

def test_stationarity(timeseries):
    
    #Determing rolling statistics
#   rolmean = pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(window=365*48, center=False).mean()
#   rolstd = pd.rolling_std(timeseries, window=12)#Plot rolling statistics:
    rolstd = timeseries.rolling(window=365*48, center=False).std()
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

#test_stationarity(df['demand'])

#df = df.asfreq('30m')

ts_log = np.log(df['demand'])
#rolmean = pd.Series(ts_log).rolling(window=365*48, center=False).mean()
#plt.plot(ts_log)
#plt.show()
decomposition = seasonal_decompose(ts_log, period=48*365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

