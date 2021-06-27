# python script to look at demand data fft.

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
import pvlib
import scipy as sp
import scipy.fftpack

# custom code
import utils

# main program

# process command line

parser = argparse.ArgumentParser(description='Clean weather data.')
parser.add_argument('set', help='weather file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

temp_daily = df['tempm'].resample('D', axis=0).mean()
temp_daily = temp_daily.interpolate()
print(temp_daily.isna().sum())
#temp_daily

temp_daily.plot()
plt.show()

# Fast fourier transform of mean tmeperature
temp_fft = sp.fftpack.fft(temp_daily.values)
#print(temp_fft)
# Power spectoral density
temp_psd = np.abs(temp_fft) ** 2
# get frequecies corresponding to psd
fftfreq = sp.fftpack.fftfreq(len(temp_psd), 1. / 365)
i = fftfreq > 0

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(fftfreq[i], 10 * np.log10(temp_psd[i]))
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('PSD (dB)')
plt.show()

#  cut out freqencies higher than 1.1
temp_fft_bis = temp_fft.copy()
temp_fft_bis[np.abs(fftfreq) > 1.1] = 0

# inverse fft to convert back to the time domain
temp_slow = np.real(sp.fftpack.ifft(temp_fft_bis))
print(len(temp_slow), len(temp_daily))
plt.plot(temp_daily.index, temp_slow)
temp_daily.plot()
plt.show()

# Fast fourier transform of half hourly demand
demand_fft = sp.fftpack.fft(df['demand'].values)
#print(temp_fft)
# Power spectoral density
demand_psd = np.abs(demand_fft) ** 2
# get frequecies corresponding to psd
fftfreq = sp.fftpack.fftfreq(len(demand_psd), 1. / ( 48 * 365) )
i = fftfreq > 0

plt.plot(fftfreq[i], 10 * np.log10(demand_psd[i]))
plt.show()

#  cut out freqencies higher than 1.1
demand_fft_bis = demand_fft.copy()
demand_fft_bis[np.abs(fftfreq) > 1.1] = 0
# inverse fft to convert back to the time domain
temp_slow = np.real(sp.fftpack.ifft(demand_fft_bis))
plt.plot(df.index, temp_slow)
df['demand'].plot()
plt.show()

#fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#ax.plot(fftfreq[i], 10 * np.log10(temp_psd[i]))
#ax.set_xlim(0, 5)
#ax.set_xlabel('Frequency (1/day)')
#ax.set_ylabel('PSD (dB)')
