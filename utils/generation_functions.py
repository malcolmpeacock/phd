# power generation functions

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date

def read_adverse(warming='2-4', eno='2', etype='s', period = '5', file_parm='windspeed', parm_name='wind_speed'):
    etypes = { 'd' : 'duration', 's' : 'severity' }
    event = 'winter_wind_drought'
    filename = '/home/malcolm/uclan/data/adverse/{}_uk_return_period_1_in_{}_years_{}_gwl{}degC_event{}_{}.nc'.format(event, period, etypes[etype], warming, eno, file_parm)
    # Read the netCDF file
    print('Reading netcdf file {} ...'.format(filename))
    nc = Dataset(filename)
    print(nc.variables)
    time = nc.variables['time'][:]
    time_units = nc.variables['time'].units
    latitude = nc.variables['latitude'][:]
    longitude = nc.variables['longitude'][:]
    wind = nc.variables[parm_name][:]
#   print(wind)

    times=num2date(time, time_units,only_use_cftime_datetimes=False,only_use_python_datetimes=True)
    df = pd.DataFrame(data=wind.reshape(len(time), len(latitude) * len(longitude)), index=pd.DatetimeIndex(times, name='time'), columns=pd.MultiIndex.from_product([latitude, longitude], names=('latitude', 'longitude')))
#   print('Read Adverse data length {} NaNs {}'.format(len(df), df.isna().sum().sum() ) )
    return df

def rayleigh(mean_wind_speed, wind_speed):
    r = math.exp( -(math.pi/4.0) * math.pow(wind_speed/mean_wind_speed, 2) )
    return r

def rayleigh_bins(mean_wind_speed, nbins, cut_in, cut_out):
    # array of speeds and probabilities
    speed=np.empty(nbins)
    probability=np.empty(nbins)
    # probability below the cut in speed
    probability[0] = 1 - rayleigh(mean_wind_speed, cut_in)
    speed[0] = cut_in
    # probability above the cut out speed
    probability[nbins-1] = rayleigh(mean_wind_speed, cut_out)
    speed[nbins-1] = cut_out
    # intermediate probabilities
    for i in range(1,nbins-1):
        bin_speed_lower = (i-1) * (cut_out - cut_in) / (nbins-2)
        bin_speed_upper = i * (cut_out - cut_in) / (nbins-2)
        probability[i] = rayleigh(mean_wind_speed, cut_in+bin_speed_upper) - rayleigh(mean_wind_speed, cut_in+bin_speed_lower )
        speed[i] = cut_in + bin_speed_upper

    return speed, np.abs(probability)

# Test Program

def run_tests():

    # rayleigh#
    print("### rayleigh\n")
    mean_wind_speeds = [4, 9, 20]
    wind_speeds = [5, 7, 10]
    print(" Probability  Speed     Mean Speed")
    for m in mean_wind_speeds:
        for s in wind_speeds:
            r = rayleigh(m, s)
            print("   {:.3f}      {:6.3f}      {:6.3f}".format(r, s, m) )

    speed=[]
    prob=[]
    mean=7.0
    for s in range(25):
        speed.append(s)
        prob.append(rayleigh(mean,s))
    plt.plot(speed, prob)
    plt.title('rayleigh test')
    plt.xlabel('Speed', fontsize=15)
    plt.ylabel('Probability speed is above {}'.format(mean), fontsize=15)
    plt.show()

    print("--- \n\n")
    # rayleigh_bins
    print("### rayleigh_bins\n")

    mean_wind_speed = 6
    nbins=10
    cut_in = 4
    cut_out = 25

    speed, probability = rayleigh_bins(mean_wind_speed, nbins, cut_in, cut_out)
    df = pd.DataFrame( {'speed' : speed, 'probability' : probability } )
    print(df)
    print('Sum of probabilities {}'.format(df['probability'].sum() ) )

    plt.plot(df['speed'], df['probability'])
    plt.title('rayleigh_bins test')
    plt.xlabel('Speed', fontsize=15)
    plt.ylabel('Probability', fontsize=15)
    plt.show()
