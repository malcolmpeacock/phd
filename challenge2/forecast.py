
# contrib code
import sys
import pandas as pd
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# forecast
#   df_in       input weather and demand
#   df_out      max and min demand for same period as df_in
#   df_forecast same variables as df_in but for the forecast period
def forecast(df_in, df_out, df_forecast):
    print('forecast: df_in {} df_out {} df_forecast {}'.format(len(df_in), len(df_out), len(df_forecast) ) )
    print('max demand ...')
    max_demand_forecast = rf_forecast(['demand', 'solar_irradiance1', 'windspeed_east1'], df_in, df_forecast, df_out['max_demand'])
    print('min demand ...')
    min_demand_forecast = rf_forecast(['demand', 'spec_humidity1', 'dailytemp'], df_in, df_forecast, df_out['min_demand'])
#   prediction = naive(df_in, df_out, df_forecast)
    data = { 'max_demand' :  max_demand_forecast, 'min_demand': min_demand_forecast }
    prediction = pd.DataFrame(data, index=df_forecast.index)
    return prediction

def rf_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return y_pred

def naive(df_in, df_out, df_forecast):
    print('naive: df_in {} df_out {} df_forecast {}'.format(len(df_in), len(df_out), len(df_forecast) ) )
#   print(df_forecast)
    df_n = df_forecast[['demand','k']]
    df_n['k'] = df_forecast['demand']
    df_n.columns = ['max_demand', 'min_demand']
#   print(df_n)
    return df_n

def assess(df_forecast, df_actual):
    print('assess: df_forecast {} df_actual {}'.format(len(df_forecast), len(df_actual) ) )
#   print(df_forecast.columns)
#   print(df_actual.columns)
#   print(df_forecast)
#   print(df_actual)
    max_diff2 = (df_forecast['max_demand'] - df_actual['max_demand']).pow(2)
    min_diff2 = (df_forecast['min_demand'] - df_actual['min_demand']).pow(2)
    rmse = math.sqrt(max_diff2.sum() + min_diff2.sum() )
    return rmse

# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--start', action="store", dest="start", help='Where to start rolling assesment from: 0=just forecast, 1=30 days before the end, 2=31 etc.' , default=0, type=int )
args = parser.parse_args()

# read in the data
output_dir = "/home/malcolm/uclan/challenge2/output/"
# merged data file
merged_filename = '{}merged_pre_august.csv'.format(output_dir)
df_in = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
# print(df_in)

# maxmin data file
merged_filename = '{}maxmin_pre_august.csv'.format(output_dir)
df_out = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
# print(df_out)

if args.start == 0:
    # read in the default forecast
    forecast_filename = '{}merged_august.csv'.format(output_dir)
    df_f_in = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    # print(df_f_in)

    # forecast it
    df_forecast = forecast(df_in, df_out, df_f_in)
    # print(df_forecast)

    # plot the forecast
    if args.plot:
        df_forecast['max_demand'].plot(label='forecast max_demand')
        df_forecast['min_demand'].plot(label='forecast min_demand')
        df_f_in['demand'].plot(label='half hourly demand')
        plt.title('Forecast min and max demand')
        plt.xlabel('Half Hour of the month', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='lower left', fontsize=15)
        plt.show()

    # output the forecast
    df_forecast.columns = ['value_max', 'value_min']
    output_filename = '{}Predictions.csv'.format(output_dir)
    df_forecast.to_csv(output_filename, float_format='%.2f')

else:
    rmses=[]
    # 30 days and 49 half hour periods
    forecast_days = 30
    print('RMSE  Naive RMSE  Skill')
    # for each window ...
    for window in range(args.start):
        # create a forecast df and shorten the input df
        win_start = window*48
        win_end  = len(df_in) - (forecast_days + args.start - window)*48
        # training data k
        df_train_in = df_in[win_start:win_end]
        # print('df_train_in')
        # print(df_train_in)
        df_train_out = df_out[win_start:win_end]
        # print('df_train_out')
        # print(df_train_out)
        df_f_in  = df_in[win_end:win_end+forecast_days*48]
        # print('df_f_in')
        # print(df_f_in)
        df_f_out  = df_out[win_end:win_end+forecast_days*48]
        # print('df_f_out')
        # print(df_f_out)
        # forecast it
        df_forecast = forecast(df_train_in, df_train_out, df_f_in)
        # calculate naive bench mark
        df_bench = naive(df_train_in, df_train_out, df_f_in)

        # assess the forecast
        rmse = assess(df_forecast, df_f_out)
        rmse_b = assess(df_bench, df_f_out)
        skill = rmse / rmse_b
        print("{:.3f} {:.3f} {:.3f}".format(rmse, rmse_b, skill))

        if args.plot:
            df_forecast['max_demand'].plot(label='forecast max_demand')
            df_f_out['max_demand'].plot(label='actual max_demand')
            df_forecast['min_demand'].plot(label='forecast min_demand')
            df_f_out['min_demand'].plot(label='actual min_demand')
            df_f_in['demand'].plot(label='half hourly demand')
            plt.title('Forecast and actual min and max demand')
            plt.xlabel('Half Hour of the month', fontsize=15)
            plt.ylabel('Demand (MW)', fontsize=15)
            plt.legend(loc='lower left', fontsize=15)
            plt.show()
        
        # store the assesment
    # output all the assessments
