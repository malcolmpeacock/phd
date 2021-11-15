
# contrib code
import sys
import pandas as pd
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

# forecast
#   df_in       input weather and demand
#   df_out      max and min demand for same period as df_in
#   df_forecast same variables as df_in but for the forecast period
def forecast(df_in, df_out, df_forecast):
    print('forecast: df_in {} df_out {} df_forecast {}'.format(len(df_in), len(df_out), len(df_forecast) ) )
    print('max demand ...')
#   max_cols = ['demand', 'solar_irradiance1', 'windspeed_east1', 'k', 'windspeed1', 'windspeed3', 'solar_irradiance2']
    max_cols = ['demand', 'solar_irradiance1', 'windspeed_east1', 'k', 'windspeed1', 'windspeed3', 'solar_irradiance2', 'spec_humidity1_lag1', 'solar_irradiance1_lag1']
    max_demand_forecast = rf_forecast(max_cols, df_in, df_forecast, df_out['max_demand'])
    print('min demand ...')
    min_demand_forecast = rf_forecast(['demand', 'spec_humidity1', 'dailytemp', 'temperature3', 'demand_lag1', 'temperature2', 'windspeed_north3', 'windspeed_east3', 'temperature5'], df_in, df_forecast, df_out['min_demand'])
#   prediction = naive(df_in, df_out, df_forecast)
    data = { 'max_demand' :  max_demand_forecast, 'min_demand': min_demand_forecast }
    prediction = pd.DataFrame(data, index=df_forecast.index)
    return prediction

def rf_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    # default error is RMSE criterion=“squared_error”
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return y_pred

def naive(df_forecast):
    print('naive: df_forecast {}'.format(len(df_forecast) ) )
#   print(df_forecast)
    max_demand_forecast = df_forecast['demand'].copy()
    min_demand_forecast = df_forecast['demand'].copy()
    data = { 'max_demand' :  max_demand_forecast, 'min_demand': min_demand_forecast }
    prediction = pd.DataFrame(data, index=df_forecast.index)
    return prediction

def assess(df_forecast, df_actual):
    print('assess: df_forecast {} df_actual {}'.format(len(df_forecast), len(df_actual) ) )
#   print(df_forecast.columns)
#   print(df_actual.columns)
#   print(df_forecast)
#   print(df_actual)
    if len(df_forecast) != len(df_actual):
        print('ERROR forecast and actual different lengths')
        quit()
    max_diff2 = (df_forecast['max_demand'] - df_actual['max_demand']).pow(2)
    min_diff2 = (df_forecast['min_demand'] - df_actual['min_demand']).pow(2)
    rmse = math.sqrt(max_diff2.sum() + min_diff2.sum() )
    return rmse

# class to create ANN

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self,num_inputs,num_outputs,num_hidden):
        super().__init__()
#       Layer 1
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        # Activation function
        self.act1 = nn.Sigmoid()
#       Layer 2
        self.act2 = nn.LeakyReLU()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    loss_history=[]
    for epoch in range(num_epochs):
        # each batch in the training ds
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        # report at each epoc
        sys.stdout.write('\repoch {:03d}, loss {:.7f} '.format(epoch, loss.item()))
        loss_history.append(loss.item() )
    sys.stdout.flush()
    return loss_history


# forecast
#   df_in       input weather and demand
#   df_out      max and min demand for same period as df_in
#   df_forecast same variables as df_in but for the forecast period
def ann_forecast(df_in, df_out, df_forecast):
    inputs = torch.tensor(df_in.values.astype(np.float32))
    targets = torch.tensor(df_out.values.astype(np.float32))
    torch.manual_seed(seed)    # reproducible
    train_ds = TensorDataset(inputs, targets)

    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))
    num_inputs = len(df_in.columns)
    num_outputs = len(df_out.columns)
    loss_fn = F.mse_loss
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = loss_fn(model(inputs), targets)
    # Train the model
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
    preds = model(inputs)
    # normalise the inputs (using same max as for the model)
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
#   print(preds)
    vals = preds.detach().numpy()[0]



# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--naive', action="store_true", dest="naive", help='Output the naive forecast', default=False)
parser.add_argument('--start', action="store", dest="start", help='Where to start rolling assesment from: 0=just forecast, 1=30 days before the end, 2=31 etc.' , default=0, type=int )
parser.add_argument('--step', action="store", dest="step", help='Rolling assesment step.' , default=1, type=int )
args = parser.parse_args()

# read in the data
output_dir = "/home/malcolm/uclan/challenge2/output/"
# merged data file ( demand, weather, augmented variables )
merged_filename = '{}merged_pre_august.csv'.format(output_dir)
df_in = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
# print(df_in)

# maxmin data file ( min/max in the period - what we are trying to predict )
merged_filename = '{}maxmin_pre_august.csv'.format(output_dir)
df_out = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
# print(df_out)


if args.start == 0:
    # read in the default forecast
    forecast_filename = '{}merged_august.csv'.format(output_dir)
    df_f_in = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    # print(df_f_in)

    if args.naive:
        # calculate naive bench mark
        df_forecast = naive(df_f_in)
    else:
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
    output_filename = '{}predictions.csv'.format(output_dir)
    df_forecast.to_csv(output_filename, float_format='%.8f')

else:
    rmses=[]
    # 30 days and 48 half hour periods
    forecast_days = 30
    # for each window ...
    for window in range(0, args.start, args.step):
        print('Window {} of {}'.format(window, args.start / args.step) )
        # create a forecast df and shorten the input df
        win_start = window*48
        win_end  = len(df_in) - (forecast_days + args.start - window)*48
        # training data ( weather and demand for prior period )
        df_train_in = df_in[win_start:win_end]
        # print('df_train_in')
        # print(df_train_in)
        # training data ( max/min that we are tyring to predict for same period)
        df_train_out = df_out[win_start:win_end]
        # print('df_train_out')
        # print(df_train_out)
        # training data ( weather and demand for forecast period)
        df_f_in  = df_in[win_end:win_end+forecast_days*48]
        # print('df_f_in')
        # print(df_f_in)
        # training data ( max/min that we are tyring to predict for forecast period)
        df_f_out  = df_out[win_end:win_end+forecast_days*48]
        # print('df_f_out')
        # print(df_f_out)
        # forecast it
        df_forecast = forecast(df_train_in, df_train_out, df_f_in)
        # calculate naive bench mark
        df_bench = naive(df_f_in)

        # assess the forecast
        rmse = assess(df_forecast, df_f_out)
        rmse_b = assess(df_bench, df_f_out)
        skill = rmse / rmse_b
        # store the assesment
        rmses.append([rmse, rmse_b, skill])

        # plot
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
        
    # output all the assessments
    skill = 0.0
    print('RMSE  Naive RMSE  Skill')
    for vals in rmses:
        print("{:.3f} {:.3f} {:.3f}".format(vals[0], vals[1], vals[2]))
        skill += vals[2]
    print('Average skill {}'.format(skill / len(rmses) ) )
