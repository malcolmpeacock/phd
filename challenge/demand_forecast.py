# python script to create a demand forecast from the previous demand and
# the weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
# Import nn.functional
import torch.nn.functional as F

# custom code
import utils

# custom loss function
def loss_max1(X,Y):
    return abs( X.max() - Y.max() )

# class for custom regression

class Demandregression(torch.nn.Module):
    def __init__(self):

        super().__init__()
        # weights
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
#       self.c = torch.nn.Parameter(torch.randn(()))
#       self.d = torch.nn.Parameter(torch.randn(()))
#       self.e = torch.nn.Parameter(torch.randn(()))
#       self.act1 = nn.ReLU() 

    # x is a tensor of input data
    def forward(self, x):
#       demand = (self.a + self.b * x[:,0] + ( self.c * x[:,0] * x[:,0] ) ) * ( self.d * x[:,1] + self.e )
        demand = (self.a + self.b * x[:,0] )
#       return demand.clamp(min=0.0)
#       print(x[:,0])
#       print(x[:,1])
#       print(demand)
        return demand
#       return self.act1(demand)

    def string(self):
        return 'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# class for custom regression#2

class Demandregression2(torch.nn.Module):
    def __init__(self):

        super().__init__()
        # weights
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
#       self.e = torch.nn.Parameter(torch.randn(()))
#       self.act1 = nn.ReLU() 

    # x is a tensor of input data
    def forward(self, x):
#       demand = (self.a + self.b * x[:,0] + ( self.c * x[:,0] * x[:,0] ) ) * ( self.d * x[:,1] + self.e )
        demand = (self.a + self.b * x[:,0] + self.c * x[:,0] * x[:,1] + self.d * x[:,1] )
#       return demand.clamp(min=0.0)
#       print(x[:,0])
#       print(x[:,1])
#       print(demand)
        return demand
#       return self.act1(demand)

    def string(self):
        return 'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# class to create ANN

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self,num_inputs,num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_inputs)
        # Activation function
        self.act1 = nn.LeakyReLU() 
#       self.act1 = nn.ReLU() 
#       self.act1 = nn.Sigmoid() 
#       Layer 1
        self.linear2 = nn.Linear(num_inputs, num_outputs)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
#       x = self.linear2(x)
        x = self.linear2(x).clamp(min=0.0)
        return x

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    loss_history=[]
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
#           print('BACKWARD', loss.item(), model.linear1.bias.grad, model.linear2.bias.grad)
#           print(xb,yb)
#           if math.isnan(loss.item()):
#               print("Stopped because loss is nan")
#               print(xb,yb)
#               print('Loss and grad', loss.item(), model.linear1.bias.grad, model.linear2.bias.grad)
#               quit()
            opt.step()
            opt.zero_grad()
#           print('TRAIN:')
#           print(loss.item(), model.linear1.bias.grad)
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        loss_history.append(loss.item() )
    return loss_history

# FORECASTING METHODS:

# naive forecast based on the previous week
def forecast_naive(df, forecast, day):
    if day == df.index[0].date:
        copy_day = df.last_valid_index().date.strftime('%Y-%m-%d')
    else:
        copy_day = (day - pd.Timedelta(days=1) ).strftime('%Y-%m-%d')
    print(day, copy_day)
    forecast['probability'] = 0.9
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = df.loc[copy_day, 'demand' ].values

# closest day

def forecast_closest_day(df, forecast, day):
    days = pd.Series(df.index.date).unique()
    demand_range = forecast['tempm'].max() - forecast['sun2'].min()
    print("Testing {}".format(day))
    closest_day, closeness = utils.find_closest_day(day, days, forecast, df, 'sunw')
    print(closest_day)
    rows = df.loc[closest_day.strftime('%Y-%m-%d')]
#       print(rows)
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = rows['demand'].values
    probability = (sun_range - closeness) / sun_range
    forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability
    print(forecast)

# closest 10 days from each weather grid, then weighted average

def forecast_closest_days(df, forecast, day):
    days = pd.Series(df.index.date).unique()
    closest_days = utils.find_closest_days(day, days, forecast, df, 'sun1', 10)
    new_day1 = utils.create_day(closest_days.index, df, 'demand')
    new_day1.columns = ['pv1']
    closest_days = utils.find_closest_days(day, days, forecast, df, 'sun2', 10)
    new_day2 = utils.create_day(closest_days.index, df, 'demand')
    new_day2.columns = ['pv2']
    closest_days = utils.find_closest_days(day, days, forecast, df, 'sun5', 10)
    new_day5 = utils.create_day(closest_days.index, df, 'demand')
    new_day5.columns = ['pv5']
    closest_days = utils.find_closest_days(day, days, forecast, df, 'sun6', 10)
    new_day6 = utils.create_day(closest_days.index, df, 'demand')
    new_day6.columns = ['pv6']
    new_days = pd.concat([new_day1, new_day2, new_day5, new_day6], axis=1)
    new_days.index=forecast.loc[day.strftime('%Y-%m-%d')].index
    utils.add_weighted(new_days, 'pv', 'demand')
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = new_days['demand'].values
    probability = 0.8
    forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability

# nreg - new regression - actually converges to something !!
def forecast_nreg(df, forecast, day):
    input_columns = ['tempm', 'dsk']
    input_df = df[input_columns].copy()
    # set up output
    output_column = 'demand'
    output = df[output_column].copy()

    # store maximum values
    input_max = {}
    # normalise the inputs
    for column in input_df.columns:
        input_max[column] = input_df[column].max()
        input_df[column] = input_df[column] / input_df[column].max()
    # normalise the output (Y)
    output_max = output.max()
    output = output / output_max
    # santity check
    for column in input_df.columns:
        if input_df[column].isna().sum() >0:
            print("ERROR NaN in {}".format(column))
            quit()
    if output.isna().sum() >0:
        print("ERROR NaN in output")
        quit()
    inputs = torch.tensor(input_df.values.astype(np.float32))
    print("inputs")
    print(inputs)
    targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
    print("targets")
    print(targets)
    torch.manual_seed(1)    # reproducible
    train_ds = TensorDataset(inputs, targets)

    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)
    # custom function didn't converge
    model = Demandregression2()

    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

    # Define loss function
    loss_fn = F.mse_loss
#   loss_fn = F.l1_loss

    loss = loss_fn(model(inputs), targets)
    print(loss)
    # Train the model for 100 epochs
    num_epochs=300
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
    preds = model(inputs)
    print(preds)
    # prediction
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    print(forecast_day)
    day_f = forecast_day
    input_f = day_f[input_columns].copy()
    print(input_f)
    # normalise the inputs (using same max as for the model)
    for column in input_f.columns:
        input_f[column] = input_f[column] / input_max[column]
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
    print('f_inputs')
    print(f_inputs)
    preds = model(f_inputs)
    print(preds)
    # denormalize using df for the original model
    prediction_values = preds.detach().numpy() * output_max
    print(prediction_values)
    # set forecast for zentih angle for daylight and denormalize using
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = prediction_values

    return losses

# reg - regression

def forecast_reg(df, forecast, day, method, plot):
    pred_values=[]
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    # for each k period, train a seperate model ...
    for index, row in forecast_day.iterrows():
        dsk = row['dsk']
        dsk_df = df[df['dsk'] == dsk]
        # set up inputs
        if method == 'regl':
            input_columns = ['sun2', 'sun1', 'sun5', 'sun6']
        if method == 'regd':
            input_columns = ['tempm', 'holiday']
        input_df = dsk_df[input_columns].copy()

        # set up output
        output_column = 'demand'
        output = dsk_df[output_column].copy()

        # store maximum values
        input_max = {}
        # normalise the inputs
        for column in input_df.columns:
            input_max[column] = input_df[column].max()
            input_df[column] = input_df[column] / input_df[column].max()
        # normalise the output (Y)
        output_max = output.max()
        output = output / output_max

        # santity check
        for column in input_df.columns:
            if input_df[column].isna().sum() >0:
                print("ERROR NaN in {}".format(column))
                quit()
        if output.isna().sum() >0:
            print("ERROR NaN in output")
            quit()

        inputs = torch.tensor(input_df.values.astype(np.float32))
        print("inputs")
        print(inputs)
#   The .view seems to tell it what shape the data is
        targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
        print("targets")
        print(targets)
        torch.manual_seed(1)    # reproducible
        train_ds = TensorDataset(inputs, targets)

        batch_size = 48
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        next(iter(train_dl))

        num_inputs = len(input_df.columns)
    
        if method == 'regl':
        # model using regression
            model = nn.Linear(num_inputs,1)
            loss_fn = F.mse_loss
        if method == 'regd':
        # custom function didn't converge
            model = Demandregression()
            loss_fn = F.l1_loss
#           loss_fn = loss_max1

        opt = torch.optim.SGD(model.parameters(), lr=1e-5)

        # Define loss function
#       loss_fn = F.mse_loss
#       loss_fn = F.l1_loss

        loss = loss_fn(model(inputs), targets)
        print(loss)
        # Train the model for 100 epochs
        num_epochs=50
        losses = fit(num_epochs, model, loss_fn, opt, train_dl)
        print('Training loss: ', loss_fn(model(inputs), targets))
        preds = model(inputs)
        print(preds)
        # prediction
        forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
        print(forecast_day)
        dsk_f = forecast_day[forecast_day['dsk'] == dsk]
        input_f = dsk_f[input_columns].copy()
        print(input_f)
        # normalise the inputs (using same max as for the model)
        for column in input_f.columns:
            input_f[column] = input_f[column] / input_max[column]
        f_inputs = torch.tensor(input_f.values.astype(np.float32))
        print('f_inputs')
        print(f_inputs)
        preds = model(f_inputs)
        print(preds)
        # denormalize using df for the original model
        prediction_values = preds.detach().numpy() * output_max
        print(prediction_values)
        pred_values.append(prediction_values[0])

        if args.plot and dsk==36:
            plt.plot(losses)
            plt.title('Demand Regression convergence')
            plt.xlabel('Epochs', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.show()

    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = pd.Series(pred_values, index=forecast_day.index)


# ann - artificial neural network.

def forecast_ann(df, forecast, day):
    # don't try to predict pv at night!
    day_df = df[df['zenith'] < 87]
    # set up inputs
#   input_columns = ['zenith', 'sunw', 'tempw']
    input_columns = ['sun1', 'sun2', 'sun5', 'sun6']
    input_df = day_df[input_columns].copy()
    print(input_df)

    # set up output
    output_column = 'demand'
    output = day_df[output_column]
    # normalise the output (Y)
    output_max = output.max()
    output = output / output_max

    # store maximum values
    input_max = {}
    # normalise the inputs (X)
    for column in input_df.columns:
        input_max[column] = input_df[column].max()
        input_df[column] = input_df[column] / input_df[column].max()

    # santity check
    for column in input_df.columns:
        if input_df[column].isna().sum() >0:
            print("ERROR NaN in {}".format(column))
            quit()
    if output.isna().sum() >0:
        print("ERROR NaN in output")
        quit()

    inputs = torch.tensor(input_df.values.astype(np.float32))
    print("inputs")
    print(inputs)
#   The .view seems to tell it what shape the data is
    targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
    print("targets")
    print(targets)
    torch.manual_seed(1)    # reproducible
#   torch.manual_seed(8)    # reproducible
    train_ds = TensorDataset(inputs, targets)
#   train_ds[0:3]

    # Define data loader
#   batch_size = 1
#   batch_size = 5
    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)
    # model using ANN
    model = SimpleNet(num_inputs,1)

    # Define optimizer
#   opt = torch.optim.SGD(model.parameters(), lr=1e-2)
#   opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)
#   opt = torch.optim.SGD(model.parameters(), lr=1e-6)

    # Define loss function
    loss_fn = F.mse_loss
#   loss_fn = F.l1_loss

    loss = loss_fn(model(inputs), targets)
    print(loss)
    # Train the model for 100 epochs
    num_epochs=100
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
    # prediction
#   print(preds)
#   print(targets)
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    day_f = forecast_day[forecast_day['zenith'] < 87]
    input_f = day_f[input_columns].copy()
    print(input_f)
    # normalise the inputs (using same max as for the model)
    for column in input_f.columns:
        input_f[column] = input_f[column] / input_max[column]
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
    preds = model(f_inputs)

    forecast_day['prediction'] = 0.0
    # denormalize using df for the original model
    prediction_values = preds.detach().numpy() * output_max
    # set forecast for zentih angle for daylight
    forecast_day.loc[forecast_day['zenith']<87, 'prediction'] = prediction_values
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = forecast_day['prediction'].values

    print(forecast)
    return losses
# closest 10 hours from each weather grid, then weighted average

def forecast_closest_hours(df, forecast, day):
    # for each hour of the day ...
    new_values=[]
    for index, row in forecast.iterrows():
        print(index)
        closest_periods = utils.find_closest_periods(row, df, 'sun2', 10)
        new_values.append( utils.create_half_hour(closest_periods.index, df['demand']) )

    new_series = pd.Series(new_values, forecast.index)
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = new_series.values
    probability = 0.8
    forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability

# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method: reg2, reg, ann, sday' , default='simple' )
parser.add_argument('--day', action="store", dest="day", help='Day to forecast: set=read the set forecast file, first= first day, last=last day, all=loop to forecast all days based on the others, otherwise integer day' , default='set' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
method = args.method
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

df['holiday'] = 0
df.loc[(df['wd']>4) | (df['ph']==1), 'holiday' ] = 1

print(df)

# weather data (forecast)
forecast_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

forecast['holiday'] = 0
forecast.loc[(forecast['wd']>4) | (forecast['ph']==1), 'holiday' ] = 1

if args.day != 'set':
    columns = forecast.columns.append(pd.Index(['demand']))
    if args.day == 'all':
        forecast = df[columns]
    else:
        days = pd.Series(df.index.date).unique()
        if args.day == 'first':
           day=0
        else: 
           if args.day == 'last':
               day=len(days)-1
           else:
               day = int(day)
        day_text = days[day].strftime("%Y-%m-%d")
        day_start = day_text + ' 00:00:00'
        day_end = day_text + ' 23:30:00'
        forecast = df.loc[day_start : day_end]
        forecast = forecast[columns]

print(forecast)

# for each day to be forecast ...
fdays = pd.Series(forecast.index.date).unique()
for day in fdays:
    print('Method {} day {}'.format(method, day) )
    day_text = day.strftime("%Y-%m-%d")
    day_start = day_text + ' 00:00:00'
    day_end = day_text + ' 23:30:00'
    # drop this day from main data
    if args.day == 'set':
        history = forecast
    else:
        history = df.drop(df.loc[day_text].index)

    # Naive Demand forecast based on same as last week
    if method == 'naive':
        forecast_naive(history, forecast, day)
    
    # closest weather day method
    if method == 'sday':
        forecast_closest_day(history, forecast, day)

    # closest weather day method using several days
    if method == 'sdays':
        forecast_closest_days(history, forecast, day)

    if method[0:3] == 'reg':
        forecast_reg(history, forecast, day, method, args.plot)

    if method == 'nreg':
        losses = forecast_nreg(history, forecast, day)
        if args.plot:
            plt.plot(losses)
            plt.title('demand nreg convergence')
            plt.xlabel('Epochs', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.show()

    if method == 'ann':
        losses = forecast_ann(history, forecast, day)
        if args.plot:
            plt.plot(losses)
            plt.title('demand ann convergence')
            plt.xlabel('Epochs', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.show()

print(forecast)

# metrics
if 'demand' in forecast.columns:
    utils.print_metrics(forecast['demand'], forecast['prediction'], args.plot)
    if args.plot:
        forecast['demand'].plot(label='actual power', color='blue')
        forecast['prediction'].plot(label='predicted power', color='red')
        plt.title('demand prediction : '+method)
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}demand_forecast_{}.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')

# only demand for bogdan
forecast = forecast['prediction']
forecast = forecast.squeeze()
forecast = forecast.rename('demand_forecast')
forecast.index.rename('datetime', inplace=True)

output_filename = '{}demand_forecast_{}_only.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')
