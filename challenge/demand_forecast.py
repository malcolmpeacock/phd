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
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# custom code
import utils

# custom loss function
def loss_max1(X,Y):
#   lossn = abs( X.max().item() - Y.max().item() )
#   print(X.max().item(), Y.max().item(), lossn)
    lossn =  torch.abs(X.max() - Y.max())
    return lossn

# class for custom regression

class Demandregression(torch.nn.Module):
    def __init__(self):

        super().__init__()
        self.linear1 = nn.Linear(3, 1)
        self.bilinear1 = nn.Bilinear(1, 1, 1)
        self.act1 = nn.LeakyReLU()

    # x is a tensor of input data
    # x[0] is mean temperature
    # x[1] is holiday
    def forward(self, x):
        # t = temperature
        t = x[:,0].view(-1,1)
#       print(t)
        # h = holiday flag
        h = x[:,1].view(-1,1)
#       print(h)
        # tsq = t-squared
        tsq = self.bilinear1(t, t)
#       print(tsq)
        xx = torch.cat([tsq, t, h], 1)
#       print(xx)
        y = self.linear1(xx)
#       demand = self.act1(y).clamp(min=0.0)
        demand = self.act1(y)
        return demand

    def string(self):
        return 'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# class for custom regression#2

class Demandregression2(torch.nn.Module):
    def __init__(self):

        super().__init__()
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
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        loss_history.append(loss.item() )
    return loss_history

# FORECASTING METHODS:

# naive forecast based on the previous week
def forecast_naive(df, forecast, day):
    copy_day = (day - pd.Timedelta(days=7) ).strftime('%Y-%m-%d')
    # if the previous day doesn't exist (because its the first day) then
    # use the last day instead
    if copy_day not in df.index:
        print('{} NOT FOUND using last day instead'.format(copy_day))
        copy_day = df.last_valid_index().date().strftime('%Y-%m-%d')
    if len(df.loc[copy_day, 'demand' ].values) == 0:
        print('{} has no demand values, using last day instead'.format(copy_day))
        copy_day = df.last_valid_index().date().strftime('%Y-%m-%d')
    forecast['probability'] = 0.9
    print('For day {} use day {} len {}'.format(day, copy_day, len(df.loc[copy_day, 'demand' ].values) ) )
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = df.loc[copy_day, 'demand' ].values

# closest day

def forecast_closest_day(df, forecast, day):
    days = pd.Series(df.index.date).unique()
    demand_range = forecast['tempm'].max() - forecast['tempm'].min()
    closest_day, closeness = utils.find_closest_day(day, days, forecast, df, 'tempm')
    print(closest_day)
    rows = df.loc[closest_day.strftime('%Y-%m-%d')]
#       print(rows)
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = rows['demand'].values
    probability = (demand_range - closeness) / demand_range
    forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability
#   print(forecast)

# closest 10 days from each weather grid, then weighted average

def forecast_closest_days(df, forecast, day, method):
    if method=='sdays':
        days = pd.Series(df.index.date).unique()
    else:
        if forecast.loc[day.strftime('%Y-%m-%d'),'holiday'][0] == 1:
            print('Looking for Holiday')
            days = pd.Series(df[df['holiday']==1].index.date).unique()
        else:
            print('Looking for a non Holiday')
            days = pd.Series(df[df['holiday']==0].index.date).unique()

    closest_days = utils.find_closest_days(day, days, forecast, df, 'tempm', 10, True)
    new_day = utils.create_day(closest_days.index, df, 'demand')
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = new_day['demand'].values
    probability = 0.8
    forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability

# skt
def forecast_skt(df, forecast, day):
    input_columns = ['tempm', 'dsk']
    input_df = df[input_columns].copy()
    # set up output
    output_column = 'demand'
    output = df[output_column].copy()

    x = input_df.values
    y = output.values

    # Step 2b: Transform input data
    x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

    # Step 3: Create a model and fit it
    model = LinearRegression().fit(x_, y)

    # Step 4: Get results
    r_sq = model.score(x_, y)
    intercept, coefficients = model.intercept_, model.coef_

    # Step 5: Predict
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    x_f = forecast_day[input_columns].copy().values
    xf = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_f)
    y_pred = model.predict(xf)
    print(y_pred)
    print(len(y_pred))

    print(forecast_day)
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = y_pred

# numpy
def forecast_numpy(df, forecast, day):
    tempm = df['tempm'].values
    dsk = df['dsk'].values
    output_column = 'demand'
    output = df[output_column].values

    # Randomly initialize weights
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    d = np.random.randn()
    e = np.random.randn()

    losses=[]
    learning_rate = 1e-6
    for t in range(2000):
        # Forward pass: compute predicted y
        y_pred = a + b * tempm + c * np.sin(dsk * d + e )

        # Compute and print loss
        loss = np.square(y_pred - output).sum()
        if t % 100 == 99:
            print(t, loss)
            losses.append(loss)

        # Backprop to compute gradients of a, b, c, d with respect to loss
        grad_y_pred = 2.0 * (y_pred - output)
        grad_a = grad_y_pred.sum()
        grad_b = (grad_y_pred * tempm).sum()
        grad_c = grad_y_pred.sum()
        grad_d = (grad_y_pred * dsk).sum()
        grad_e = grad_y_pred.sum()
        # Update weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
        e -= learning_rate * grad_e

    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    f_tempm = forecast_day['tempm'].values
    f_dsk = forecast_day['dsk'].values
    y_pred = a + b * f_tempm + c * numpy.sine(f_dsk * d + e )
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = y_pred

    return losses

# nreg - new regression - actually converges to something !!
def forecast_nreg(df, forecast, day, seed, num_epochs):
    df['dskd'] = df['dsk'] - 37
    forecast['dskd'] = forecast['dsk'] - 37
    input_columns = ['tempm', 'dskd']
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
#   print("inputs")
#   print(inputs)
    targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
#   print("targets")
#   print(targets)
    torch.manual_seed(seed)    # reproducible
    train_ds = TensorDataset(inputs, targets)

    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)
    # custom function didn't converge
#   model = Demandregression2()
    model = nn.Linear(2, 1)

    opt = torch.optim.SGD(model.parameters(), lr=1e-5)

    # Define loss function - mse_loss converges but doesn't look good
    loss_fn = F.mse_loss
#   loss_fn = loss_max1
#   loss_fn = F.l1_loss

    loss = loss_fn(model(inputs), targets)
    print(loss)
    # Train the model for 100 epochs
#   num_epochs=200
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

def forecast_reg(df, forecast, day, method, plot, seed, num_epochs, period, ki):
    pred_values=[]
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    if period == all:
        # for each k period, train a seperate model ...
        for index, row in forecast_day.iterrows():
            dsk = row['dsk']
            if ki and (row['k'] < 32 or row['k'] > 42):
                prediction_values = 1.0
            else:
                print('Period {}'.format(dsk) )
                dsk_df = df[df['dsk'] == dsk]
                dsk_f = forecast_day[forecast_day['dsk'] == dsk]
                prediction_values = forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ki)
                if math.isnan(prediction_values):
                    print('WARNING NaN set to Zero at period {}'.format(row['k']))
                    prediction_values = 0
                if not math.isfinite(prediction_values):
                    print('WARNING Inf set to Zero at period {}'.format(row['k']))
                    prediction_values = 0
            pred_values.append(prediction_values)
        forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = pd.Series(pred_values, index=forecast_day.index)
    else:
        dsk = int(period)
        dsk_df = df[df['dsk'] == dsk]
        dsk_f = forecast_day[forecast_day['dsk'] == dsk]
        prediction_values = forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ki)
        print('Predicted value {} for period {}'.format(prediction_values, period))
        quit()

def forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ki):
    # set up inputs
    if method == 'regl':
#       input_columns = ['tempm', 'zenith']
#       input_columns = ['tempm', 'zenith', 'holiday']
        input_columns = ['tempm', 'zenith', 'holiday', 'nothol']
        batch_size = 1
        rate = 1e-4
    if method == 'regm':
# sunw causes nans in predicition?
#       input_columns = ['tempm', 'zenith', 'holiday', 'nothol', 'tsqd', 'th', 'tnh', 'sunw', 'sh']
        input_columns = ['tempm', 'zenith', 'holiday', 'nothol', 'tsqd', 'th', 'tnh']
        batch_size = 1
        rate = 1e-4
    if method == 'regd':
        input_columns = ['tempm', 'holiday']
        # could more than 1 mean trying to assess holiday=1 and holiday=0
        # at the same time?
        batch_size = 1
        rate = 1e-7
    input_df = dsk_df[input_columns].copy()

    # set up output
    output_column = 'demand'
    output = dsk_df[output_column].copy()

    # store maximum values
    input_max = {}
    # normalise the inputs
    for column in input_df.columns:
        input_max[column] = input_df[column].max()
        if input_df[column].max() > 0.0:
            input_df[column] = input_df[column] / input_df[column].max()
        else:
            input_df[column] = 0.0
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
#   print("inputs")
#   print(inputs)
#   The .view seems to tell it what shape the data is
    targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
#   print("targets")
#   print(targets)
    torch.manual_seed(seed)    # reproducible
    train_ds = TensorDataset(inputs, targets)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)

    if method == 'regl':
    # model using regression
        model = nn.Linear(num_inputs,1)
        loss_fn = F.mse_loss
    if method == 'regm':
    # model using regression
        model = nn.Linear(num_inputs,1)
        loss_fn = F.mse_loss
    if method == 'regd':
    # custom function didn't converge
        model = Demandregression()
        loss_fn = F.l1_loss
#       loss_fn = loss_max1

    opt = torch.optim.SGD(model.parameters(), lr=rate)

    # Define loss function
#   loss_fn = F.mse_loss
#   loss_fn = F.l1_loss

    loss = loss_fn(model(inputs), targets)
#   print(loss)
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
    preds = model(inputs)
#   print(preds)
    # prediction
    #forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    #print(forecast_day)
    #dsk_f = forecast_day[forecast_day['dsk'] == dsk]
    input_f = dsk_f[input_columns].copy()
#   print(input_f)
    # normalise the inputs (using same max as for the model)
    for column in input_f.columns:
        input_f[column] = input_f[column] / input_max[column]
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
#   print(preds)
    # denormalize using df for the original model
    prediction_values = preds.detach().numpy() * output_max
#   print(prediction_values)

    if args.plot:
        plt.plot(losses)
        plt.title('Demand Regression convergence. period {}'.format(dsk))
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.show()

    return prediction_values[0][0]


# ann - artificial neural network.

def forecast_ann(df, forecast, day, seed, num_epochs):
    # set up inputs
    input_columns = ['zenith', 'tempm', 'dsk', 'holiday']
    input_df = df[input_columns].copy()
    print(input_df)

    # set up output
    output_column = 'demand'
    output = df[output_column]
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
    torch.manual_seed(seed)    # reproducible
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
parser.add_argument('--seed', action="store", dest="seed", help='Random seed, default=1.0', type=float, default=1.0)
parser.add_argument('--epochs', action="store", dest="epochs", help='Number of epochs', type=int, default=100)
parser.add_argument('--period', action="store", dest="period", help='Period k to forecast', type=float, default=all)
parser.add_argument('--step', action="store", dest="step", help='If using days=all or step only do every step days', type=int, default=1)
parser.add_argument('--ki', action="store_true", dest="ki", help='Only forecast K of interest.', default=False)

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
df['nothol'] = 0
df.loc[(df['wd']<2) | (df['ph']==0), 'nothol' ] = 1

df['tsqd'] = df['tempm'] * df['tempm']
df['th'] = df['tempm'] * df['holiday']
df['tnh'] = df['tempm'] * df['nothol']

print(df)

# weather data (forecast)
forecast_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

forecast['holiday'] = 0
forecast.loc[(forecast['wd']>4) | (forecast['ph']==1), 'holiday' ] = 1
forecast['nothol'] = 0
forecast.loc[(forecast['wd']<2) | (forecast['ph']==0), 'nothol' ] = 1
forecast['tsqd'] = forecast['tempm'] * forecast['tempm']
forecast['th'] = forecast['tempm'] * forecast['holiday']
forecast['tnh'] = forecast['tempm'] * forecast['nothol']

if args.day != 'set':
    columns = forecast.columns.append(pd.Index(['demand']))
    if args.day == 'all' or args.day == 'sh' or args.day == 'ph' or args.day == 'hol':
        forecast = df[columns].copy()
        if args.day == 'sh':
            print('School holidays only')
            forecast.drop( forecast[ forecast['sh']==0].index, inplace=True)
        if args.day == 'ph':
            print('Public holidays only')
            forecast.drop( forecast[ forecast['ph']==0].index, inplace=True)
        if args.day == 'hol':
            print('Public holidays and weekends')
            forecast.drop( forecast[ forecast['holiday']==0].index, inplace=True)
    else:
        days = pd.Series(df.index.date).unique()
        if args.day == 'first':
           day=0
        else: 
           if args.day == 'last':
               day=len(days)-1
           else:
               day = int(args.day)
        day_text = days[day].strftime("%Y-%m-%d")
        day_start = day_text + ' 00:00:00'
        day_end = day_text + ' 23:30:00'
        forecast = df.loc[day_start : day_end]
        forecast = forecast[columns]

print(forecast)

# for each day to be forecast ...
fdays = pd.Series(forecast.index.date).unique()
num_fdays = len(fdays)
count=0
print('Forecasting {} days in steps of {}'.format(len(fdays), args.step) )
for id in range(len(fdays)):
    day = fdays[id]
    count+=1
    print('Method {} day {} of {} date {}'.format(method, count, num_fdays, day) )
    day_text = day.strftime("%Y-%m-%d")
    day_start = day_text + ' 00:00:00'
    day_end = day_text + ' 23:30:00'
    if id%args.step != 0:
        print('SKIPPING due to step')
        forecast.drop(forecast.loc[day_text].index, inplace=True)
    else:

        # drop this day from main data
        if args.day == 'set':
            history = df.copy()
        else:
            history = df.drop(df.loc[day_text].index).copy()

        # Naive Demand forecast based on same as last week
        if method == 'naive':
            forecast_naive(history, forecast, day)
    
        # closest weather day method
        if method == 'sday':
            forecast_closest_day(history, forecast, day)

        # closest weather day method using several days
        if method == 'sdays' or method == 'sdaysh':
            forecast_closest_days(history, forecast, day, method)

        if method[0:3] == 'reg':
            forecast_reg(history, forecast, day, method, args.plot, args.seed, args.epochs, args.period, args.ki)

        if method == 'skt':
            forecast_skt(df, forecast, day)
        if method == 'nreg':
            losses = forecast_nreg(history, forecast, day, args.seed, args.epochs)
            if args.plot:
                plt.plot(losses)
                plt.title('demand nreg convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

        if method == 'numpy':
            losses = forecast_numpy(history, forecast, day)
            if args.plot:
                plt.plot(losses)
                plt.title('demand numpy convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

        if method == 'ann':
            losses = forecast_ann(history, forecast, day, args.seed, args.epochs)
            if args.plot:
                plt.plot(losses)
                plt.title('demand ann convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

print(forecast)

# metrics
if 'demand' in forecast.columns:
    if args.ki:
        kf = forecast[ (forecast['k'] > 31) & (forecast['k'] < 43)]
        print(kf['demand'])
        print(kf['prediction'])
        utils.print_metrics(kf['demand'], kf['prediction'], False)
    utils.print_metrics(forecast['demand'], forecast['prediction'], args.plot)
    fpeak = forecast.loc[(forecast['k']>31) & (forecast['k']<43)]
    peak = (fpeak['demand'].max() - fpeak['prediction'].max() ) / fpeak['demand'].max()
    print('Peak prediction {} '.format(peak) )
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
