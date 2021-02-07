# python script to create a demand forecast from the previous demand and
# the weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
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
def fit(num_epochs, model, loss_fn, opt):
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
    print('Training loss: ', loss_fn(model(inputs), targets))
    return loss_history



# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method:' , default='simple' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)


args = parser.parse_args()
method = args.method
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# demand data
demand_filename = '{}demand_fixed_{}.csv'.format(output_dir, dataset)
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(demand)

# weather data (historic)
weather_filename = '{}weather_{}.csv'.format(output_dir, dataset)
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

weather['demand'] = demand

print(weather)


# weather data (forecast)
forecast_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(forecast)

# naive forecast based on the previous week

demand_forecast = demand.tail(7*48)
next_day = demand_forecast.last_valid_index() + pd.Timedelta(minutes=30)
last_day = demand_forecast.last_valid_index() + pd.Timedelta(days=7)
new_index = pd.date_range(start = next_day, end= last_day, freq='30min')
demand_forecast.index = new_index

forecast['demand'] = demand_forecast

# extract k=32,42 from the advance weather forecast week
kabove31 = forecast['k'] > 31
kabove31_forecast = forecast[kabove31]
kbelow43 = kabove31_forecast['k'] < 43
forecast = kabove31_forecast[kbelow43]
print(forecast)

# extract k=32,42 from the historic weather
kabove31 = weather['k'] > 31
kabove31_forecast = weather[kabove31]
kbelow43 = kabove31_forecast['k'] < 43
weather = kabove31_forecast[kbelow43]
print(weather)

if args.plot:
    day1 = weather['2018-06-04 00:00:00' : '2018-06-04 23:30:00']
    day2 = weather['2018-06-05 00:00:00' : '2018-06-05 23:30:00']
    day3 = weather['2018-06-06 00:00:00' : '2018-06-06 23:30:00']
    day4 = weather['2018-06-07 00:00:00' : '2018-06-07 23:30:00']
    day5 = weather['2018-06-08 00:00:00' : '2018-06-08 23:30:00']
    plt.plot(day1['k'], day1['demand'], label='monday', color='red')
    plt.plot(day2['k'], day2['demand'], label='tuesday', color='blue')
    plt.plot(day3['k'], day3['demand'], label='wednesday', color='green')
    plt.plot(day4['k'], day4['demand'], label='thursday', color='orange')
    plt.plot(day5['k'], day5['demand'], label='friday', color='purple')
    plt.title('Demand profiles for 1 week')
    plt.xlabel('K period of the day', fontsize=15)
    plt.ylabel('Demand (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

input_df = weather[['k', 'temp2', 'sun2', 'holiday']].copy()

print(input_df)

# set up output
output = weather['demand']

# santity check
for column in input_df.columns:
    if input_df[column].isna().sum() >0:
        print("ERROR NaN in {}".format(column))
        quit()
if output.isna().sum() >0:
    print("ERROR NaN in output")
    quit()

if method == 'ann' or method == 'reg':
    # normalise the inputs (X)
    for column in input_df.columns:
        input_df[column] = input_df[column] / input_df[column].max()
    # normalise the output (Y)
    output = output / output.max()

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

    # Define data loader
#   batch_size = 1
#   batch_size = 5
    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)
    # model using regression
    if method == 'reg':
        model = nn.Linear(num_inputs,1)
    else:
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

    losses = fit(num_epochs, model, loss_fn, opt)
    # prediction
    preds = model(inputs)
#   print(preds)
#   print(targets)
    if args.plot:
        plt.plot(losses)
        plt.title('demand ann convergence')
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.show()
#   hack to allow tensor plotting
#       torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
        prediction = preds.detach().numpy() * output.max()
        plotdf = weather[['demand','k']]
        plotdf['prediction'] = prediction
        plt.plot(plotdf['k'], plotdf['demand'], label='actual demand', color='red')
        plt.plot(plotdf['k'], plotdf['prediction'] ,label='predicted demand', color='blue')
        plt.title('demand ann prediction')
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    # TODO use the model to do another prediction based on the future 
    # weather week.

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}demand_forecast_{}.csv'.format(output_dir, dataset, method)

demand_forecast.to_csv(output_filename, float_format='%.2f')
