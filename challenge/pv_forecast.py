# python script to create a pv forecast from the previous pv and
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

parser = argparse.ArgumentParser(description='Create pv forecast.')
parser.add_argument('set', help='PV file eg set0')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method:' , default='simple' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
method = args.method
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# pv data
pv_filename = '{}pv_fixed_{}.csv'.format(output_dir, dataset)
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
print(pv)

# weather data (historic)
weather_filename = '{}weather_{}.csv'.format(output_dir, dataset)
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(weather)

# weather data (forecast)
forecast_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(forecast)

# Set up Naive PV forecast based on same as last week
pv['probability'] = 0.9
pv_forecast = pv[['pv_power','probability']].tail(7*48)
next_day = pv_forecast.last_valid_index() + pd.Timedelta(minutes=30)
last_day = pv_forecast.last_valid_index() + pd.Timedelta(days=7)
new_index = pd.date_range(start = next_day, end= last_day, freq='30min')
pv_forecast.index = new_index
pv_forecast.index.name = 'datetime'
print(pv_forecast)

# set up inputs
# weather for location 
# ( using location 2 since it seems most closely correlated )
#input_df = weather[['cs_ghi', 'sun2']].copy()
input_df = weather[['cs_ghi', 'sun2']].copy()
#input_df['period'] = input_df.index.hour * 2 + (input_df.index.minute / 30)
print(input_df)

# set up output
output = pv['pv_power']

# santity check
for column in input_df.columns:
    if input_df[column].isna().sum() >0:
        print("ERROR NaN in {}".format(column))
        quit()
if output.isna().sum() >0:
    print("ERROR NaN in output")
    quit()

if method == 'r2':
    rmodel = sm.OLS(output.values, sm.add_constant(weather['sun2'].values))
    residual_results = rmodel.fit()
    res_const = residual_results.params[0]
    res_grad = residual_results.params[1]
    print("Intercept {} Gradient {}".format(res_const, res_grad) )
    pv['prediction'] = weather['sun2'] * res_grad + res_const
#   pred = pv[['prediction', 'pv_power']]
#   print(pred['2018-06-01 10:00:00' : '2018-06-01 15:00:00'] )
    pv_forecast['pv_power'] = forecast['sun2'] * res_grad + res_const

    # plots
    if args.plot:
        plt.scatter(weather['sun2'], output, s=12, color='blue')
        x = np.array([weather['sun2'].min(),weather['sun2'].max()])
        y = res_const + res_grad * x
        plt.plot(x,y,color='green')
        plt.title('Regression sun2 v power')
        plt.xlabel('Weather Irradiance')
        plt.ylabel('Power ')
        plt.show()
        
        pv['pv_power'].plot(label='actual power', color='blue')
        pv['prediction'].plot(label='predicted power', color='red')
        plt.title('pv regression prediction')
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('PV Generation (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
    

if method == 'ann' or method == 'reg':
#   torch.autograd.set_detect_anomaly(True) 
    # Use the index from the pv to get weather inputs the same length
#   print('INPUTS')
#   inputs = input_df.loc[pv.index]
#   print(inputs)
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
#   train_ds[0:3]

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
        plt.title('pv ann convergence')
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.show()
#   hack to allow tensor plotting
#       torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
        pv['prediction'] = preds.detach().numpy() * pv['pv_power'].max()
        pv['pv_power'].plot(label='actual power', color='blue')
        pv['prediction'].plot(label='predicted power', color='red')
        plt.title('pv ann prediction')
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('PV Generation (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    # TODO use the model to do another prediction based on the future 
    # weather week.


output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}pv_forecast_{}.csv'.format(output_dir, dataset, method)

pv_forecast.to_csv(output_filename, float_format='%.2f')
