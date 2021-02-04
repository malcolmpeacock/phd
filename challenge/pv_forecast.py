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

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.act1 = nn.ReLU() # Activation function
#       self.linear2 = nn.Linear(3, 2)
        self.linear2 = nn.Linear(3, 1)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
    print('Training loss: ', loss_fn(model(inputs), targets))


# main program

# process command line

parser = argparse.ArgumentParser(description='Create pv forecast.')
parser.add_argument('pv', help='PV file')
parser.add_argument('weather', help='Weather file')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method:' , default='simple' )
parser.add_argument('--set', action="store", dest="set", help='Data set name' , default='fake' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)

args = parser.parse_args()
weather_file = args.weather
pv_file = args.pv
method = args.method
dataset = args.set

# read in the data
input_dir = "/home/malcolm/uclan/challenge/input/"

# pv data
pv_filename = input_dir + pv_file
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

print(pv)

# weather data - read in and up sample to 30 mins (with interpolation)
weather_filename = input_dir + weather_file
weather = pd.read_csv(weather_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
weather = weather.resample('30min').interpolate()

print(weather)

# error checks
n_missing = utils.missing_times(pv, '30min')
print('Number of missing pv rows: {}'.format(n_missing) )
print('NaNs pv_power_mw {} irradiance {} temp {}'.format(pv['pv_power_mw'].isna().sum(), pv['irradiance_Wm-2'].isna().sum(), pv['panel_temp_C'].isna().sum()) )

large_pv = pv['pv_power_mw'].max() * 0.8
small_pv = pv['pv_power_mw'].max() * 0.2
large_irrad = pv['irradiance_Wm-2'].max() * 0.8
small_irrad = pv['irradiance_Wm-2'].max() * 0.2
large_temp = pv['panel_temp_C'].max() * 0.8
small_temp = pv['panel_temp_C'].max() * 0.2

# pv large but irradiance small or temp small
pv_large = pv[pv['pv_power_mw']>large_pv]
suspect = pv_large[pv_large['irradiance_Wm-2']<small_irrad]
print('PV large but irradiance small {}'.format(len(suspect)) )
print(suspect)
suspect = pv_large[pv_large['panel_temp_C']<small_temp]
print('PV large but temp small {}'.format(len(suspect)) )
print(suspect)

# pv small but irradiance large or temp large
pv_small = pv[pv['pv_power_mw']<small_pv]
suspect = pv_small[pv_small['irradiance_Wm-2']>large_irrad]
print('PV small but irradiance large {}'.format(len(suspect)) )
print(suspect)
suspect = pv_small[pv_small['panel_temp_C']>large_temp]
print('PV small but temp large {}'.format(len(suspect)) )
print(suspect)

# fix errors
if dataset!= 'fake':
    pv['pv_power_mw']['2018-05-08 14:00:00'] = pv['pv_power_mw']['2018-05-08 13:00:00']
    pv['pv_power_mw']['2018-06-15 11:30:00'] = pv['pv_power_mw']['2018-06-15 11:00:00']
    pv['pv_power_mw']['2018-06-15 12:00:00'] = pv['pv_power_mw']['2018-06-15 12:30:00']

# plot pv
if args.plot:
    pv['pv_power_mw'].plot(label='pv power', color='blue')
    plt.title('pv')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('PV Generation (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    pv['irradiance_Wm-2'].plot(label='iradiance', color='blue')
    plt.title('PV System Measured Irradiance')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('Irradiance (MW)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

    pv['panel_temp_C'].plot(label='panel temp', color='blue')
    plt.title('Panel Temperature')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('temperature (degrees C)', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()

# Set up Naive PV forecast based on same as last week
pv['probability'] = 0.9
pv_forecast = pv[['pv_power_mw','probability']].tail(7*48)
#print(pv_forecast)
next_day = pv_forecast.last_valid_index() + pd.Timedelta(minutes=30)
#print(next_day)
last_day = pv_forecast.last_valid_index() + pd.Timedelta(days=7)
#print(last_day)
new_index = pd.date_range(start = next_day, end= last_day, freq='30min')
#print(new_index)
pv_forecast.index = new_index
print(pv_forecast)

# weather for location ( bilinear interpolation )
weather['temp'] = weather['temp_location3']
weather['solar'] = weather['solar_location3']
input_df = weather[['temp', 'solar']].copy()
input_df['period'] = input_df.index.hour * 2 + (input_df.index.minute / 30)
print(input_df)

if method == 'ann':
    # join the inputs onto input pv so its same length
#   inputs = pv.join(inputs, how='left')
    # then just include the columns we want
#   input_df = weather[['day', 'period', 'solar', 'temp']]
#   inputs = input_df[pv.index]
    print('INPUTS')
    inputs = input_df.loc[pv.index]
    print(inputs)
#   dataset = input_df.values
#   print(dataset)
#   X = dataset[:,0:4]
    # normalise the inputs (X)
    inputs['solar'] = inputs['solar'] / inputs['solar'].max()
    inputs['temp'] = inputs['temp'] / inputs['temp'].max()
    # normalise the output (Y)
    output = pv['pv_power_mw'] / pv['pv_power_mw'].max()

    inputs = torch.tensor(inputs.values.astype(np.float32))
#   X_train = torch.from_numpy(input_df['solar'].values.astype(np.float32)).view(-1,1)
    print("inputs")
    print(inputs)
#   targets = torch.tensor(output.values.astype(np.float32))
    targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
#   Y_train = torch.from_numpy(pv['pv_power_mw'].values.astype(np.float32)).view(-1,1)
    print("targets")
    print(targets)
    torch.manual_seed(1)    # reproducible
    train_ds = TensorDataset(inputs, targets)
    train_ds[0:3]

    # Define data loader
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))
    # model using ANN
    model = SimpleNet()

    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=1e-5)


    # Define loss function
    loss_fn = F.mse_loss

    loss = loss_fn(model(inputs), targets)
    print(loss)
    # Train the model for 100 epochs
    fit(100, model, loss_fn, opt)
    # prediction
    preds = model(inputs)
    print(preds)
    print(targets)
    if args.plot:
#   hack to allow tensor plotting
#       torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it
        pv['prediction'] = preds.detach().numpy() * pv['pv_power_mw'].max()
        pv['pv_power_mw'].plot(label='actual power', color='blue')
        pv['prediction'].plot(label='predicted power', color='red')
        plt.title('pv ann prediction')
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('PV Generation (MW)', fontsize=15)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()


output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}pv_forecast_{}'.format(output_dir, dataset, method)

pv_forecast.to_csv(output_filename)
