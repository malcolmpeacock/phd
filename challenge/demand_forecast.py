# python script to create a demand forecast from the previous demand and
# the weather

# contrib code
import sys
import pandas as pd
from datetime import datetime
from datetime import timedelta
from datetime import date
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import gpytorch
# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
# Import nn.functional
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.signal import gaussian
from scipy.ndimage import filters

# custom code
import utils

# loss function for weighted L1
def loss_wl1(X,Y,W):
    lossn =  torch.mean(torch.abs(X - Y) * W)
    return lossn

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

# class to create ANN

class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self,num_inputs,num_outputs,num_hidden):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        # Activation function
#       self.act1 = nn.LeakyReLU() 
        self.act1 = nn.Sigmoid() 
        self.act2 = nn.LeakyReLU() 
#       self.act1 = nn.ReLU() 
#       Layer 1
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
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        loss_history.append(loss.item() )
    return loss_history

# train the model using weighted loss
def wfit(num_epochs, model, loss_fn, opt, train_dl, w):
    loss_history=[]
    for epoch in range(num_epochs):
        # each batch in the training ds
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb, w)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        # report at each epoc
#       print('epoch {:03d}, loss {:.7f}'.format(epoch, loss.item()))
        sys.stdout.write('\repoch {:03d}, loss {:.7f} '.format(epoch, loss.item()))
        loss_history.append(loss.item() )
    sys.stdout.flush()
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

#   closest_days = utils.find_closest_days(day, days, forecast, df, 'tempm', 10, True, True)
#   closest_days = utils.find_closest_days(day, days, forecast, df, 'tempm', 10, True, False)
    df_k = df[ (df['k'] > 31) & (df['k'] < 43)]
    f_k = forecast[ (forecast['k'] > 31) & (forecast['k'] < 43)]
    closest_days = utils.find_closest_days(day, days, f_k, df_k, 'tempm', 10, True, False)
    new_day = utils.create_day(closest_days.index, df, 'demand')
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = new_day['demand'].values
    probability = 0.8
    forecast.loc[day.strftime('%Y-%m-%d'), 'probability'] = probability

def find_closest_peak_days(df, forecast, day, peak):
    days = pd.Series(df.index.date).unique()
    df_k = df[ (df['k'] > 31) & (df['k'] < 43)]
    f_k = forecast[ (forecast['k'] > 31) & (forecast['k'] < 43)]
    # find the 30 days with peak demand closest to peak
    closest_peak_days = utils.find_closest_days_max(days, df_k, 'demand', peak, 30)
    # from those 30 days, find the 10 with the closest temperature pattern
    closest_days = utils.find_closest_days(day, closest_peak_days, f_k, df_k, 'tempm', 10, True, False)
    # create a new day using the mean demands of those 10
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

def reg_inputs(df_nw1, df_nwn):
    # dropping the first week
    # temperatures for a range of dsks
    dsk_data={}
    for dsk in range(32,44):
        dsk_df = df_nw1[ df_nw1['dsk'] == dsk]
        dsk_data['tk{}'.format(dsk)] = dsk_df['tempm'].values
        # Temperature Squared
        dsk_data['tsqdk{}'.format(dsk)] = dsk_df['tsqd'].values
        # TH  - temperature * irradiance 
        dsk_data['tsdk{}'.format(dsk)] = dsk_df['ts'].values
        # Irradiance (sun light? ) 
        dsk_data['sk{}'.format(dsk)] = dsk_df['sunw'].values
        # data from last week
        dsk_dfl = df_nwn[ df_nwn['dsk'] == dsk]
        dsk_data['tlk{}'.format(dsk)] = dsk_dfl['tempm'].values
        dsk_data['dmk{}'.format(dsk)] = dsk_dfl['demand'].values
        
    input_df = pd.DataFrame(dsk_data)
    # other columns
    dsk_df = df_nw1[ df_nw1['dsk'] == 32]
    base_columns = ['season', 'week', 'month']
    for col in base_columns:
        input_df[col] = dsk_df[col].values
    print(input_df)
    return input_df

def forecast_new_inputs(df):
    columns = ['sh', 'season', 'week', 'month']
    input_df = df[columns].resample('D', axis=0).first().dropna()
    input_df['avtemp'] = df['tempm'].resample('D', axis=0).mean().dropna()
    input_df['maxtemp'] = df['tempm'].resample('D', axis=0).max().dropna()
    input_df['avsus'] = df['sunm'].resample('D', axis=0).mean().dropna()
    input_df['maxsun'] = df['sunm'].resample('D', axis=0).max().dropna()
#   print(input_df)
    return input_df

# new - regression to find the peak and sum. Then use sdays to find  
#       days with this peak with matching temperature profile
def forecast_new(df, forecast, day, seed, num_epochs, ann):

    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    # days with the same dtype
    dfd = df[df['dtype'] == forecast_day['dtype'].iloc[0]]
    # periods of interest
    df_k = dfd[ (dfd['k'] > 31) & (dfd['k'] < 43)]
    input_df = forecast_new_inputs(df_k)
#   print(input_df)

    # outputs - demand for the k periods of interest
    data = {}
    data['peak'] = df_k['demand'].resample('D', axis=0).max().dropna()
    output_df = pd.DataFrame(data).dropna()
#   print(output_df)
        
    # store maximum values
    input_max = utils.df_normalise(input_df)
    output_max = utils.df_normalise(output_df)
    # santity check
    utils.sanity_check(input_df)
    utils.sanity_check(output_df)

    inputs = torch.tensor(input_df.values.astype(np.float32))
    targets = torch.tensor(output_df.values.astype(np.float32))
    torch.manual_seed(seed)    # reproducible
    train_ds = TensorDataset(inputs, targets)

    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)
    num_outputs = len(output_df.columns)
    if ann:
#       model = SimpleNet(num_inputs, num_outputs, num_inputs)
        model = SimpleNet(num_inputs, num_outputs, 50)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    else:
        model = nn.Linear(num_inputs, num_outputs)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)


    # Define loss function
#   loss_fn = F.mse_loss
    loss_fn = F.l1_loss

    loss = loss_fn(model(inputs), targets)
#   print(loss)
    # Train the model
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
#   preds = model(inputs)
#   print(preds)
    # prediction
#   print(forecast_day)
    # set up the values to forecast
    input_f = forecast_new_inputs(forecast_day)
#   print(input_f)
    # normalise the inputs (using same max as for the model)
    utils.df_normalise_by(input_f, input_max)
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
#   print(preds)
    # denormalize using df for the original model
    vals = preds.detach().numpy()[0]
#   print(vals)
    count=0
    for column in output_max:
        vals[count] = vals[count] * output_max[column]
        count+=1
  
    # if we have demand, then get the original
    if 'demand' in forecast_day.columns:
        f_k = forecast_day[ (forecast_day['k'] > 31) & (forecast_day['k'] < 43)]
        original_peak = f_k['demand'].resample('D', axis=0).max().values[0]
    else:
        original_peak = 0.0
    return losses, vals[0], original_peak

# nreg - new regression - with all 11 periods of interest as seperate 
#        inputs and outputs as per bogdan
def forecast_nreg(df, forecast, day, seed, num_epochs, ann):
    # only use days of the same type
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    dfd = df[df['dtype'] == forecast_day['dtype'].iloc[0]]
    # drop the first week so we can use data from last week
    df_nw1 = dfd.drop(dfd.head(48*7).index)
    # drop the last week so we can use data from last week
    df_nwn = dfd.drop(dfd.tail(48*7).index)
    input_df = reg_inputs(df_nw1, df_nwn)

    # outputs - demand for the k periods of interest
    k_demand={}
    for k in range(32,44):
        k_df = df_nw1[ df_nw1['k'] == k]
        k_demand['demand{}'.format(k)] = k_df['demand'].values
    output_df = pd.DataFrame(k_demand)

    # store maximum values
    input_max = utils.df_normalise(input_df)
    output_max = utils.df_normalise(output_df)
    # santity check
    utils.sanity_check(input_df)
    utils.sanity_check(output_df)
        
    inputs = torch.tensor(input_df.values.astype(np.float32))
    targets = torch.tensor(output_df.values.astype(np.float32))
    torch.manual_seed(seed)    # reproducible
    train_ds = TensorDataset(inputs, targets)

    batch_size = 48
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))

    num_inputs = len(input_df.columns)
    num_outputs = len(output_df.columns)
    if ann:
        model = SimpleNet(num_inputs, num_outputs, 40)
    else:
        model = nn.Linear(num_inputs, num_outputs)

#   opt = torch.optim.SGD(model.parameters(), lr=1e-5)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Define loss function
#   loss_fn = F.mse_loss
#   loss_fn = loss_max1
    loss_fn = F.l1_loss

    loss = loss_fn(model(inputs), targets)
#   print(loss)
    # Train the model
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
    preds = model(inputs)
#   print(preds)
    # prediction
#   print(forecast_day)
    # we need a previous day to get demand from, but in assessing the method
    # there might not be one if it was removed due to bad data. So we then 
    # look further back
    first_day = dfd.first_valid_index().date()
    previous_found = False
    day_last_week  = day
    while not previous_found:
        day_last_week  = day_last_week - pd.Timedelta(days=7)
        print('Looking to base demand of previous week on {}'.format(day_last_week))
        if day_last_week < first_day:
            print('Previous day for demand before start of data!!!!')
            quit()
        if day_last_week.strftime('%Y-%m-%d') in dfd.index:
            print('Found using {}'.format(day_last_week))
            previous_day = dfd.loc[day_last_week.strftime('%Y-%m-%d')].copy()
            if len(previous_day) > 0:
                previous_found = True
        else:
            print('Not Found')

#   print(previous_day)
    # set up the values to forecast
    input_f = reg_inputs(forecast_day, previous_day)
#   print(input_f)
    # normalise the inputs (using same max as for the model)
    for column in input_f.columns:
        if input_max[column]>0:
            input_f[column] = input_f[column] / input_max[column]
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
#   print(preds)
    vals = preds.detach().numpy()[0]
#   print(vals)
#   print(type(vals))
    count=0
    for column in output_max:
        vals[count] = vals[count] * output_max[column]
        count+=1
    # smooth the output
#   b = gaussian(39, 1)
#   vals = filters.convolve1d(vals, b/b.sum())

    count=0
    prediction_values = np.zeros(48)
#   print(prediction_values)
#   print(type(prediction_values))
    for k in range(32,44):
        prediction_values[k-1] = vals[count]
        count+=1
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = prediction_values

    return losses

def sanity_check(df):
    for column in df.columns:
        if df[column].isna().sum() >0:
            print("ERROR NaN in {}".format(column))
            quit()

def normalise_df(df):
    df_max = {}
    # normalise the inputs
    for column in df.columns:
        df_max[column] = df[column].max()
        df[column] = df[column] / df[column].max()
    return df_max

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Gaussian Process Regression.
def forecast_gpr(df, forecast, day, seed, num_epochs):

    input_df = reg_inputs(df)
    print(input_df)

    # outputs - demand for the k periods of interest
    k_demand={}
    for k in range(32,44):
        k_df = df[ df['k'] == k]
        k_demand['demand{}'.format(k)] = k_df['demand'].values
    output_df = pd.DataFrame(k_demand)
    print(output_df)
        
    # store maximum values and normalise
    input_max = normalise_df(input_df)
    output_max = normalise_df(output_df)
    # santity check
    sanity_check(input_df)
    sanity_check(output_df)

    inputs = torch.tensor(input_df.values.astype(np.float32))
    print(inputs)
    targets = torch.tensor(output_df.values.astype(np.float32))
    print(targets)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(inputs, targets, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_epochs):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(inputs)
        print(output)
        print(targets)
        # Calc loss and backprop gradients
        loss = -mll(output, targets)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    # prediction
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    # set up the values to forecast
    input_f = reg_inputs(forecast_day)
    # normalise the inputs (using same max as for the model)
    for column in input_f.columns:
        input_f[column] = input_f[column] / input_max[column]
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.tensor(input_f.values.astype(np.float32))
        print(test_x)
        observed_pred = likelihood(model(test_x))
        print(observed_pred)

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

    print(lower)
    print(upper)
    quit()

#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
#   print(preds)
    # denormalize using df for the original model
    vals = preds.detach().numpy()[0]
#   print(vals)
#   print(type(vals))
    count=0
    for column in output_max:
        vals[count] = vals[count] * output_max[column]
        count+=1
    count=0
    prediction_values = np.zeros(48)
#   print(prediction_values)
#   print(type(prediction_values))
    for k in range(32,44):
        prediction_values[k-1] = vals[count]
        count+=1
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = prediction_values

    return losses

def set_weight(dfd, forecast_day):
    # set weight to use in the weighted loss function to give less importance
    # to some values

    # set weight based on closeness to day of year.
    # closer days should have a greater weight.
    # difference between day of year and that of the forecast day
    # doy_diff = np.abs(dfd['doy'].values - forecast_day['doy'].iloc[0])
    # to cope with days at the end of one year being close to those at the
    # start of the next
    # weight = np.minimum( doy_diff, np.abs(doy_diff - 365) )

    # set weight based on the week counter in the data
    # ( how far away the week we are forecasting is away )
    weight = np.abs(dfd['week'].values - forecast_day['week'].iloc[0])
#   dfd['weight'] = weight
#   dfd.loc['weight'] = weight
    # normalise
    weight = weight / np.max(weight)
    #dfd['weight'] = dfd['weight'] / dfd['weight'].max()
#   dfd.loc['weight'] = dfd['weight'] / dfd['weight'].max()

    # subtract from 1 so that higher values have less impact on the loss.
    # multiply by ww so that the further off days have some impact not zero.
    weight = 1.0 - ( weight * ww)
    dfd['weight'] = weight
    print(dfd['weight'])

def forecast_pub_hol(dsk_df, dsk_f, plot):
    demands = dsk_df['demand'].values
    temps = dsk_df['tempm'].values
    # Fit line through the points - the add constant bit gives us 
    # the intercept as well as the gradient of the fit line.
    rmodel = sm.OLS(demands, sm.add_constant(temps))
    residual_results = rmodel.fit()
#   print(residual_results.summary())
    res_const = residual_results.params[0]
    res_grad = residual_results.params[1]
    print('Gradient {} intercept {}'.format(res_grad, res_const) )
    if plot:
        # Fit of residuals line
        x = np.array([min(temps),max(temps)])
        y = res_const + res_grad * x
        fig, ax = plt.subplots()
        ax.scatter(temps, demands)
        plt.title('Sunday Temp vs demand')
        plt.xlabel('Temperature ', fontsize=15)
        plt.ylabel('Demand (MWh)', fontsize=15)
#       for i in range(len(labels)):
#           ax.annotate(labels[i], (temps[i], demands[i]))
        plt.plot(x,y,color='red')
        plt.show()

    f_temp = dsk_f['tempm'].values
    print(f_temp)
    prediction_values = res_const + res_grad * f_temp[0]
    return prediction_values


# reg - regression - with different models for each k=32,42
#        regl  - first 
#        regm  - as submitted for set2
#        regd  - only look at same day type

def forecast_reg(df, forecast, day, method, plot, seed, num_epochs, period, ki, ann, ww, ploss):
    pred_values=[]
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    # look for days of similar type.
    fd_type = forecast_day['dtype'].iloc[0]
    if method == 'regd':
        dfd = df[df['dtype'] == fd_type]
    else:
        dfd = df

    # set weight to use in the weighted loss function to give less importance
    # to some values

    # set weight based on closeness to day of year.
    # closer days should have a greater weight.
    # difference between day of year and that of the forecast day
    # doy_diff = np.abs(dfd['doy'].values - forecast_day['doy'].iloc[0])
    # to cope with days at the end of one year being close to those at the
    # start of the next
    # weight = np.minimum( doy_diff, np.abs(doy_diff - 365) )

    # set weight based on the week counter in the data
    # ( how far away the week we are forecasting is away )
    weight = np.abs(dfd['week'].values - forecast_day['week'].iloc[0])
#   dfd['weight'] = weight
#   dfd.loc['weight'] = weight
    # normalise
    weight = weight / np.max(weight)
    #dfd['weight'] = dfd['weight'] / dfd['weight'].max()
#   dfd.loc['weight'] = dfd['weight'] / dfd['weight'].max()

    # subtract from 1 so that higher values have less impact on the loss.
    # multiply by ww so that the further off days have some impact not zero.
    weight = 1.0 - ( weight * ww)
    dfd['weight'] = weight
    print(dfd['weight'])
    forecast_day['weight'] = 1.0

    if period == all:
        # for each k period, train a seperate model ...
        for index, row in forecast_day.iterrows():
            dsk = row['dsk']
            if ki and (row['k'] < 32 or row['k'] > 42):
                prediction_values = 1.0
            else:
                print('Period k {} dsk {}'.format(row['k'],dsk) )
                dsk_df = dfd[dfd['dsk'] == dsk]
                dsk_f = forecast_day[forecast_day['dsk'] == dsk]
                # if pub hol or christmas then we don't have enough so interp
                if fd_type >= 7:
                    prediction_values = forecast_pub_hol(dsk_df, dsk_f, ploss)
                else:
                    prediction_values = forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ki, ann)
                if math.isnan(prediction_values):
                    print('WARNING NaN replaced by interpolation at period {}'.format(row['k']))
                if not math.isfinite(prediction_values):
                    print('WARNING Inf replaced by interpolation at period {}'.format(row['k']))
                    prediction_values = float("NaN")
            pred_values.append(prediction_values)
        pred_series = pd.Series(pred_values, index=forecast_day.index)
        # Replace any NaNs at the start or end with adjacent values
        pred_series = pred_series.fillna(method='bfill')
        pred_series = pred_series.fillna(method='ffill')
        # Replace any NaNs in the middle by interpolation.
        pred_series = pred_series.interpolate()
        forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = pred_series
    else:
        dsk = int(period)
        dsk_df = dfd[dfd['dsk'] == dsk]
        dsk_f = forecast_day[forecast_day['dsk'] == dsk]
        if fd_type == 7:
            prediction_values = forecast_pub_hol(dsk_df, dsk_f, ploss)
        else:
            prediction_values = forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ki, ann)
        print('Predicted value {} for period {}'.format(prediction_values, period))
        quit()

def forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ki, ann):
    # set up inputs
    if method == 'regd':
#       input_columns = ['tempm']
        input_columns = ['tempm', 'sunm', 'season', 'zenith', 'tsqd', 'ts', 'month', 'tm', 'weight', 'sh', 'dailytemp', 'sfactor']
#       With the weighted loss function batch size has to be 1
        batch_size = 1
        rate = 1e-3

    if method == 'regl':
#       input_columns = ['tempm', 'zenith']
#       input_columns = ['tempm', 'zenith', 'holiday']
#       input_columns = ['tempm', 'sun2', 'holiday', 'nothol']
        input_columns = ['tempm', 'zenith', 'holiday', 'nothol']
        batch_size = 1
        rate = 1e-4

    if method == 'regm':
# sunw causes nans in predicition? 
#       input_columns = ['tempm', 'zenith', 'holiday', 'nothol', 'tsqd', 'th', 'tnh', 'sunw', 'sh']
# this is ok, but a bit worse
#       input_columns = ['tempm', 'zenith', 'holiday', 'nothol', 'tsqd', 'th', 'tnh', 'sunw']
        input_columns = ['tempm', 'sun2', 'holiday', 'nothol', 'season', 'zenith']
        # days of the week 1-0 flags
        for wd in range(7):
            wd_key = 'wd{}'.format(wd)
            input_columns.append(wd_key)

        batch_size = 1
        rate = 1e-4

    input_df = dsk_df[input_columns].copy()

    # set up output
    output_column = 'demand'
    output = dsk_df[output_column].copy()

    # store maximum values
    input_max = utils.df_normalise(input_df)
    # santity check
    utils.sanity_check(input_df)

    # store maximum values
#   input_max = {}
    # normalise the inputs
#   for column in input_df.columns:
#       input_max[column] = input_df[column].max()
#       if input_df[column].max() > 0.0:
#           input_df[column] = input_df[column] / input_df[column].max()
#       else:
#           input_df[column] = 0.0
    # normalise the output (Y)
    output_max = output.max()
    if output_max>0:
        output = output / output_max

    # santity check
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
    if ann:
        model = SimpleNet(num_inputs,1,20)
        loss_fn = F.l1_loss
    else:
        model = nn.Linear(num_inputs,1)

    if method == 'regl':
    # model using regression
#       loss_fn = F.mse_loss
        loss_fn = F.l1_loss
    if method == 'regm':
    # model using regression
#       loss_fn = F.mse_loss
        loss_fn = F.l1_loss

    if method == 'regd':
#       loss_fn = F.l1_loss
        loss_fn = loss_wl1
        weights = torch.tensor(dsk_df['weight'].values.astype(np.float32)).view(-1,1)
        opt = torch.optim.SGD(model.parameters(), lr=rate)
#       opt = torch.optim.Adam(model.parameters(), lr=rate)
        loss = loss_fn(model(inputs), targets, weights)
        losses = wfit(num_epochs, model, loss_fn, opt, train_dl, weights)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=rate)
        loss = loss_fn(model(inputs), targets)
        losses = fit(num_epochs, model, loss_fn, opt, train_dl)

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
        if input_max[column]>0:
            input_f[column] = input_f[column] / input_max[column]
    f_inputs = torch.tensor(input_f.values.astype(np.float32))
#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
#   print(preds)
    # denormalize using df for the original model
    prediction_values = preds.detach().numpy() * output_max
#   print(prediction_values)

    if args.ploss:
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
    model = SimpleNet(num_inputs,1,1)

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

def additional_values(df):

    # holiday
    df['holiday'] = 0
    df.loc[(df['wd']>4) | (df['ph']==1), 'holiday' ] = 1
    # not holiday
    df['nothol'] = 0
    df.loc[(df['wd']<2) | (df['ph']==0), 'nothol' ] = 1

    # seasonal factor:
    #  summer =1, autumn or spring=2, winter=3
    df['sfactor'] = 1
    df.loc[(df['season']==3) | (df['season']==1), 'sfactor' ] = 2
    df.loc[(df['season']==0) , 'sfactor' ] = 3

    df['tsqd'] = df['tempm'] * df['tempm']
    df['th'] = df['tempm'] * df['holiday']
    df['ts'] = df['tempm'] * df['sunm']
    df['tnh'] = df['tempm'] * df['nothol']
    df['tm'] = df['tempm'] * df['month']

    # days of the week 1-0 flags
    for wd in range(7):
        wd_key = 'wd{}'.format(wd)
        df[wd_key] = 0
        df.loc[df['wd']==wd, wd_key] = 1
# main program

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('set', help='input data eg set0')
parser.add_argument('--method', action="store", dest="method", help='Forecasting method: reg2, reg, ann, sday' , default='simple' )
parser.add_argument('--day', action="store", dest="day", help='Day to forecast: set=read the set forecast file, first= first day, last=last day, all=loop to forecast all days based on the others, otherwise integer day' , default='set' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--ploss', action="store_true", dest="ploss", help='Plot the loss', default=False)
parser.add_argument('--ann', action="store_true", dest="ann", help='Replace regression with ANN', default=False)
parser.add_argument('--mname', action="store_true", dest="mname", help='Name the output file using the method', default=False)
parser.add_argument('--seed', action="store", dest="seed", help='Random seed, default=1.0', type=float, default=1.0)
parser.add_argument('--ww', action="store", dest="ww", help='Loss Weight weight=0.5', type=float, default=0.5)
parser.add_argument('--epochs', action="store", dest="epochs", help='Number of epochs', type=int, default=100)
parser.add_argument('--nnear', action="store", dest="nnear", help='Number of days when using the --day near option', type=int, default=10)
parser.add_argument('--period', action="store", dest="period", help='Period k to forecast', type=float, default=all)
parser.add_argument('--step', action="store", dest="step", help='If using days=all or step only do every step days', type=int, default=1)
parser.add_argument('--use', action="store", dest="use", help='Number of days data to use, 0=all', type=int, default=0)
parser.add_argument('--ki', action="store_true", dest="ki", help='Only forecast K of interest.', default=False)

args = parser.parse_args()
method = args.method
dataset = args.set

# read in the data
output_dir = "/home/malcolm/uclan/challenge/output/"

# merged data file
merged_filename = '{}merged_{}.csv'.format(output_dir, dataset)
df = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# remove some days of data ( 48 half hours per day)
if args.use > 0:
    df = df.tail(args.use * 48)

# additional values
additional_values(df)

print(df)

# weather data (forecast)
forecast_filename = '{}forecast_{}.csv'.format(output_dir, dataset)
forecast = pd.read_csv(forecast_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

additional_values(forecast)


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
        print('Number of days in data {}'.format(len(days)) )
        if args.day[0:4] == 'week' :
            start_date = args.day[4:]
            print('Week starting {}'.format(start_date))
            start_date = date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]) )
            end_date = start_date + pd.Timedelta(days=6)
            day_text = start_date.strftime("%Y-%m-%d")
            day_start = day_text + ' 00:00:00'
            day_end = end_date.strftime("%Y-%m-%d") + ' 23:30:00'
            forecast = df.loc[day_start : day_end]
            forecast = forecast[columns]
        else:
            if args.day[0:4] == 'date' :
                start_date = args.day[4:]
                print('One date to forecast {}'.format(start_date))
                start_date = date(int(start_date[0:4]), int(start_date[4:6]), int(start_date[6:8]) )
                day_text = start_date.strftime("%Y-%m-%d")
                day_start = day_text + ' 00:00:00'
                day_end = day_text + ' 23:30:00'
                forecast = df.loc[day_start : day_end]
                forecast = forecast[columns]
            else:
                if args.day[0:4] == 'near' :
                    forecast = df[columns].copy()
                    near_date = args.day[4:]
                    print('Day of the year near to {}'.format(near_date))
                    near_date_doy = date(2018, int(near_date[0:2]), int(near_date[2:4]) ).timetuple().tm_yday
                    near_range = args.nnear
                    # in case we are close to the end of the year
                    if near_date_doy > 366-near_range:
                        near_date_doy = near_date_doy - 366
                    print(near_date_doy)
                    for day in days:
                        day_str = day.strftime('%Y-%m-%d')
                        doy = day.timetuple().tm_yday
                        if abs(near_date_doy - doy) > near_range:
                            forecast.drop(forecast.loc[day_str].index, inplace=True)
                else:
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

#print(forecast)

peak_pred = {'peak_actual' : [], 'peak_predict' : [], 'sum_actual' : [], 'sum_predict' : [] }

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
            forecast_reg(history, forecast, day, method, args.plot, args.seed, args.epochs, args.period, args.ki, args.ann, args.ww, args.ploss)

        if method == 'skt':
            forecast_skt(df, forecast, day)
        if method == 'nreg':
            losses = forecast_nreg(history, forecast, day, args.seed, args.epochs, args.ann)
            if args.ploss:
                plt.plot(losses)
                plt.title('demand nreg convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

        if method == 'gpr':
            losses = forecast_gpr(history, forecast, day, args.seed, args.epochs)
            if args.ploss:
                plt.plot(losses)
                plt.title('demand gpr convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

        if method == 'ann':
            losses = forecast_ann(history, forecast, day, args.seed, args.epochs)
            if args.ploss:
                plt.plot(losses)
                plt.title('demand ann convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

        if method == 'new':
            losses, pred_peak, ac_peak = forecast_new(history, forecast, day, args.seed, args.epochs, args.ann)
            if args.ploss:
                plt.plot(losses)
                plt.title('demand peak convergence')
                plt.xlabel('Epochs', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                plt.show()

#           print('Peak actual {} prediction {} Sum actual {} prediction {}'.format(ac_peak, pred_peak, ac_sum, pred_sum) )
            print('Peak actual {} prediction {}'.format(ac_peak, pred_peak) )
            peak_pred['peak_actual'].append(ac_peak)
            peak_pred['peak_predict'].append(pred_peak)
#           peak_pred['sum_actual'].append(ac_sum)
#           peak_pred['sum_predict'].append(pred_sum)
            # find closest days with this peak demand with similar temp patterns
            find_closest_peak_days(history, forecast, day, pred_peak)

#print(forecast)

# metrics
if 'demand' in forecast.columns:
    if len(peak_pred['peak_actual']) >2:
        peak_actual = pd.Series(peak_pred['peak_actual'])
        peak_predict = pd.Series(peak_pred['peak_predict'])
        print('Prediction of peak demand')
        utils.print_metrics(peak_actual, peak_predict, args.plot)
    if args.ki:
        kf = forecast[ (forecast['k'] > 31) & (forecast['k'] < 43)]
#       print(kf['demand'])
#       print(kf['prediction'])
        print('Metrics for k>31 and k<43')
        utils.print_metrics(kf['demand'], kf['prediction'], args.plot)
        print('Metrics for whole day')
        utils.print_metrics(forecast['demand'], forecast['prediction'], False)
    else:
        utils.print_metrics(forecast['demand'], forecast['prediction'], args.plot)
    fpeak = forecast.loc[(forecast['k']>31) & (forecast['k']<43)]
    peak = (fpeak['demand'].max() - fpeak['prediction'].max() ) / fpeak['demand'].max()
    print('Peak prediction {} '.format(peak) )
    if args.plot:
        forecast['demand'].plot(label='actual demand', color='blue')
        forecast['prediction'].plot(label='predicted demand', color='red')
        plt.title('demand prediction : '+method)
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='lower left', fontsize=15)
        plt.show()

output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}demand_forecast_{}.csv'.format(output_dir, dataset)
if args.mname:
    output_filename = '{}demand_forecast_{}_{}.csv'.format(output_dir, method, dataset)
forecast.to_csv(output_filename, float_format='%.2f')

# only demand for bogdan
forecast = forecast['prediction']
forecast = forecast.squeeze()
forecast = forecast.rename('demand_forecast')
forecast.index.rename('datetime', inplace=True)

output_filename = '{}demand_forecast_{}_only.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')
