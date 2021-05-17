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
#import gpytorch
# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
# Import nn.functional
import torch.nn.functional as F
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
#from scipy.signal import gaussian
#from scipy.ndimage import filters
import sklearn.gaussian_process as gp
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import seaborn as sns
from sklearn.linear_model import LassoCV, Lasso
#from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

# custom code
import utils

def select_features(input_df,output,plot):
    print('Select Features')
    df = input_df.copy()
    df['output'] = output

    # Using Pearson Correlation
    cor = df.corr()
    if plot:
        plt.figure(figsize=(12,10))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()

    # Correlation with output variable
    cor_target = abs(cor["output"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.2]
    print(relevant_features)

    X = input_df.values
    y = output.values
    reg = LassoCV(max_iter=8000)
    reg.fit(X, y)
#   print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
#   print("Best score using built-in LassoCV: %f" %reg.score(X,y))
    coef = pd.Series(reg.coef_, index = input_df.columns)
    imp_coef = coef.sort_values()
#   print(imp_coef)
    if plot:
        matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
        imp_coef.plot(kind = "barh")
        plt.title("Feature importance using Lasso Model")
        plt.show()
    return coef

# using sklearn GPR to predict the demand for one k period

def period_gpr(input_df, output, input_f, slope):
    X_train = input_df
    y_train = output
    X_test = input_f
#   kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    if slope:
        kernel = gp.kernels.ConstantKernel(1.0, (1e-2, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e13))
    else:
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e8))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=False, random_state=0)

    model.fit(X_train, y_train)
    sk_pred, std = model.predict(X_test, return_std=True)
#   sd factor +3.0 seems good for set 2 but bad for set 3!
    sdf = 0.0
#   print(sk_pred,std)
#   pred = sk_pred[0] - std[0]*2.0
    pred = sk_pred[0] + std[0]*sdf
#   pred = sk_pred[0]
    return pred

# using sklearn GBR to predict the demand for one k period

def period_gbr(input_df, output, input_f):
    X_train = input_df
    y_train = output
    X_test = input_f
    print('Creating Regressor ...')
    # loss='ls' or 'lad'
    model = GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=3, loss='lad')
    print('Fitting model ...')
    model.fit(X_train, y_train)
    sk_pred = model.predict(X_test)
    print(sk_pred)
    return sk_pred

# using sklearn Random Forest to predict the demand for one k period

def period_rf(input_df, output, input_f):
    X_train = input_df
    y_train = output
    X_test = input_f
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
#   print(y_pred)
    return y_pred

# metric of how close the peak reduction score for the forecast demand
# is to the peak reduction score obtained from the actual demand.

def print_peak_metric(actual, predicted):
    days = pd.Series(actual.index.date).unique()
    actual_score = 0
    predicted_score = 0
    for day in days:
        actual_day = actual.loc[day.strftime('%Y-%m-%d')]
        predicted_day = predicted.loc[day.strftime('%Y-%m-%d')]
        actual_pattern = utils.discharge_pattern(6.0, actual_day)
        actual_score += utils.peak_score(actual_day, actual_pattern)
        predicted_pattern = utils.discharge_pattern(6.0, predicted_day)
        predicted_score += utils.peak_score(actual_day, predicted_pattern)
    print('Peak Reduction Score {}'.format( (actual_score - predicted_score) / actual_score) )
    plot=False
    if plot:
        k=range(len(actual))
        plt.plot(k,actual, label='actual demand')
        plt.plot(k,predicted,label='predicted demand')
        plt.plot(k,actual_pattern,label='actual pattern')
        plt.plot(k,predicted_pattern,label='predicted pattern')
        plt.title('demand prediction : '+method)
        plt.xlabel('Hour of the year', fontsize=15)
        plt.ylabel('Demand (MW)', fontsize=15)
        plt.legend(loc='lower left', fontsize=15)
        plt.show()

# pytorch loss function for 1-R ( Pearson correlation )
def loss_1mr(X,Y):
    vx = X - torch.mean(X)
    vy = Y - torch.mean(Y)
    lossn = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return 1.0 - lossn

# loss function for difference
def loss_diff(X,Y):
#   print(X)
#   print(Y)
    diff =  X - Y
#   print(diff)
#   dmax =  torch.max(diff,1)
#   print(dmax)
#   dmin =  torch.min(diff,1)
#   print(dmin)
#   lossn =  dmax - dmin
    lossn = torch.var(diff,1).sum()
#   print(lossn)
#   quit()
    return lossn

# loss function for weighted L1
def loss_wl1(X,Y,W):
    lossn =  torch.mean(torch.abs(X - Y) * W)
    return lossn

# custom loss function - weighted to passed in weight
def loss_max1(X,Y):
#   lossn = abs( X.max().item() - Y.max().item() )
#   print(X.max().item(), Y.max().item(), lossn)
    lossn =  torch.abs(X.max() - Y.max())
    return lossn

# custom loss function - weighted to peaks
def loss_maxw(X,Y):
    # weight in favour of the peaks
    W = Y / Y.max()
    lossn =  torch.mean(torch.abs(X - Y) * W)
    return lossn

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

# class for LSTM which is a RNN algorithm
class LSTM(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=100):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

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

# Define a utility function to train the model
def fitlstm(num_epochs, model, loss_fn, opt, train_dl):
    loss_history=[]
    for epoch in range(num_epochs):
        # each batch in the training ds
        for xb,yb in train_dl:
            opt.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
        # report at each epoc
        sys.stdout.write('\repoch {:03d}, loss {:.7f} '.format(epoch, loss.item()))
        loss_history.append(loss.item() )
    sys.stdout.flush()
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

# svr
def forecast_svr(df, forecast, day):
    input_columns = ['tempm', 'dsk']
    input_df = df[input_columns].copy()
    # set up output
    output_column = 'demand'
    output = df[output_column].copy()

    x = input_df.values
#   y = output.values.reshape(-1, 1)
    y = output.values.reshape(len(output), 1)
#   y = output.values.reshape(len(output), )
    # normalise
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    regressor = SVR(kernel = 'rbf')
    regressor.fit(x, y)

    # Step 5: Predict
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    x_f = forecast_day[input_columns].copy().values
    print(x_f)
    x_pred = sc_x.transform(x_f)
    y_pred = regressor.predict(x_pred)
    y_pred = sc_y.inverse_transform(y_pred) 
    print(y_pred)
    print(len(y_pred))

    print(forecast_day)
    forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = y_pred

# using SVR to predict the demand for one k period

def period_svr(input_df, output, input_f):

    x = input_df.values
    y = output.values.reshape(len(output), 1)
    # normalise
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    regressor = SVR(kernel = 'rbf')
    regressor.fit(x, y)

    # Step 5: Predict
    x_f = input_f
    x_pred = sc_x.transform(x_f)
    y_pred = regressor.predict(x_pred)
    y_pred = sc_y.inverse_transform(y_pred) 

    return y_pred

# multi output SVR

def multi_svr(input_df, output, input_f):

    x = input_df.values
    y = output.values.reshape(len(output), len(output.columns))
    # normalise
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(x)
    y = sc_y.fit_transform(y)

    svr = SVR(kernel = 'rbf')
    regressor = MultiOutputRegressor(svr)
    regressor.fit(x, y)

    # Step 5: Predict
    x_f = input_f
    x_pred = sc_x.transform(x_f)
    y_pred = regressor.predict(x_pred)
    y_pred = sc_y.inverse_transform(y_pred) 

    return y_pred

def reg_inputs(df_nw1, df_nwn, alg):
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
#   base_columns = ['season', 'week', 'month']
    base_columns = ['season', 'week', 'month', 'dailytemp', 'tempyd', 'sh']
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
def forecast_new(df, forecast, day, seed, num_epochs, alg):

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
    if alg == 'ann':
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
def forecast_nreg(df, forecast, day, seed, num_epochs, alg):
    # only use days of the same type
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    dfd = df[df['dtype'] == forecast_day['dtype'].iloc[0]]
    # df_nw1 is the data we use to forecast from so we
    # drop the first week from it so we never try and forecast that so
    # we always have data from the previous week.
    df_nw1 = dfd.drop(dfd.head(48*7).index)
    # df_nw1 is where we get last weeks data from so we 
    # drop the last week from it so its the same length as df_nw1
    df_nwn = dfd.drop(dfd.tail(48*7).index)
    input_df = reg_inputs(df_nw1, df_nwn, alg)

    # outputs - demand for the k periods of interest
    k_demand={}
    for k in range(32,44):
        k_df = df_nw1[ df_nw1['k'] == k]
        k_demand['demand{}'.format(k)] = k_df['demand'].values
    output_df = pd.DataFrame(k_demand)

    # set up the values to forecast
    previous_day = utils.get_previous_week_day(dfd, day)
    print(previous_day)
    input_f = reg_inputs(forecast_day, previous_day, alg)
#   print(input_f)

    if alg == 'svr':
        prediction_values = multi_svr(input_df, output_df, input_f)
        vals = prediction_values[0]
        losses = []
    else:

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
        if alg == 'ann':
#           model = SimpleNet(num_inputs, num_outputs, 800)
            model = SimpleNet(num_inputs, num_outputs, 1400)
#           loss_fn = F.l1_loss
            loss_fn = loss_maxw
#           loss_fn = loss_1mr
#           loss_fn = loss_diff
#           batch_size = 1
        else:
            loss_fn = F.l1_loss
            model = nn.Linear(num_inputs, num_outputs)

#       opt = torch.optim.SGD(model.parameters(), lr=1e-5)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)

        # Define loss function
#       loss_fn = F.mse_loss
#       loss_fn = loss_max1

        loss = loss_fn(model(inputs), targets)
#   print(loss)
        # Train the model
        losses = fit(num_epochs, model, loss_fn, opt, train_dl)
        print('Training loss: ', loss_fn(model(inputs), targets))
        preds = model(inputs)
#       print(preds)

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
    weight = 1.0 - weight
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
    print('Public Holiday algorithm: Gradient {} intercept {}'.format(res_grad, res_const) )
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
#        regs  - model per season, dtype as a flag

def forecast_reg(df, forecast, day, method, plot, seed, num_epochs, period, ka, alg, wl, ploss, slope):
    pred_values=[]
    forecast_day = forecast.loc[day.strftime('%Y-%m-%d')].copy()
    # look for days of similar type.
    fd_type = forecast_day['dtype'].iloc[0]
    if method == 'regd':
        dfd = df[df['dtype'] == fd_type]
    else:
        # look for days in the same season
        if method == 'regs':
            s_type = forecast_day['season'].iloc[0]
            dfd = df[df['season'] == s_type]
            # exclude public holidays as these are treated differently
            dfd = dfd[dfd['dtype'] < 7]
        else:
            dfd = df

    if wl:
        set_weight(dfd, forecast_day)
    else:
        weight = np.abs(dfd['week'].values - forecast_day['week'].iloc[0])
        weight = weight / np.max(weight)
        weight = 1.0 - weight
        dfd['weight'] = weight
        forecast_day['weight'] = 1.0

    # set doy_diff based on closeness to day of year.
    # closer days should have a smaller value. 
    # difference between day of year and that of the forecast day
    doy_diff = np.abs(dfd['doy'].values - forecast_day['doy'].iloc[0])
    # to cope with days at the end of one year being close to those at the
    # start of the next
    doy_diff = np.minimum( doy_diff, np.abs(doy_diff - 365) )
    dfd['doydiff'] = doy_diff
    forecast_day['doydiff'] = 0.0


    if period == all:
        # for each k period, train a seperate model ...
        for index, row in forecast_day.iterrows():
            dsk = row['dsk']
            if not ka and (row['k'] < 32 or row['k'] > 42):
                prediction_values = 1.0
            else:
                print('Period k {} dsk {}'.format(row['k'],dsk) )
                dsk_f = forecast_day[forecast_day['dsk'] == dsk]
                dsk_f1 = forecast_day[forecast_day['dsk'] == dsk -1]
                # if pub hol or christmas then we don't have enough so interp
                if fd_type >= 7:
                    dfd = df[df['dtype'] == fd_type]
                    dsk_df = dfd[dfd['dsk'] == dsk]
                    prediction_values = forecast_pub_hol(dsk_df, dsk_f, ploss)
                else:
                    dsk_df = dfd[dfd['dsk'] == dsk]
                    dsk_df1 = dfd[dfd['dsk'] == dsk - 1]
                    if slope:
                        dsk_df = utils.to_slopes(row['k'], dsk_df, dsk_df1)
                    prediction_values = forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ka, alg, wl, dsk_df1, dsk_f1, slope)
                    if slope:
                        dsk_df = utils.from_slopes(row['k'], dsk_df, dsk_df1)
                if alg != 'features':
                    if math.isnan(prediction_values):
                        print('WARNING NaN replaced by interpolation at period {}'.format(row['k']))
                    if not math.isfinite(prediction_values):
                        print('WARNING Inf replaced by interpolation at period {}'.format(row['k']))
                        prediction_values = float("NaN")
            pred_values.append(prediction_values)
        if alg == 'features':
            print('Features')
            count=0
            p_list={}
            for period in pred_values:
                count+=1
                if count>31 and count <43:
                    p_list[count] = period
            pdf = pd.DataFrame(p_list)
            print(pdf)
            plt.figure(figsize=(12,10))
            sns.heatmap(pdf, annot=True, cmap=plt.cm.Reds)
            plt.show()
            quit()
        else:
            pred_series = pd.Series(pred_values, index=forecast_day.index)
            # Replace any NaNs at the start or end with adjacent values
            pred_series = pred_series.fillna(method='bfill')
            pred_series = pred_series.fillna(method='ffill')
            # Replace any NaNs in the middle by interpolation.
            pred_series = pred_series.interpolate()
            if slope:
                utils.series_from_slopes(pred_series)
            forecast.loc[day.strftime('%Y-%m-%d'), 'prediction'] = pred_series
    else:
        dsk = int(period)
        dsk_f = forecast_day[forecast_day['dsk'] == dsk]
        dsk_f1 = forecast_day[forecast_day['dsk'] == dsk -1]
        if fd_type >= 7:
            dfd = df[df['dtype'] == fd_type]
            dsk_df = dfd[dfd['dsk'] == dsk]
            prediction_values = forecast_pub_hol(dsk_df, dsk_f, ploss)
        else:
            dsk_df = dfd[dfd['dsk'] == dsk]
            dsk_df1 = dfd[dfd['dsk'] == dsk - 1]
            prediction_values = forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ka, alg, wl, dsk_df1, dsk_f1, slope)
        print('Predicted value {} for period {}'.format(prediction_values, period))
        quit()

def forecast_reg_period(dsk_df, dsk_f, method, plot, seed, num_epochs, dsk, ka, alg, wl, dsk_df1, dsk_f1, slope):
    lagged = []
    # set up inputs
    if method == 'regd':
        # sfactor seems to make set 1 worse
        input_columns = ['tempm', 'sunm', 'season', 'zenith', 'month', 'weight', 'sh', 'dailytemp', 'lockdown', 'tempyd', 'temp1', 'temp2', 'temp3', 'temp4','temp5', 'temp6']
#       removing these makes regd svr and ann worse
#       if alg != 'ann':
        input_columns.extend(['tsqd', 'ts','tm'])
#       With the weighted loss function batch size has to be 1
        batch_size = 1
        rate = 1e-3
        nhidden = 120

    if method == 'regl':
#       input_columns = ['tempm', 'zenith']
#       input_columns = ['tempm', 'zenith', 'holiday']
#       input_columns = ['tempm', 'sun2', 'holiday', 'nothol']
        input_columns = ['tempm', 'zenith', 'holiday', 'nothol']
        batch_size = 1
        rate = 1e-4
        nhidden = 20

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
        nhidden = 20

    if method == 'regs':
        # no point having dtype in here, as we have flags for each day below
#       input_columns = ['tempm', 'sunm', 'ph', 'zenith', 'tsqd', 'ts', 'month', 'tm', 'weight', 'sh', 'dailytemp', 'lockdown', 'tempyd', 'doydiff']
        # tempdb is the day before yesterday and has no effect.
#       lasoo version set0
#       input_columns = ['tsqd', 'ts', 'tm', 'dailytemp', 'tempyd', 'doydiff', 'tempdb']
#       lasoo version set2
#       input_columns = ['tempm', 'sunm', 'ph', 'zenith', 'tsqd', 'ts', 'month', 'tm', 'weight', 'sh', 'dailytemp', 'lockdown', 'tempyd', 'doydiff', 'tempdb']
#       lasoo version set4
#       input_columns = ['sunm', 'zenith', 'tsqd', 'ts', 'tm', 'weight', 'sh', 'dailytemp', 'lockdown', 'doydiff', 'tempdb']
        if alg == 'features':
#           input_columns = ['tempm', 'sunm', 'ph', 'zenith', 'tsqd', 'ts', 'month', 'tm', 'weight', 'sh', 'dailytemp', 'lockdown', 'tempyd', 'doydiff', 'tempdb']
            input_columns = ['tempm', 'sunm', 'ph', 'zenith', 'tsqd', 'ts', 'month', 'tm', 'weight', 'sh', 'dailytemp', 'lockdown', 'tempyd', 'doydiff', 'tempdb', 'sun1', 'sun2', 'sun3', 'sun4', 'sun5', 'sun6', 'sunw', 'temp1', 'temp2', 'temp3', 'temp4', 'temp5', 'temp6', 'cs_ghi']
        else:
            input_columns = ['tempm', 'sunm', 'zenith', 'tsqd', 'ts', 'month', 'tm', 'weight', 'sh', 'dailytemp', 'lockdown', 'tempyd', 'doydiff', 'tempdb']
        # variables from the previous period - doesn't seem to make any
        # difference
        #lagged = ['tempm']
        lagged = []
        # days of the week 1-0 flags
        for wd in range(7):
            wd_key = 'wd{}'.format(wd)
            input_columns.append(wd_key)

        batch_size = 1
        rate = 1e-3
        nhidden = 50

    input_df = dsk_df[input_columns].copy()
    for lcol in lagged:
        input_df['l'+lcol] = dsk_df1[lcol].values

    # set up output
    output_column = 'demand'
    output = dsk_df[output_column].copy()

    if alg == 'features':
        features = select_features(input_df,output,plot)
        return features

    if alg == 'svr' or alg == 'gpr' or alg=='rf' or alg=='gbr':
        input_f = dsk_f[input_columns].copy()
        for lcol in lagged:
            input_f['l'+lcol] = dsk_f1[lcol].values
        if alg == 'svr':
            prediction_values = period_svr(input_df, output, input_f)
            return prediction_values[0]
        else:
            if alg == 'gpr':
                prediction_values = period_gpr(input_df, output, input_f, slope)
                return prediction_values
            else:
                if alg == 'rf':
                    prediction_values = period_rf(input_df, output, input_f)
                    return prediction_values[0]
                else:
                    prediction_values = period_gbr(input_df, output, input_f)
                    return prediction_values[0]
    else:

        # store maximum values
        input_max = utils.df_normalise(input_df)
        # santity check
        utils.sanity_check(input_df)
        # normalise the output (Y)
        output_max = output.max()
        if output_max>0:
            output = output / output_max

        # santity check
        if output.isna().sum() >0:
            print("ERROR NaN in output")
            quit()

        inputs = torch.tensor(input_df.values.astype(np.float32))
        targets = torch.tensor(output.values.astype(np.float32)).view(-1,1)
        torch.manual_seed(seed)    # reproducible

        input_f = dsk_f[input_columns].copy()
        for lcol in lagged:
            input_f['l'+lcol] = dsk_f1[lcol].values
        # normalise the inputs (using same max as for the model)
        for column in input_f.columns:
            if input_max[column]>0:
                input_f[column] = input_f[column] / input_max[column]
        f_inputs = torch.tensor(input_f.values.astype(np.float32))

        train_ds = TensorDataset(inputs, targets)

        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        next(iter(train_dl))

        num_inputs = len(input_df.columns)
        if alg == 'ann':
            model = SimpleNet(num_inputs,1,nhidden)
            loss_fn = F.l1_loss
        else:
            if alg == 'lstm':
                model = LSTM(num_inputs,1,nhidden)
                loss_fn = F.l1_loss
            else:
                model = nn.Linear(num_inputs,1)
                loss_fn = F.l1_loss

        # day types treated differently and weighted loss
        if wl:
            loss_fn = loss_wl1
            weights = torch.tensor(dsk_df['weight'].values.astype(np.float32)).view(-1,1)
            opt = torch.optim.SGD(model.parameters(), lr=rate)
#               opt = torch.optim.Adam(model.parameters(), lr=rate)
            loss = loss_fn(model(inputs), targets, weights)
            losses = wfit(num_epochs, model, loss_fn, opt, train_dl, weights)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=rate)
            if alg == 'lstm':
                losses = fitlstm(num_epochs, model, loss_fn, opt, train_dl)
            else:
                loss = loss_fn(model(inputs), targets)
                losses = fit(num_epochs, model, loss_fn, opt, train_dl)

#           preds = model(inputs)
        # prediction
        preds = model(f_inputs)
        # denormalize using df for the original model
        prediction_values = preds.detach().numpy() * output_max

        if args.ploss:
            plt.plot(losses)
            plt.title('Demand Regression convergence. period {}'.format(dsk))
            plt.xlabel('Epochs', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.show()

        if alg == 'lstm':
            return prediction_values[0]
        else:
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
parser.add_argument('--slope', action="store_true", dest="slope", help='Use slopes', default=False)
parser.add_argument('--alg', action="store", dest="alg", help='Algorithm: linear, ann, svm' , default='linear' )
parser.add_argument('--mname', action="store_true", dest="mname", help='Name the output file using the method', default=False)
parser.add_argument('--seed', action="store", dest="seed", help='Random seed, default=1.0', type=float, default=1.0)
parser.add_argument('--wl', action="store_true", dest="wl", help='Use Weight loss function', default=False)
parser.add_argument('--epochs', action="store", dest="epochs", help='Number of epochs', type=int, default=100)
parser.add_argument('--nnear', action="store", dest="nnear", help='Number of days when using the --day near option', type=int, default=10)
parser.add_argument('--period', action="store", dest="period", help='Period k to forecast', type=float, default=all)
parser.add_argument('--step', action="store", dest="step", help='If using days=all or step only do every step days', type=int, default=1)
parser.add_argument('--use', action="store", dest="use", help='Number of days data to use, 0=all', type=int, default=0)
parser.add_argument('--ka', action="store_true", dest="ka", help='Forecast all K (default is only k of interest)', default=False)
parser.add_argument('--window', action="store", dest="window", help='Rolling window of days data to use, 0=all', type=int, default=0)

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

# add extra variables
additional_values(forecast)

# process the day options
forecast, int_day = utils.set_forecast_days(args.day, forecast, df, args.nnear, 'demand' )

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

        # if forecasting a competition week, just copy the data.
        if args.day == 'set':
            history = df.copy()
        else:
            # drop this day and anything following from the history
            if args.window > 0:
                window_end = df.index.get_loc(day_start)
                window_start = window_end - args.window * 48
                print(window_end, window_start)
                if window_start>=0:
                    history = df.iloc[window_start:window_end].copy()
                    print(history)
                else:
                    print('ERROR: window start negative')
                    quit()
            # drop this day from main data
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
            forecast_reg(history, forecast, day, method, args.plot, args.seed, args.epochs, args.period, args.ka, args.alg, args.wl, args.ploss, args.slope)

        if method == 'svr':
            forecast_svr(df, forecast, day)
        if method == 'nreg':
            losses = forecast_nreg(history, forecast, day, args.seed, args.epochs, args.alg)
            if args.ploss:
                plt.plot(losses)
                plt.title('demand nreg convergence')
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
            losses, pred_peak, ac_peak = forecast_new(history, forecast, day, args.seed, args.epochs, args.alg)
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
    if not args.ka:
        kf = forecast[ (forecast['k'] > 31) & (forecast['k'] < 43)]
#       print(kf['demand'])
#       print(kf['prediction'])
        print('Metrics for k>31 and k<43')
        utils.print_metrics(kf['demand'], kf['prediction'])
        print_peak_metric(kf['demand'], kf['prediction'])
        print('Metrics for whole day')
        utils.print_metrics(forecast['demand'], forecast['prediction'], False)
    else:
        utils.print_metrics(forecast['demand'], forecast['prediction'], args.plot)
        print_peak_metric(forecast['demand'], forecast['prediction'])
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
    output_filename = '{}demand_forecast_{}_{}_{}.csv'.format(output_dir, method, args.alg, dataset)
forecast.to_csv(output_filename, float_format='%.2f')

# only demand for bogdan
forecast = forecast['prediction']
forecast = forecast.squeeze()
forecast = forecast.rename('demand_forecast')
forecast.index.rename('datetime', inplace=True)

output_filename = '{}demand_forecast_{}_only.csv'.format(output_dir, dataset)
forecast.to_csv(output_filename, float_format='%.2f')
