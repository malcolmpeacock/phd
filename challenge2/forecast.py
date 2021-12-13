# contrib code
import sys
import pandas as pd
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import sklearn.gaussian_process as gp
import lightgbm as lgb
import torch
import torch.nn as nn
# Import tensor dataset & data loader
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import xgboost as xgb
#from catboost import CatBoostRegressor

import time
import datetime

# refine the original forecast using an ANN
#
# Forecast week:
#  prediction  - the original forecast    [max_demand, min_demand]
#  df_forecast - the augmented demand and weather data
# Test data (pre-august ):
#  out_fits    - the values fit test data [max_demand, min_demand]
#  df_in       - the augmented demand and weather data
#  df_out      - the actual [max_demand, min_demand]
#
# Train an ANN to predict df_out from out_fits and df_in
# Then use it to do another prediction from prediction and df_in
def add_refine(prediction, out_fits, df_in, df_forecast, df_out):
    print('Refining original prediction with ANN')
#   cols = ['demand']
    cols = ['solar_irradiance1']
    in_data = pd.concat([out_fits, df_in[cols] ], axis=1)
    out_data = df_out[['max_demand', 'min_demand']]
    f_data = pd.concat([prediction, df_forecast[cols] ], axis=1)

    new_prediction = ann_forecast(in_data, out_data, f_data, args.plot, args.epochs)
    return new_prediction

def to_diffs(df_in, df_out):
    df_out['max_demand'] = df_out['max_demand'] - df_in['demand']
    df_out['min_demand'] = df_in['demand'] - df_out['min_demand']

def from_diffs(df_in, df_out):
    df_out['max_demand'] = df_in['demand'] + df_out['max_demand']
    df_out['min_demand'] = df_in['demand'] - df_out['min_demand']

# feature correlation
def correlation(input_df, output, plot=False):
#   print(input_df.index)
#   print(output.index)
    coef = {}
    for column in input_df.columns:
        coef[column] = output.corr(input_df[column])
    return(coef)

def add_parm(parameter):
    parms = []
    for p in range(5):
        parms.append(parameter + str(p+1))
    return parms

# forecast
#   df_in       input weather and demand
#   df_out      max and min demand for same period as df_in
#   df_forecast same variables as df_in but for the forecast period
def forecast(df_in, df_out, df_forecast, out_cols=['max_demand', 'min_demand']):
    print('forecast: df_in {} df_out {} df_forecast {}'.format(len(df_in), len(df_out), len(df_forecast) ) )
    # basic columns - no augmentations
    basic_cols = ['demand']
    basic_cols += add_parm('solar_irradiance')
    basic_cols += add_parm('windspeed_east')
    basic_cols += add_parm('windspeed_north')
    basic_cols += add_parm('spec_humidity')
    basic_cols += add_parm('temperature')
    basic_cols += add_parm('pressure')
    max_cols = ['demand', 'solar_irradiance1', 'windspeed_east1', 'k', 'windspeed1', 'spec_humidity1', 'solar_irradiance_var', 'dailytemp', 'spec_humidity1_lag1', 'demand_lag1', 'spec_humidity_var', 'solar_irradiance_var_lag1', 'temperature3', 'temperature1', 'solar_irradiance2', 'demand_lag2', 'demand_lag3' ,'demand_lag4', 'windspeed_var', 'windspeed3', 'windspeed_north3', 'windspeed_north1', 'solar_irradiance1_lag1', 'temperature_var']
#   lass_max_cols = ['demand', 'solar_irradiance_var', 'solar_irradiance1', 'windspeed_east1', 'cloud', 'windspeed1', 'k', 'dailyhume', 'solar_irradiance_var_lag1', 'solar_irradiance1_lag1', 'windspeed1_diff', 'windspeed3', 'demand_lag1', 'solar_irradiance2']
    var_cols = ['spec_humidity1'
               ,'solar_irradiance1'
               ,'windspeed_east1'
               ,'solar_irradiance_var'
               ,'cloud'
               ,'windspeed1'
               ,'temperature1'
               ,'dailyhume'
               ,'dailytemp'
               ,'solar_irradiance1_diff'
               ,'windspeed_east3'
#              ,'windspeed_var_lag1'
#              ,'zenith'
#              ,'windspeed_east3_lag1'
               ,'solar_irradiance_var_lag1'
#              ,'solar_irradiance1_lag1'
               ,'windspeed3'
               ]
#   max demand - lasso
    lass_max_cols = ['demand'
                    ,'spec_humidity1'
                    ,'solar_irradiance1'
                    ,'windspeed_east1'
                    ,'solar_irradiance_var'
                    ,'demand_diff'
                    ,'windspeed1'
                    ,'cloud'
                    ,'k'
                    ,'temperature1'
                    ,'solar_irradiance1_diff'
                    ,'dailytemp'
                    ,'dailyhume'
#                   ,'zenith'
                    ,'windspeed3'
#                   ,'windspeed1_diff'
#                   ,'windspeed_var_lag1'
#                   ,'windspeed_var'
#                   ,'windspeed_east3_lag1'
                    ,'windspeed_east3'
#                   ,'temperature3'
#                   ,'temperature_var'
#                   ,'spec_humidity_var'
#                   ,'solar_irradiance2'
#                   ,'solar_irradiance1_lag1'
                    ,'solar_irradiance_var_lag1'
#                   ,'month'
                    ,'hdh'
#                   ,'demand_lag1'
#                   ,'cs_ghi'
# more from lgbm?
                    ,'doy'
#                   ,'presyd'
#                   ,'demand_lag4'
#                   ,'presdb'
#                   ,'wd'
#                   ,'spec_humidity_var_lag1'
#                   ,'windspeed_east5'
#                   ,'solar_irradiance5_lag1'
#                   ,'solar_irradiance4_lag1'
#                   ,'solar_irradiance_var_lag2'
                    ]
#   min demand - lasso
    lass_min_cols = ['demand'
                   , 'solar_irradiance_var'
                   , 'solar_irradiance1'
                   , 'demand_diff'
                   , 'cloud'
                   , 'spec_humidity1'
                   , 'windspeed_east1'
                   , 'windspeed1'
                   , 'k'
                   , 'dailytemp'
                   , 'dailyhume'
                   , 'windspeed_var'   # new
                   , 'temperature3'    # new
#                  , 'temperature_var'
#                  , 'solar_irradiance2'
#                  , 'temperature1'
#                   ,'zenith'
                    ,'windspeed3'    # new
#                   ,'windspeed1_diff'
#                   ,'windspeed_var_lag1'
                    ,'windspeed_east3_lag1'
#                   ,'windspeed_east3'
#                   ,'spec_humidity_var'
#                   ,'solar_irradiance_var_lag1'
                    ,'hdh'    # new
#                   ,'demand_lag1'
                    ,'cs_ghi'    # new
#                   ,'windspeed1_cube'
#                   ,'windspeed_north3'
                    ,'windspeed_north2_lag1'    # new
                    ,'windspeed_north1'    # new
#                   ,'trend'
#                  , 'temperature5_lag1'
#                  , 'temperature5'
#                  , 'temperature4'
                   , 'temperature3_lag1'    # new
                   , 'temperature2_lag1'    # new
                   , 'temperature2'    # new
#                  , 'temperature1_lag1'
#                  , 'spec_humidity5_lag1'
#                  , 'spec_humidity5'
#                  , 'spec_humidity4_lag1'
#                  , 'spec_humidity4'
                   , 'spec_humidity3'    # new
#                   ,'solar_irradiance3'
#                   ,'solar_irradiance2_lag1'
#                   ,'season'
#                   ,'humedb'
#                   ,'demand_lag4'
#                   ,'demand_lag3'
#                   ,'demand_lag2'
# more from lgbm?
#                   ,'doy'
#                   ,'presyd'
#                   ,'presdb'
#                   ,'wd'
#                   ,'spec_humidity_var_lag1'
#                   ,'windspeed_east5'
#                   ,'solar_irradiance5_lag1'
#                   ,'solar_irradiance4_lag1'
                   ]
    lgb_max_cols = ['demand', 'demand_diff', 'demand_lag1', 'cs_ghi', 'demand_lag2', 'trend', 'zenith', 'cloud', 'solar_irradiance1_diff', 'presyd', 'demand_lag4', 'presdb', 'k', 'solar_irradiance_var_lag1', 'spec_humidity_var_lag1', 'windspeed_var_lag1', 'windspeed_var', 'wd', 'spec_humidity_var']
    lgb_min_cols = ['demand', 'solar_irradiance1', 'solar_irradiance_var', 'windspeed_east1', 'windspeed1', 'cloud', 'k', 'dailyhume', 'solar_irradiance_var_lag1', 'solar_irradiance2', 'windspeed1_diff', 'demand_lag1']
    lass_cols_maxd = ['temperature1'
                    , 'temperature1_lag1'
                    , 'demand'
                    , 'spec_humidity2'
                    , 'spec_humidity1_lag1'
                    , 'humeyd'
                    , 'spec_humidity4'
                    , 'temperature5'
                    , 'windspeed_east1'
                    , 'demand_diff'
                    , 'tempyd'
                    , 'solar_irradiance5'
                    , 'humedb'
                    , 'temperature3_lag1'
                    , 'windspeed_east3'
                    , 'dailyhume'
                    , 'windspeed_var_lag1'
                    , 'windspeed1_cube'
                    , 'windspeed_east2'
                    , 'zenith'
                    , 'windspeed_east3_lag1'
                    , 'tempdb'
                    , 'dailytemp'
#                   , 'solar_irradiance_var_lag1'
#                   , 'windspeed1_diff'
#                   , 'cloud'
#                   , 'demand_lag2'
#                   , 'dsk'
#                   , 'windspeed_north1_lag1'
#                   , 'season'
#                   , 'windspeed_north3_lag1'
                    ]
    lass_cols_mind = ['spec_humidity2'
                    , 'temperature5'
                    , 'spec_humidity1_lag1'
                    , 'windspeed_east3_lag1'
                    , 'temperature1'
                    , 'windspeed_east1'
                    , 'cloud'
                    , 'solar_irradiance_var'
                    , 'windspeed_var_lag1'
                    , 'windspeed1_cube'
                    , 'solar_irradiance1'
                    , 'spec_humidity3_lag1'
                    , 'temperature1_lag1'
                    , 'demand_diff'
                    , 'solar_irradiance5'
                    , 'hdh'
                    , 'humeyd'
                    , 'humedb'
                    , 'windspeed_east3'
                    , 'windspeed_east2'
                    , 'solar_irradiance3'
                    , 'solar_irradiance2'
                    , 'windspeed_east4'
                    , 'windspeed1_diff'
                    , 'windspeed3'
#                   , 'solar_irradiance3_lag1'
#                   , 'windspeed3'
#                   , 'solar_irradiance3_lag1'
#                   , 'windspeed_east4_lag1'
#                   , 'dailyhume'
#                   , 'solar_irradiance_var_lag1'
#                   , 'solar_irradiance1_lag1'
#                   , 'windspeed1_power'
#                   , 'season'
#                   , 'k'
#                   , 'trend'
#                   , 'windspeed_east5'
#                   , 'windspeed_north2_lag1'
#                   , 'windspeed_north2_lag1'
#                   , 'demand'
                    ]
    my_cols_maxd = [ 'solar_irradiance_var'
                   , 'solar_irradiance_var_lag1'
                   , 'solar_irradiance1'
                   , 'cloud'
                   , 'spec_humidity1'
                   , 'spec_humidity1_lag1'
                   , 'humeyd'
                   , 'windspeed_var'
                   , 'windspeed_east1'
                   , 'windspeed_east1_lag1'
                   , 'windspeed_north1'
                   , 'windspeed_north1_lag1'
                   , 'temperature1'
                   , 'demand_diff'
                   , 'temperature1'
                   , 'demand'
#                  , 'pressure1'
#                  , 'spec_humidity1_lag1'
#                   , 'tempyd'
#                   , 'solar_irradiance5'
#                   , 'humedb'
#                  , 'dailyhume'
#                   , 'temperature3_lag1'
#                   , 'windspeed_east3'
#                   , 'windspeed_var_lag1'
#                   , 'zenith'
#                   , 'windspeed1_cube'
#                   , 'windspeed_east2'
#                   , 'windspeed_east3_lag1'
#                   , 'tempdb'
#                   , 'dailytemp'
#                   , 'windspeed1_diff'
#                   , 'cloud'
#                   , 'demand_lag2'
#                   , 'dsk'
#                   , 'season'
#                   , 'windspeed_north3_lag1'
                    ]
    my_cols_mind = [ 'solar_irradiance_var'
                   , 'solar_irradiance_var_lag1'
                   , 'solar_irradiance1'
                   , 'cloud'
                   , 'spec_humidity1'
                   , 'spec_humidity1_lag1'
                   , 'humeyd'
#                  , 'windspeed_var'
                   , 'windspeed_east1'
                   , 'windspeed_east1_lag1'
                   , 'windspeed_north1'
                   , 'windspeed_north1_lag1'
                   , 'temperature1'
                   , 'demand_diff'
                   , 'temperature1'
                   , 'demand'
                   , 'pressure1'
                    ]
    my_cols_max = [ 'solar_irradiance_var'
                   , 'solar_irradiance_var_lag1'
                   , 'solar_irradiance1'
                   , 'cloud'
                   , 'spec_humidity1'
                   , 'spec_humidity1_lag1'
                   , 'humeyd'
                   , 'windspeed_var'
                   , 'windspeed_east1'
                   , 'windspeed_east1_lag1'
                   , 'windspeed_north1'
                   , 'windspeed_north1_lag1'
                   , 'temperature1'
                   , 'demand_diff'
                   , 'temperature1'
                   , 'demand'
#                  , 'pressure1'
#                  , 'spec_humidity1_lag1'
#                   , 'tempyd'
#                   , 'solar_irradiance5'
                    ]
    my_cols_min = [ 'solar_irradiance_var'
                   , 'solar_irradiance_var_lag1'
                   , 'solar_irradiance1'
                   , 'cloud'
                   , 'spec_humidity1'
                   , 'spec_humidity1_lag1'
                   , 'humeyd'
#                  , 'windspeed_var'
                   , 'windspeed_east1'
                   , 'windspeed_east1_lag1'
                   , 'windspeed_north1'
                   , 'windspeed_north1_lag1'
                   , 'temperature1'
                   , 'demand_diff'
                   , 'temperature1'
                   , 'demand'
                   , 'pressure1'
                    ]
    if args.diffs:
        max_cols = ['demand', 'demand_diff', 'windspeed_east1','windspeed_east3_lag1', 'spec_humidity2', 'spec_humidity1_lag1', 'spec_humidity4', 'humeyd', 'solar_irradiance_var','temperature1_lag1', 'temperature5', 'temperature1', 'zenith', 'hdh', 'windspeed_var_lag1', 'windspeed_east2', 'windspeed1_cube','windspeed_east3', 'cloud', 'solar_irradiance_var_lag1', 'temperature5','solar_irradiance5']
    min_cols = max_cols
    if args.cols == 'all':
        max_cols = df_in.columns
        min_cols = max_cols
    if args.cols == 'basic':
        max_cols = basic_cols
        min_cols = max_cols
    if args.cols == 'lass':
        max_cols = lass_max_cols
        min_cols = lass_min_cols
    if args.cols == 'lassd':
        max_cols = lass_cols_maxd
        min_cols = lass_cols_mind
    if args.cols == 'myd':
        max_cols = my_cols_maxd
        min_cols = my_cols_mind
    if args.cols == 'my':
        max_cols = my_cols_max
        min_cols = my_cols_min

    input_cols = { 'max_demand' : max_cols, 'min_demand' : min_cols }

    if args.method=='ann':
#       ann_cols = ['demand', 'spec_humidity1', 'dailytemp']
#       ann_cols = ['demand', 'solar_irradiance1', 'windspeed_east1', 'k', 'windspeed1', 'windspeed3', 'solar_irradiance2', 'spec_humidity1_lag1', 'solar_irradiance1_lag1']
        ann_cols = max_cols
        prediction = ann_forecast(df_in[ann_cols], df_out[['max_demand', 'min_demand']], df_forecast[ann_cols], args.plot, args.epochs)
    else:
        if args.method=='variance':
            diffs = df_out['max_demand'] - df_out['min_demand']
            variance, fits = lgbm_forecast(var_cols, df_in, df_forecast, diffs)
            max_demand = df_forecast['demand'] + (variance * 0.5)
            min_demand = df_forecast['demand'] - (variance * 0.5)
            forecasts = {'max_demand' : max_demand, 'min_demand' : min_demand }
            prediction = pd.DataFrame(forecasts, index=df_forecast.index)
        else:
            forecasts = {}
            out_fits = {}
            for out_col in out_cols:
                print('Method {} Output {} ...'.format(args.method, out_col) )
        
                if args.method=='rf':
                    forecasts[out_col] = rf_forecast(max_cols, df_in, df_forecast, df_out[out_col])
                if args.method=='gpr':
                    forecasts[out_col] = gpr_forecast(max_cols, df_in, df_forecast, df_out[out_col])
                if args.method=='lgbm':
                    forecasts[out_col], out_fits[out_col] = lgbm_forecast(input_cols[out_col], df_in, df_forecast, df_out[out_col])
                if args.method=='cb':
                    forecasts[out_col] = cb_forecast(max_cols, df_in, df_forecast, df_out['max_demand'])
                if args.method=='xgb':
                    forecasts[out_col] = xgb_forecast(max_cols, df_in, df_forecast, df_out[out_col])

            prediction = pd.DataFrame(forecasts, index=df_forecast.index)
            if args.refine:
                pred_fit = pd.DataFrame(out_fits, index=df_in.index)
                prediction = add_refine(prediction, pred_fit, df_in, df_forecast, df_out)
    return prediction

def rf_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    # normalise the inputs 
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_x.fit_transform(X_train.values.astype(np.float32))
    y_train = sc_y.fit_transform(y_train.values.astype(np.float32).reshape(-1, 1))
    # normalise the inputs (using same max as for the model)
    X_test = sc_x.transform(X_test.values.astype(np.float32))
    # default error is RMSE criterion=“squared_error”
    regressor = RandomForestRegressor(n_estimators=300, random_state=0)
#   regressor = RandomForestRegressor(n_estimators=130, random_state=0)
    regressor.fit(X_train, y_train.ravel())
#   regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
#   print('RF ypred')
#   print(type(y_pred))
#   print(np.shape(y_pred))
#   print(y_pred)
    y_pred = sc_y.inverse_transform(y_pred)
    return y_pred

# lgbm

def lgbm_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    # normalise the inputs 
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_x.fit_transform(X_train.values.astype(np.float32))
    y_train = sc_y.fit_transform(y_train.values.astype(np.float32).reshape(-1, 1))
    # normalise the inputs (using same max as for the model)
    X_test = sc_x.transform(X_test.values.astype(np.float32))
    print('Creating Regressor ...')
    # defaults
    # num_leaves=31 learning_rate=0.1, n_estimators=100, boosting='gbdt'
    # (n_estimators is number of iterations)
    # dart not good
#   model = lgb.LGBMRegressor(num_leaves=41, learning_rate=0.05, n_estimators=200, boosting_type='dart', deterministic=True)
    # TODO - try max_bin= 255 is the default . 511 makes it worse
    # 2550 makes it very slightly better.
    # try 2 settings: num_leaves=45, learning_rate=0.03, n_estimators=300, max_bin=2550 )
    model = lgb.LGBMRegressor(num_leaves=45, learning_rate=0.03, n_estimators=500, boosting_type='gbdt', deterministic=True, max_bin=5000 )
    print('Fitting model ...')
    model.fit(X_train, y_train.ravel(), eval_metric='l2')
    y_pred = model.predict(X_test)
#   print('RF ypred')
#   print(type(y_pred))
#   print(np.shape(y_pred))
#   print(y_pred)
    y_pred = sc_y.inverse_transform(y_pred)
    
    if args.refine:
        f_pred = model.predict(X_train)
        f_pred = sc_y.inverse_transform(f_pred)
    else:
        f_pred = None
    return y_pred, f_pred
  
# xgBoost

def xgb_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    # normalise the inputs 
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_x.fit_transform(X_train.values.astype(np.float32))
    y_train = sc_y.fit_transform(y_train.values.astype(np.float32).reshape(-1, 1))
    # normalise the inputs (using same max as for the model)
    X_test = sc_x.transform(X_test.values.astype(np.float32))
    print('Creating Regressor ...')
    #  lowering learing rate and increasing estimators seem to make it
    #  worse.
    model = xgb.XGBRegressor(max_depth=7, n_estimators=300, n_jobs=1,
                           objective='reg:squarederror', booster='gbtree',
                           random_state=42, learning_rate=0.05)
    print('Fitting model ...')
    model.fit(X_train, y_train.ravel() )
    y_pred = model.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    return y_pred

# catBoost

def cb_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    # normalise the inputs 
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_x.fit_transform(X_train.values.astype(np.float32))
    y_train = sc_y.fit_transform(y_train.values.astype(np.float32).reshape(-1, 1))
    # normalise the inputs (using same max as for the model)
    X_test = sc_x.transform(X_test.values.astype(np.float32))
    print('Creating catboost Regressor ...')
    #  lowering learing rate and increasing estimators seem to make it
    #  worse.
    model = CatBoostRegressor(verbose=0, n_estimators=100)
    print('Fitting model ...')
    model.fit(X_train, y_train.ravel() )
    y_pred = model.predict(X_test)
    y_pred = sc_y.inverse_transform(y_pred)
    return y_pred



# gpr

def gpr_forecast(columns, df_in, df_forecast, df_out):
    X_train = df_in[columns]
    y_train = df_out
    X_test = df_forecast[columns]
    # normalise the inputs 
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_x.fit_transform(X_train.values.astype(np.float32))
    y_train = sc_y.fit_transform(y_train.values.astype(np.float32).reshape(-1, 1))
    # normalise the inputs (using same max as for the model)
    X_test = sc_x.transform(X_test.values.astype(np.float32))
    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e8))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=False, random_state=0)

    model.fit(X_train, y_train)
    y_pred, std = model.predict(X_test, return_std=True)
    y_pred = y_pred.reshape(len(X_test),)
    y_pred = sc_y.inverse_transform(y_pred)
    return y_pred

def naive(df_forecast):
    print('naive: df_forecast {}'.format(len(df_forecast) ) )
#   print(df_forecast)
    max_demand_forecast = df_forecast['demand'].copy()
    min_demand_forecast = df_forecast['demand'].copy()
    data = { 'max_demand' :  max_demand_forecast, 'min_demand': min_demand_forecast }
    prediction = pd.DataFrame(data, index=df_forecast.index)
    return prediction

def mape(a,f,parm):
    mape = ( (f[parm] - a[parm]) / a[parm]).abs().mean()
    return mape

def assess(df_forecast, df_actual):
    print('assess: df_forecast {} df_actual {}'.format(len(df_forecast), len(df_actual) ) )
    if len(df_forecast) != len(df_actual):
        print('ERROR forecast and actual different lengths')
        quit()
    max_diff2 = (df_forecast['max_demand'] - df_actual['max_demand']).pow(2)
    min_diff2 = (df_forecast['min_demand'] - df_actual['min_demand']).pow(2)
    rmse = math.sqrt(max_diff2.sum() + min_diff2.sum() )

    min_mape = mape(df_forecast, df_actual, 'min_demand' )
    max_mape = mape(df_forecast, df_actual, 'max_demand' )

    if args.model == 'minute' and 'std' in df_forecast.columns:
        std_mape = mape(df_forecast, df_actual, 'std' )
        var_mape = mape(df_forecast, df_actual, 'var' )
        sem_mape = mape(df_forecast, df_actual, 'sem' )
    else:
        std_mape = 0.0
        var_mape = 0.0
        sem_mape = 0.0
    return rmse, min_mape, max_mape, std_mape, var_mape, sem_mape

def check_max_min_diff(df):
    diff = df['max_demand'] - df['min_demand']
    return diff.mean()

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
#       self.act2 = nn.LeakyReLU()
        self.act2 = nn.Sigmoid()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x

# class to create ANN

class TwoLayers(nn.Module):
    # Initialize the layers
    def __init__(self,num_inputs,num_outputs,num_hidden):
        super().__init__()
#       Layer 1
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        # Activation function
        self.act1 = nn.Sigmoid()
#       Layer 2
#       self.act2 = nn.LeakyReLU()
        self.act2 = nn.Sigmoid()
        self.linear2 = nn.Linear(num_hidden, num_hidden)
#       Layer 3
        self.act3 = nn.Sigmoid()
        self.linear3 = nn.Linear(num_hidden, num_outputs)

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)
        return x

# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    loss_history=[]
    for epoch in range(num_epochs):
        # each batch in the training ds
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
#           loss = torch.sqrt(loss_fn(pred, yb))
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


# ann_forecast
#   df_in       input weather and demand
#   df_out      max and min demand for same period as df_in
#   df_forecast same variables as df_in but for the forecast period
def ann_forecast(df_in, df_out, df_forecast, plot=False, num_epochs=2):
    # settings:
    seed = 1
    batch_size = 48
    num_neurons = args.nodes
    learning_rate = math.pow(10, -1 * args.rate)
    print('ann_forecast batch {} nodes {} rate {} optimizer {}'.format(batch_size, num_neurons, learning_rate, args.opt) )

    # normalise:
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x = sc_x.fit_transform(df_in.values.astype(np.float32))
    y = sc_y.fit_transform(df_out.values.astype(np.float32))

    inputs = torch.tensor(x)
    targets = torch.tensor(y)
    torch.manual_seed(seed)    # reproducible
    train_ds = TensorDataset(inputs, targets)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    next(iter(train_dl))
    num_inputs = len(df_in.columns)
    num_outputs = len(df_out.columns)
    loss_fn = F.mse_loss
    if args.model=='two':
        model = TwoLayers(num_inputs, num_outputs, num_neurons)
    else:
        model = SimpleNet(num_inputs, num_outputs, num_neurons)
    # adam optimiser
    if args.opt=='adam':
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
    # Stochiastic gradient descent
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss = loss_fn(model(inputs), targets)
    # Train the model
    losses = fit(num_epochs, model, loss_fn, opt, train_dl)
    print('Training loss: ', loss_fn(model(inputs), targets))
    preds = model(inputs)
    # normalise the inputs (using same max as for the model)
    x_f = sc_x.transform(df_forecast.values.astype(np.float32))
    f_inputs = torch.tensor(x_f)
#   print('f_inputs')
#   print(f_inputs)
    preds = model(f_inputs)
    vals = preds.detach().numpy()
    vals = sc_y.inverse_transform(vals)
    if plot:
        plt.plot(losses)
        plt.title('demand ann convergence')
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.show()

    data = { 'max_demand' :  vals[:,0], 'min_demand': vals[:,1] }
    prediction = pd.DataFrame(data, index=df_forecast.index)
    return prediction

# main program

methods = ['df', 'lgbm', 'xgb', 'ann', 'cb', 'gpr']

# process command line

parser = argparse.ArgumentParser(description='Create demand forecast.')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--diffs', action="store_true", dest="diffs", help='Predict the difference between max demand and demand', default=False)
parser.add_argument('--cols', action="store", dest="cols", help='Column set to use in the prediction', default='standard')
parser.add_argument('--errors', action="store_true", dest="errors", help='Show Error diagnostics', default=False)
parser.add_argument('--naive', action="store_true", dest="naive", help='Output the naive forecast', default=False)
parser.add_argument('--start', action="store", dest="start", help='Where to start rolling assesment from: 0=just forecast, 1=30 days before the end, 2=31 etc.' , default=0, type=int )
parser.add_argument('--data', action="store", dest="data", help='Where to start data from: 0=use all data, n= use the n most recent days.' , default=0, type=int )
parser.add_argument('--method', action="store", dest="method", help='Forecast method to use: '+', '.join(methods) , default='rf' )
parser.add_argument('--step', action="store", dest="step", help='Rolling assesment step.' , default=1, type=int )
parser.add_argument('--epochs', action="store", dest="epochs", help='Number of epochs to train ann' , default=1, type=int )
parser.add_argument('--nodes', action="store", dest="nodes", help='Number of neurons in hidden layer for ann' , default=5000, type=int )
parser.add_argument('--rate', action="store", dest="rate", help='Learning rate' , default=4, type=int )
parser.add_argument('--model', action="store", dest="model", help='ANN Model' , default='simple')
parser.add_argument('--opt', action="store", dest="opt", help='ANN Optimizer' , default='SGD')
parser.add_argument('--refine', action="store_true", dest="refine", help='Run an ANN aftwerwards to improve things', default=False)
args = parser.parse_args()

# read in the data
output_dir = "/home/malcolm/uclan/challenge2/output/"
# merged data file ( demand, weather, augmented variables )
merged_filename = '{}merged_pre_august.csv'.format(output_dir)
df_in = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
# print(df_in)

out_cols=['max_demand', 'min_demand']
if args.model == 'minute':
    minute_filename = '{}minutely.csv'.format(output_dir)
    df_out = pd.read_csv(minute_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
else:
# maxmin data file ( min/max in the period - what we are trying to predict )
    merged_filename = '{}maxmin_pre_august.csv'.format(output_dir)
    df_out = pd.read_csv(merged_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
if args.diffs:
    to_diffs(df_in, df_out)

if args.model == 'diffs':
    df_out['diff'] = df_out['max_demand'] - df_out['min_demand']
    out_cols = out_cols + ['diff']
if args.model == 'minute':
    out_cols = out_cols + ['std', 'var', 'sem']
# print(df_out)

print(time.strftime("Starting at: %Y-%m-%d %H:%M:%S", time.gmtime()) )
startTime = time.time()

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
        df_forecast = forecast(df_in, df_out, df_f_in, out_cols)
    if args.diffs:
        # set values in df_forecast back to actual max and min
        from_diffs(df_f_in, df_forecast)
        # set values in df_out back to actual max and min
        from_diffs(df_in, df_out)
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
    df_forecast.to_csv(output_filename, float_format='%.12f')

else:
    rmses=[]
    # 30 days and 48 half hour periods
    forecast_days = 30
    data_start = 0
    if args.data >0:
        data_start = len(df_in) - (args.data * 48)
    # for each window ...
    for window in range(0, args.start, args.step):
        # create a forecast df and shorten the input df
        win_start = (window * 48) + data_start
        win_end  = len(df_in) - (forecast_days + args.start - window)*48
        print('Window {} of {} start {} end {}'.format(window+1, math.floor(args.start / args.step), win_start, win_end ) )
        # training data ( weather and demand for prior period )
        df_train_in = df_in[win_start:win_end]
        # training data ( max/min that we are tyring to predict for same period)
        df_train_out = df_out[win_start:win_end]
        # training data ( weather and demand for forecast period)
        df_f_in  = df_in[win_end:win_end+forecast_days*48]
        # training data ( max/min that we are tyring to predict for forecast period)
        df_f_out  = df_out[win_end:win_end+forecast_days*48].copy()
        # forecast it
        df_forecast = forecast(df_train_in, df_train_out, df_f_in, out_cols)
        # forecasting diffs
        if args.diffs:
            # set values in df_forecast back to actual max and min
            from_diffs(df_f_in, df_forecast)
            # set values in df_f_out back to actual max and min
            from_diffs(df_f_in, df_f_out)
        # calculate naive bench mark
        df_bench = naive(df_f_in)

        # assess the forecast
        rmse, min_mape, max_mape, std_mape, var_mape, sem_mape = assess(df_forecast, df_f_out)
        rmse_b, min_mape_b, max_mape_b, b_std_mape, b_var_mape, b_sem_mape = assess(df_bench, df_f_out)
        skill = rmse / rmse_b
        # store the assesment
        rmses.append([rmse, rmse_b, skill, min_mape, max_mape, std_mape, var_mape, sem_mape])
        # check max min diff
        diff_o = check_max_min_diff(df_f_out)
        diff_f = check_max_min_diff(df_forecast)
        diff_b = check_max_min_diff(df_bench)
        print('Diffs original {} forecast {} bench {}'.format(diff_o, diff_f, diff_b) )
        if args.model == 'diffs':
            print('Diff prediction {} '.format(df_forecast['diff'].mean() ))

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

        # errors
        if args.errors:
            corr_max = {}
            corr_min = {}
            # correlation of errors with variables (max)
            max_errors = df_forecast['max_demand'] - df_f_out['max_demand']
            coeffs = correlation(df_f_in, max_errors)
            print('Correlation max_demand errors')
            for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
                print('{:15}         {:.3f}'.format(col,value))
                corr_max[col] = value

            # correlation of errors with variables (min)
            min_errors = df_forecast['min_demand'] - df_f_out['min_demand']
            coeffs = correlation(df_f_in, max_errors)
            print('Correlation max_demand errors')
            for col, value in sorted(coeffs.items(), key=lambda item: item[1], reverse=True ):
                print('{:15}         {:.3f}'.format(col,value))
                corr_min[col] = value
            df_errs = pd.concat([pd.Series(corr_max), pd.Series(corr_min)], keys = ['err_max', 'err_min'], axis=1)
            print(df_errs)
            # output correlation values
            output_filename = 'errors.csv'
            df_errs.to_csv(output_dir+output_filename, float_format='%.2f')

        
        
    # output all the assessments
    skill = 0.0
    if args.model == 'minute':
        print('RMSE  Naive RMSE  Skill  MAPE-min MAPE-max MAPE-std MAPE-var MAPE-sem')
        for vals in rmses:
            print("{:.3f} {:.3f}      {:.3f}  {:.3f}    {:.3f}    {:.3f}    {:.3f}    {:.3f}".format(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]))
            skill += vals[2]
    else:
        print('RMSE  Naive RMSE  Skill  MAPE-min MAPE-max')
        for vals in rmses:
            print("{:.3f} {:.3f}      {:.3f}  {:.3f}    {:.3f}".format(vals[0], vals[1], vals[2], vals[3], vals[4]))
            skill += vals[2]
    print('Average skill {}'.format(skill / len(rmses) ) )
    minutes = (time.time() - startTime) / 60.0
    print('It took {0:0.2f} minutes'.format(minutes))
