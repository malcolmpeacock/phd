import pandas as pd
import statsmodels.api as sm
import math
from datetime import datetime
from datetime import timedelta

def missing_times(df,f):
    d = pd.date_range(start = df.index[0], end = df.last_valid_index() , freq=f ).difference(df.index)
    return len(d)

def replace_day(df,d,r):
    bad_day = df.loc[d]
    print(bad_day)
    previous_day = df.loc[r]
    previous_day.index = bad_day.index
    print(previous_day)
    df.update(previous_day)
    return(df)

def krange(df):
    # extract k=32,42 from the datafram
    kabove31 = df['k'] > 31
    kabove31_df = df[kabove31]
    kbelow43 = kabove31_df['k'] < 43
    output = kabove31_df[kbelow43]
    return output

def print_metrics(s1,s2):
    # Mean Absolute Error (MAE)
    mae = ( s1 - s2 ).abs().mean()
    # Root Mean Square Error (RMSE)
    rmse = ( ( s1 - s2 ) **2 ).mean() ** .5
    average = s1.mean()
    # Normalised Root Mean Square Error (nRMSE)
    nrmse = rmse / average
    # Pearsons correlation coefficient
    corr = s1.corr(s2)
    # Regression Rsquared.
    model = sm.OLS(s1.to_numpy(), s2.to_numpy())
    results = model.fit()
    p = results.params
    gradient = p[0]
    rsquared = results.rsquared
    # output results
    print("MAE   Correlation nRMSE rsquared ")
    print('{0:.2f}  {1:.2f}        {2:.2f}  {3:.3f} '.format(mae, corr, nrmse, rsquared) )

# extract the week to forecast into a different df.
def extract_forecast_week(week_arg,df,parm,forecast):
    days = df.resample('D', axis=0).mean().index.date
    nweeks = math.floor(len(days) / 7 )
    print("Number of weeks {}".format(nweeks) )
    if week_arg == 'first':
       week=0
    else:
       if week_arg == 'last':
           week = nweeks -1
       else:
           week = int(week)
    first_day = days[len(days)-1] + pd.Timedelta(days=1) - pd.Timedelta(weeks=nweeks-week)
    print(type(first_day))
    last_day  = first_day + pd.Timedelta(days=6)
    last_day  = datetime.combine(last_day, datetime.min.time())
    last_day  += timedelta(hours=23,minutes=30)
    print(first_day, last_day)
    columns = forecast.columns.append(pd.Index([parm]))
    print(type(columns))
    forecast = df.loc[first_day : last_day]
    forecast = forecast[columns]
    # drop this week from main data as we will forecast it
    return df.drop(df[first_day : last_day].index), forecast

