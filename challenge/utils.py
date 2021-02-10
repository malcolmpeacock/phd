import pandas as pd
import statsmodels.api as sm
import math
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

def index2k(index):
    k = (index.hour * 2) + (index.minute / 30) + 1
    return math.floor(k)

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

def print_metrics(s1,s2, plot=False):
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
    # plots
    if plot:
        x = np.array([s2.min(),s2.max()])
        y = p[0] * x
        plt.scatter(s1,s2,s=12)
        plt.plot(x,y,color='red')
        plt.title('Fit ' )
        plt.xlabel(s1.name)
        plt.ylabel(s2.name)
        plt.show()


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

# function to find the difference in weather between 2 days

def day_diff(day1, day2, df1, df2, parm):
    day1_data = df1.loc[day1.strftime('%Y-%m-%d')]
#   print(day1_data)
    day2_data = df2.loc[day2.strftime('%Y-%m-%d')]
#   print(day2_data)
    diff = day1_data[parm].values - day2_data[parm].values
#   print(diff)
    score = np.abs(diff).sum() / len(day1_data)
#   print(day1,day2,score)
    return score

# find closest weather day to a given day
def find_closest_day(given_day, days, df1, df2, parm):
    closest_day = days[0]
    closest_day_score = day_diff(given_day, closest_day, df1, df2, parm)
    for day in days:
        if day!=given_day:
            day_diff_score = day_diff(given_day, day, df1, df2, parm)
            if day_diff_score < closest_day_score:
                closest_day = day
                closest_day_score = day_diff_score
    return closest_day, closest_day_score

# function to assess the forecast accuracy between 2 days
def forecast_diff(day1, day2, parm, df):
    day1_data = df.loc[day1.strftime('%Y-%m-%d')]
    day2_data = df.loc[day2.strftime('%Y-%m-%d')]
    day1k = krange(day1_data)
    day2k = krange(day2_data)
    diff = day1k[parm].values - day2k[parm].values
    score = np.abs(diff).sum()
#   print(day1,day2,score)
    return score

def locations():
    points = {'pv' : [50.33, -4.034],
              'w1' : [50.5,  -4.375],
              'w2' : [50.5,  -3.75 ],
              'w3' : [51.0,  -3.75 ],
              'w4' : [51.5,  -2.5  ],
              'w5' : [50.0,  -4.375],
              'w6' : [50.0,  -3.75 ] }
    return points
 

def location_lists():
    points = locations()
    lat=[]
    lon=[]
    label=[]
    for key, point in points.items():
        lat.append(point[0])
        lon.append(point[1])
        label.append(key)
    return label, lat, lon
    
def solution_score(solution, pv, demand):
    C1 = 3.0
    C2 = 1.0
    Sd = []
    new_demand = demand['prediction'].copy()
    days = pd.Series(solution.index.date).unique()
    # For each day ...
    for day in days:
        peak_old = 0
        peak_new = 0
        day_demand = demand.loc[day.strftime('%Y-%m-%d'), 'prediction']
        day_pv = pv.loc[day.strftime('%Y-%m-%d'), 'prediction']
        day_solution = solution.loc[day.strftime('%Y-%m-%d')]
#       print(day_demand, day_pv, day_solution)
        B=[]
        P=[]
        # for each period ...
        for index, value in day_pv.iteritems():
#           k = index.hour * 2 + (index.minute / 30)
            k = index2k(index)
#           G=[]
            # Calculate PV / Grid proportion
            if k<32:
                Bk = day_solution[index]
#               print('k {} Bk {}'.format(k,Bk) )
                B.append( day_solution[index] )
                P.append( min( Bk, day_pv[index] ) )
            # Calculate Peak reduction
            if k>31 and k<43:
                if peak_old < day_demand[index]:
                    peak_old = day_demand[index]
                new_demand[index] = day_demand[index] + day_solution[index]
                if peak_new < new_demand[index]:
                    peak_new = new_demand[index]
#               G[k] = B[k] - P[k]
        P1 = sum(P) / sum(B)
        P2 = 1 - P1
        Rp = 100 * ( peak_old - peak_new ) / peak_old
        print('P1 {} P2 {} peak_old {} peak_new {} Rp {}'.format(P1, P2, peak_old, peak_new, Rp) )
        Sd.append(Rp * ( P1 * C1 + P2 * C2 ) )
    Sfinal = sum(Sd) / 7.0
    return Sfinal, new_demand