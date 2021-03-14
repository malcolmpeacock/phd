import pandas as pd
import statsmodels.api as sm
import math
from datetime import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import pytz
from scipy.optimize import minimize
from scipy.optimize import Bounds

# get k for a single index value
def index2k(index):
    k = (index.hour * 2) + (index.minute / 30) + 1
    return math.floor(k)

# get series of k's
def index2ks(index):
    k = (index.hour * 2) + (index.minute / 30) + 1
    return k.astype(int)

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
    # peak difference
    pd = s1.max() - s2.max()
    # Mean Error (ME)
    me = ( s1 - s2 ).mean()
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
    print("MAE   Correlation nRMSE rsquared   ME   Peak Difference")
    print('{0:.2f}  {1:.2f}        {2:.2f}  {3:.3f}   {4:.3f} {5:.3f} '.format(mae, corr, nrmse, rsquared, me, pd) )
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

# look at the first hour of the day and convert it to DST then look at
# the difference.
# if day1 is in winter and day2 is in summer then it will be -2
# if they are both in winter or summer it will be zero
def dst_diff(day1_data, day2_data):
    hour_day1 = day1_data.index[0].tz_localize('UTC').astimezone(tz=pytz.timezone('Europe/London')).hour
    hour_day2 = day2_data.index[0].tz_localize('UTC').astimezone(tz=pytz.timezone('Europe/London')).hour
    hour_diff = hour_day1 - hour_day2
    return hour_diff*2

def get_forecast(day_data, similar_day_data, parm, dst=False):
    values = similar_day_data[parm].values
    if dst:
        dd = dst_diff(day_data, similar_day_data)
        # if similar_day is in summer and the current day is winter then dd
        # will be -2 indicating we need to set its demand value further back 
        # 2 half hour periods (the clocks having moved forwards in spring)
        values[2+dd:len(values)+dd-2] = values[2:-2]
    return values

# function to find the difference in weather between 2 days

def day_diff(day1, day2, df1, df2, parm, dst=False):
    day1_data = df1.loc[day1.strftime('%Y-%m-%d')]
#   print(day1_data)
    day2_data = df2.loc[day2.strftime('%Y-%m-%d')]
#   print(day2_data)
    values1 = day1_data[parm].values
    values2 = day2_data[parm].values
    if dst:
        dd = dst_diff(day1_data, day2_data)
#       print('day1 {} day2 {} hour_diff {}'.format(day1, day2, dd) )
        # this is ignores the first 2 and last 2 elements
        values1 = values1[2:-2]
        # if day 2 is in summer then dd will be -2 indicating we need to go back
        # 2 half hour periods
        values2 = values2[2+dd:len(values2)+dd-2]
    diff = values1 - values2
#   diff = day1_data[parm].values - day2_data[parm].values
#   print(diff)
    score = np.abs(diff).sum() / len(values1)
#   print(day1,day2,score)
    return score

# find closest weather day to a given day
# Parameters:
#  given_day - day we are searching for
#       days - list of days we are searching in
#        df1 - dataframe that given_day is in
#        df2 - dataframe that days are in
#       parm - column within the df1 and df2 to compare
#        dst - to match up by Daylight Saving Time, rather than UTC
def find_closest_day(given_day, days, df1, df2, parm, dst=False):
    closest_day = days[0]
    if given_day == days[0]:
        closest_day = days[1]
    closest_day_score = day_diff(given_day, closest_day, df1, df2, parm, dst)
    for day in days:
        if day!=given_day:
            day_diff_score = day_diff(given_day, day, df1, df2, parm, dst)
            if day_diff_score < closest_day_score:
                closest_day = day
                closest_day_score = day_diff_score
    return closest_day, closest_day_score

# check day to be included
def valid_find_day(day, given_day, season, df1, df2):
    if day==given_day:
        return False
    day_data = df1.loc[day.strftime('%Y-%m-%d')]
    given_day_data = df2.loc[given_day.strftime('%Y-%m-%d')]
    if season:
        return day_data.iloc[0]['season'] == given_day_data.iloc[0]['season']
    else:
        return True

# find n closest weather days to a given day
def find_closest_days(given_day, days, df1, df2, parm, n, dst=False, season=False):
    closest_day_score = day_diff(given_day, days[0], df1, df2, parm, dst)
    closest_days=pd.Series([closest_day_score], index=[days[0]], name='sdays')
    for day in days:
        if valid_find_day(given_day, day, season, df1, df2):
            day_diff_score = day_diff(given_day, day, df1, df2, parm, dst)
            # if not got enough days yet, just add the new one.
            if len(closest_days) < n:
                closest_days[day] = day_diff_score
            else:
                if day_diff_score < closest_days.max():
                    idmax = closest_days.idxmax()
                    closest_days.drop(idmax, inplace=True)
                    closest_days[day] = day_diff_score
    return closest_days

def day_peak_diff(day, df, parm, value):
    day_data = df.loc[day.strftime('%Y-%m-%d')]
    diff = abs( day_data[parm].max() - value )
    return diff

# find n closest peak days to a given value
def find_closest_days_max(days, df, parm, value, n):
    closest_day = days[0]
    closest_day_score = day_peak_diff(closest_day, df, parm, value)
    closest_days=pd.Series([closest_day_score], index=[days[0]], name='sdays')
    for day in days:
        day_diff_score = day_peak_diff(closest_day, df, parm, value)
        # if not got enough days yet, just add the new one.
        if len(closest_days) < n:
            closest_days[day] = day_diff_score
        else:
            if day_diff_score < closest_days.max():
                idmax = closest_days.idxmax()
                closest_days.drop(idmax, inplace=True)
                closest_days[day] = day_diff_score
    return closest_days.keys()

# function to create a new day from a set of others by interpolation
def create_day(days, df, parm):
    df_days = pd.concat([ df.loc[day.strftime('%Y-%m-%d')] for day in days])
    df_days = df_days[['k', parm]]

    newday = df_days.groupby('k').mean()
    return newday

# function to get the mean power value for the given half hour periods
def create_half_hour(periods, pv_power):
    df_periods = pd.Series([ pv_power.loc[period] for period in periods] )
    return df_periods.mean()

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
    Rpd = []
    P1d = []
    new_demand = demand.copy()
    days = pd.Series(solution.index.date).unique()
    # For each day ...
    for day in days:
        peak_old = 0
        peak_new = 0
        day_demand = demand.loc[day.strftime('%Y-%m-%d')]
        day_pv = pv.loc[day.strftime('%Y-%m-%d')]
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
#       print('P1 {} P2 {} peak_old {} peak_new {} Rp {}'.format(P1, P2, peak_old, peak_new, Rp) )
        Sd.append(Rp * ( P1 * C1 + P2 * C2 ) )
        Rpd.append(Rp)
        P1d.append(P1)
    print('Rpd average {} P1 average {}'.format( sum(Rpd) / 7.0, sum(P1d) / 7.0 ) )
    Sfinal = sum(Sd) / 7.0
    return Sfinal, new_demand

# calculate the peak reduction part of the score

def peak_score(demand, discharge_pattern):
    peak_old = np.max(demand)
    new_demand = demand + discharge_pattern
    peak_new = np.max(new_demand)
    Rp = 100 * ( peak_old - peak_new ) / peak_old
    return Rp

# function to add new column to df based on weighted average
def add_weighted(df, col_prefix, newcol):
    points = locations()
    weights = { '1' : math.dist(points['pv'], points['w1']),
                '2' : math.dist(points['pv'], points['w2']),
                '5' : math.dist(points['pv'], points['w5']),
                '6' : math.dist(points['pv'], points['w6']) }

    weight_sum = 0
    for weight in weights.values():
        weight_sum += weight

    df[newcol] = 0.0
    sun_sum = df[newcol].copy()
    for loc,weight in weights.items():
        df[newcol] = df[newcol] + (df[col_prefix+loc] * weight )
    sun_sum = sun_sum + df[col_prefix+loc]
    #   print('loc {} weight {} weight_sum {}'.format(loc, weight, weight_sum) )
    df[newcol] = df[newcol] / weight_sum
    # incase there was a divide by zero due to zero temperature
    df[newcol] = df[newcol].fillna(0)

# function to find the points and threshold to charge at
def charge_points(df):
    # get periods with k < 32
    df['k'] = (df.index.hour * 2) + (df.index.minute / 30) + 1
    df['k'] = df['k'].astype(int)
    kbelow32 = df['k'] < 32
    df = df[kbelow32]
    # start with a zero line, threshold=0 and move up and down alternately
    # until sum(demands - threshold) within limit of the battery capacity
    #  this is effectively the area under the curve.
    threshold = 0
    limit = 0.05
    capacity = 12.0
    csum, points = charge_sum(df, threshold)
#   move_amount = 0.8 * df['average'].max()
    move_amount = 0.8 * csum
    print('PV Charge sum: {}'.format(csum))
    # whilst difference between sum of generation for demand and capacity
    # is less than limit ...
    while abs(csum - capacity) > limit:
        print('csum {} threshold {} move_amount {} '.format(csum, threshold, move_amount) )
        # move up or down a smaller and smaller bit
        if csum > capacity:
            threshold += move_amount
        else:
            threshold -= move_amount
        csum, points = charge_sum(df, threshold)
        move_amount = move_amount * 0.5
#       print('csum {} threshold {} move_amount {} '.format(csum, threshold, move_amount) )

    # return the points and the ( demand - threshold )
    return points

# function to return the sum of the differences of the demand values
# and the threshold for all points where this is positive
def charge_sum(df, threshold):
    above_threshold = df['average'] > threshold
    df_above = df[above_threshold]
    diff = df_above['average'] - threshold
    csum = diff.sum()
    return csum, diff

# function to find the closest half hour periods given certain parameters
# and thresholds
def find_closest(needle, haystack, thresholds, parm):
    distance = {}
    for tparm in thresholds:
        distance[tparm] = thresholds[tparm] * ( haystack[tparm].max() - haystack[tparm].min() )
#   print(distance)
    values_in_range = []
    for index, row in haystack.iterrows():
        matching = True
#       print(row)
        for tparm in thresholds:
            if abs(row[tparm] - needle[tparm]) > distance[tparm] :
                matching = False
        #       print('Did not match {} {} {} {}'.format(tparm, row[tparm], needle[tparm], distance[tparm]) )
        if matching:
        #   print('Matched {} {}'.format(parm, row[parm]) )
            values_in_range.append(row[parm])
    if len(values_in_range) ==0:
            print('ERROR: no matches found for {}'.format(index) )
            quit()
    s = pd.Series(values_in_range)
    prediction = { 'mean' : s.mean(), 'sd' : s.std(), 'n' : len(s) }
    print(prediction)
    return prediction
    
# function to find the closest half hour periods to a given period
# with a similar solar zenith
def find_closest_periods(needle, haystack, tparm, n):
    zparm = 'zenith'
    first_row = haystack.iloc[0]
    closest_score = abs(first_row[tparm] - needle[tparm])
    closest_hours=pd.Series([closest_score], index=[haystack.index[0]], name='shours')
    for index, row in haystack.iterrows():
        if abs(row[zparm] - needle[zparm]) < 10.0 :
            closest_score = abs(row[tparm] - needle[tparm])
            # if not got enough hours yet, just add the new one.
            if len(closest_hours) < n:
                closest_hours[index] = closest_score
            else:
                if closest_score < closest_hours.max():
                    idmax = closest_hours.idxmax()
                    closest_hours.drop(idmax, inplace=True)
                    closest_hours[index] = closest_score
    return closest_hours
    
# function to find the k nearest neighbours to a given period
def find_knn(needle, haystack, tparms, n):
    row = haystack.iloc[0]
# Calulate the Euclidean distance between the points based on
# all parameters in the list
    closest_score = np.linalg.norm(row[tparms].values - needle[tparms].values, axis=0)
    closest_hours=pd.Series([closest_score], index=[haystack.index[0]], name='shours')
    for index, row in haystack.iterrows():
#       print(row[tparms].values)
#       print(needle[tparms].values)
        closest_score = np.linalg.norm(row[tparms].values - needle[tparms].values, axis=0)
#       print(closest_score)
        # if not got enough hours yet, just add the new one.
        if len(closest_hours) < n:
            closest_hours[index] = closest_score
        else:
            if closest_score < closest_hours.max():
                idmax = closest_hours.idxmax()
                closest_hours.drop(idmax, inplace=True)
                closest_hours[index] = closest_score
    return closest_hours

def df_normalise(df):
    df_max = {}
    # normalise the inputs
    for column in df.columns:
        df_max[column] = df[column].max()
        if df_max[column] > 0:
            df[column] = df[column] / df_max[column]
    return df_max

def df_normalise_by(df,df_max):
    # normalise the inputs
    for column in df.columns:
        if df_max[column] > 0:
            df[column] = df[column] / df_max[column]

def sanity_check(df):
    for column in df.columns:
        if df[column].isna().sum() >0:
            print("ERROR in sanity_check {} NaN in {}".format(df[column].isna().sum(),column))
            print(df[df[column].isna()][column])
            quit()

def get_previous_week_day(dfd, day):
    # we need a previous day to get data from, but in assessing the method
    # there might not be one if it was removed due to bad data. So we then 
    # look further back
    first_day = dfd.first_valid_index().date()
    previous_found = False
    day_last_week  = day
    while not previous_found:
        day_last_week  = day_last_week - pd.Timedelta(days=7)
        print('Looking to base data of previous week on {}'.format(day_last_week))
        if day_last_week < first_day:
            print('Previous day for demand before start of data!!!!')
            quit()
        if day_last_week.strftime('%Y-%m-%d') in dfd.index:
            print('Found. Using {}'.format(day_last_week))
            previous_day = dfd.loc[day_last_week.strftime('%Y-%m-%d')].copy()
            if len(previous_day) > 0:
                previous_found = True
        else:
            print('Not Found')
    return previous_day

# function defining the constraints on the battery charge
# do we need a constraint that charge is always -ve ? and <2.5 ?
def discharge_con(charge, battery):
    return sum(charge) + battery

# function to optimize ie the peak demand by adding the battery 
# charge (which is negative)
def discharge_peak(c, demand):
    return (demand + c ).max()

def discharge_pattern(battery, demand):
    discharge = np.full(len(demand),  battery / len(demand) )
    cons = { 'type' : 'eq', 'fun': discharge_con, 'args': [battery] }
    x0 = np.array(discharge)
    res = minimize(discharge_peak, x0, args=demand, method='SLSQP', options={'disp': True, 'maxiter':200, 'ftol':1e-11}, constraints=cons, bounds=Bounds(-2.5,0.0) )
    return res.x
