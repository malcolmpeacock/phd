# sanity check on year of hourly data
# check for missing data.

import pandas as pd
from datetime import datetime, timedelta

def sanity(s,year,name):
    # create a 1 hourly index for the whole year and find the difference with
    # the input index (ie missing values)
    d = pd.date_range(start = year + '-01-01 00:00:00', end = year + '-12-31 23:00:00', freq='H' ).difference(s.index)
    print(d)
#   print(s)
    print('SANITY CHECK: {} values {} missing for {} '.format(len(s),len(d),name))
    # return if no missing values
    if len(d) == 0:
        return s
    # for each missing value get one from the same hour on a different day
    # (preferrably the day before, but otherwise just the earliest we can 
    #  find)
    # need to find a previous day to use. 
    # then find the correct hour - but need to do this in cases where there
    # is a record but no value for the parameter we want eg wind speed.
    newdata={}
    for dt in d:
        # if january 1st, use the lowest date we have
#       if dt.month == 1 and dt.day ==1:
#            rday = dt.values().min().day
#       else:
        # otherwise use the previous day
#            rday = dt - timedelta(days=1)
        # if no previous day, 
#       newdata[dt] = s.loc[rdate]
        newdata[dt] = float("NaN")
    
    # create a new series containing the new data and concatonate it onto
    # the orignal 
    newseries = pd.DataFrame.from_dict(data=newdata,orient='index',columns=s.keys())
    newseries.index=d
    df = pd.concat([s,newseries])
    df.sort_index(inplace=True)

    # interpolate missing values (replace the NaNs)
#   print(df)
    df = df.interpolate()
#   print(df['2018-09-12 11:00:00' : '2018-09-12 17:00:00'])
    print("Adjusted number of values {}".format(len(df)))
    return df
