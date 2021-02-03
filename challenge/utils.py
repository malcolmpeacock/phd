import pandas as pd

def missing_times(df,f):
    d = pd.date_range(start = df.index[0], end = df.last_valid_index() , freq=f ).difference(df.index)
    return len(d)

def replace_day(df,d,r):
    bad_day = df[d]
    print(bad_day)
    previous_day = df[r]
    previous_day.index = bad_day.index
    print(previous_day)
    df.update(previous_day)
    return(df)
