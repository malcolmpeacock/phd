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

def krange(df):
    # extract k=32,42 from the datafram
    kabove31 = df['k'] > 31
    kabove31_df = df[kabove31]
    kbelow43 = kabove31_df['k'] < 43
    output = kabove31_df[kbelow43]
    return output

