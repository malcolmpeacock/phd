# common functions
import pandas as pd

def add_diffs(df, maxmin_df):
    maxmin_df['max_diff'] = maxmin_df['max_demand'] - df['demand']
    maxmin_df['min_diff'] = df['demand'] - maxmin_df['min_demand']
