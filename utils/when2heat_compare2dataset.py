# python script to validate the when2heat stuff.
import sys
from pandas import read_csv
# from pandas import ols
# from statsmodels.formula.api import ols
import statsmodels.api as sm
import stats

# read the when3heat file

def read_when2heat(filename, year):
    when2heat = read_csv(filename, header=0, sep=';', decimal=',', parse_dates=[0], index_col=0, squeeze=True, usecols=['utc_timestamp','GB_heat_demand_space','GB_heat_demand_water'] )
    when2heat.columns = ['space','water']
    when2heat_year = when2heat.loc[year+'-01-01 00:00:00':year+'-12-31 23:00:00']
    print(when2heat_year.head(7))
    print(when2heat_year.tail(7))
    return when2heat_year


# main program
my_filename = '/home/malcolm/uclan/tools/python/output/2020-04-02/when2heat.csv'
original_filename = '/home/malcolm/uclan/tools/python/output/2020-04-02/when2heat1318.csv'
year = '2016'

# read when2heat demand for GB space total
my_demand = read_when2heat(my_filename, year)

original_demand = read_when2heat(original_filename, year)

# model = ols(y=my_demand['space'], x=original_demand.ix[:, ['space']])
# model = ols(y=my_demand['space'], x=original_demand['space'])
# model = sm.OLS(y=my_demand['space'], x=original_demand['space'],exog=None)
# results = model.fit()
# print(results.summary())
rmse = ( ( my_demand['space'] - original_demand['space']) **2 ).mean() ** .5
average = my_demand['space'].mean()
nrmse = rmse / average
print(rmse, nrmse)
corr = my_demand['space'].corr(original_demand['space'])
print(corr)

# this is the real r sqaured - but correlation looks correct
# from sklearn.metrics import r2_score
# coefficient_of_dermination = r2_score(y, p(x))
stats.print_stats_header()
stats.print_stats(my_demand['space'], original_demand['space'], '2013vs201318')
