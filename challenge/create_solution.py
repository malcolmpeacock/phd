# python script to create a charging strategy from the demand and pv forecast

# contrib code
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.signal import gaussian
from scipy.ndimage import filters

# custom code
import utils

# function defining the constraints on the battery charge
# do we need a constraint that charge is always -ve ? and <2.5 ?
def con(charge):
    return sum(charge) + battery

# function to optimize ie the peak demand by adding the battery 
# charge (which is negative)
def peak(c):
    return (demand_krange['prediction'] + c ).max()

# main program

# process command line

parser = argparse.ArgumentParser(description='Create charging strategy.')
parser.add_argument('set', help='weather file eg set0')
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
parser.add_argument('--perfect', action="store_true", dest="perfect", help='Use actual PV instead of the forecast', default=False)
parser.add_argument('--capacity', action="store", dest="capacity", help='Battery capacity in 0.5MWhs' , type=float, default=12.0 )

args = parser.parse_args()
dataset = args.set
print('Creating soution for set {}'.format(dataset) )

# read in the data
input_dir = "/home/malcolm/uclan/challenge/output/"
new_dir = "/home/malcolm/uclan/challenge/input/"

# demand data
demand_filename = '{}demand_forecast_{}.csv'.format(input_dir, dataset)
print('Reading demand from {}'.format(demand_filename) )
demand = pd.read_csv(demand_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
demand['k'] = utils.index2ks(demand.index)

# pv data
pv_filename = '{}pv_forecast_{}.csv'.format(input_dir, dataset)
print('Reading pv from {}'.format(pv_filename) )
pv = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)

# create pv weighted average
#pv['average'] = pv['prediction'].rolling(window=7, center=True).mean()
#pv['average'].fillna(0.0)

#b = gaussian(39, 7)
b = gaussian(39, 3)
gf = filters.convolve1d(pv['prediction'].values, b/b.sum())
pv['average'] = gf

# create the perfect pv solution from the next set
if args.perfect:
    set_num = int(dataset[3:4]) + 1
    pv_filename = new_dir + 'pv_train_set{}.csv'.format(set_num)
    pv_new = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    pv_filename = new_dir + 'pv_train_set{}.csv'.format(set_num-1)
    pv_old = pd.read_csv(pv_filename, header=0, sep=',', parse_dates=[0], index_col=0, squeeze=True)
    new_start = len(pv_old)
    if set_num == 5:
        new_start = 0
    print('Perfect solution using set {} new_start {}'.format(set_num, new_start))
#   set forecast to the pv power from the next set.
    pv['average'] = pv_new['pv_power_mw'].values[new_start:new_start+48*7]

print('PV sum {}'.format(pv['average'].sum()))

# create solution file of zeros

solution = pv.copy()
solution['charge_MW'] = 0.0
solution['k'] = utils.index2ks(solution.index)

# For each day ...
days = solution.resample('D', axis=0).mean().index
for day in days:
    print('Creating solution for day: {}'.format(day) )
    pv_day = pv[day : day + pd.Timedelta(hours=23,minutes=30)].copy()

    # get charging pattern
    cpoints = utils.charge_points(pv_day)

    tolerance = 0.001

    # this is 12 rather than 6 to avoid multiplying by 0.5 all the time
    # ( so its 6 half MWhs )
#   capacity = 12 - tolerance
    capacity = args.capacity

    # take off anything exceeding the max pv we could have
    cs_ghi = pv_day[pv_day['k'] < 32]['cs_ghi']
    pv_max = cs_ghi * 0.8 * 5.0 * 0.002
    pv_max = pv_max.loc[cpoints.index]
    charge_points = np.minimum(cpoints, pv_max)

    # if the charge pattern didn't fully charge, then top up a bit
    points_sum = np.sum(charge_points)
    remaining = capacity - points_sum

    # keep dividing the remaining charge evenly amongst the points where the
    # theoretical power generation is greater than zero
    npoints = len( pv_max[pv_max > 0.0] )
    addition = 1.0
    while remaining > 0.0001 and addition > 0.0001:
        topup = remaining / npoints
#       print('Day {} Remaining {} topup {} npoints {}'.format(day, remaining, topup, npoints) )
        # top charge values from remaining for each k period
        for i in range(len(charge_points)):
            k = i+1
            charge_value = charge_points[i]
            pm = pv_max.values[i]
#           print('{} Remaining {} topup {} charge_value {} at k {} pv_max {}'.format(day, remaining, topup, charge_value, k, pm) )
            # don't exceed battery limit or pv generation limit
            limit = min(2.5,pm)
            old_value = charge_value
            # incase addition ended up negative.
            addition = max(min(topup, limit - old_value), 0.0)
            charge_value = old_value + addition
#           print('Topping up from {} to {} with {} topup {} remaining {} at k {}'.format(old_value, charge_value, addition, topup, remaining, k) )
            remaining = remaining - addition
            charge_points[i] = charge_value
#       print(charge_points)

    if addition<=0.0001:
        print('WARNING: did not top up with max PV. remaining {} adding more to other times of day'.format(remaining))
        for i in range(len(charge_points)):
            if charge_points[i] < 2.5:
                addition = min(2.5 - charge_points[i], remaining)
                charge_points[i] += addition
                remaining -= addition
#               print('Adding {} remaining {}'.format(addition, remaining))
#   print(charge_points)
    cpoints.update(charge_points)
#   print(cpoints)

    # now output stored these points in the solution

    battery = 0.0
    for index, value in cpoints.iteritems():
        # double check on not exceeding the battery
        charge_output = min(value, capacity-battery)
#       print('Adding Charge {}'.format(charge_output) )
            # don't overfill the battery
        if charge_output>0:
#           solution['charge_MW'][index] = charge_output
            solution.loc[index, 'charge_MW'] = charge_output
            battery += charge_output

    # get the demand for this day and solution
    demand_day = demand[day : day + pd.Timedelta(hours=23,minutes=30)]
    solution_day = solution[day : day + pd.Timedelta(hours=23,minutes=30)]
    # Get k=32 to 42
    demand_krange = utils.krange(demand_day)
    solution_krange = utils.krange(solution_day)
    # intialise the discharge solution equally.
    solution_krange['charge_MW'] = battery / len(solution_krange)
    # define the contstraint as an equality
    #  :-con is a function checking sum(charges) = battery 
    cons = { 'type' : 'eq', 'fun': con }
    x0 = np.array(solution_krange['charge_MW'])
#   res = minimize(peak, x0, method='trust-constr', options={'disp': True, 'maxiter':1000}, constraints=cons, bounds=Bounds(-2.5,0.0) )
#   res = minimize(peak, x0, method='SLSQP', options={'xatol': 1e-8, 'disp': True, 'maxiter':50}, constraints=cons, bounds=Bounds(-2.5,0.0) )
#
# descreasing ftol seems to improve the result
    res = minimize(peak, x0, method='SLSQP', options={'disp': True, 'maxiter':200, 'ftol':1e-11}, constraints=cons, bounds=Bounds(-2.5,0.0) )
#   res = minimize(peak, x0, method='SLSQP', options={'disp': True, 'maxiter':200, finite_diff_rel_step: None}, constraints=cons, bounds=Bounds(-2.5,0.0), jac='2-point' )
#   print(res.x)
    print('Battery: {} Sum of Charges {}'.format(battery, np.sum(res.x)) )
    solution.loc[solution_krange.index, 'charge_MW'] = res.x

final_score, new_demand = utils.solution_score(solution['charge_MW'], pv['average'], demand['prediction'])
print('Final Score {}'.format(final_score) )

if args.plot:
    pv['prediction'].plot(label='PV Generation Forecast', color='red')
    pv['average'].plot(label='PV Generation Forecast', color='red', linestyle = 'dotted')
    # theorectical max? PR=0.8, 5.0MwP of plant, 2 because of half hour periods
    pv['pv_max'] = (pv['cs_ghi'] * 0.8 * 5.0 ) * 0.002
    pv['pv_max'].plot(label='Max PV from clear sky ghi', color='orange', linestyle = 'dotted')
    new_demand.plot(label='Modified demand', color='yellow')
    demand['prediction'].plot(label='Demand Forecast', color='blue')
    solution['charge_MW'].plot(label='Battery Charge', color='green')
    plt.title('Charging Solution')
    plt.xlabel('Hour of the year', fontsize=15)
    plt.ylabel('MWh', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)
    plt.show()



output_dir = "/home/malcolm/uclan/challenge/output/"
output_filename = '{}solution_{}.csv'.format(output_dir,dataset)

solution = solution['charge_MW']
solution = solution.squeeze()
solution.index.rename('datetime', inplace=True)
#solution.to_csv(output_filename, float_format='%g')
solution.to_csv(output_filename)

output_filename = output_dir + 'modified_demand.csv'
new_demand.to_csv(output_filename, float_format='%g')
