# python script to do some plots to investigate demand

# contrib code
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pytz
import math
import pvlib
import glob
import torch

# custom code
import utils

# pytorch loss function for 1-R ( Pearson correlation )
def loss_1mr(X,Y):
    vx = X - torch.mean(X)
    vy = Y - torch.mean(Y)
    lossn = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) 

    return 1.0 - lossn

error=0.2
# time 0 to 4 in steps of 0.1
t = np.arange(0, 4, 0.1)
# demand is a sine function
demand = np.sin(t)
# over prediction - add error amount
over = demand + error
battery = 2.0
discharge = utils.discharge_pattern(battery, demand)
#discharge = utils.discharge_pattern_simple(battery, demand)
#print(discharge)
discharge_over = utils.discharge_pattern(battery, over)
modified = demand + discharge
modified_over = demand + discharge_over

plt.plot(t, demand, color='red', linestyle='solid', label='Actual Demand')
plt.plot(t, over, color='red', linestyle='dotted', label='Over predicted demand')
plt.plot(t, discharge, color='green', linestyle='solid', label='Battery Discharge')
plt.plot(t, discharge_over, color='green', linestyle='dotted', label='Battery Discharge for forecast')
plt.plot(t, modified, color='blue', linestyle='solid', label='actual demand modified ')
plt.plot(t, modified_over, color='blue', linestyle='dotted', label='actual demand modified by discharge based on forecast')
plt.title('Over Prediction of demand')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend(loc='lower right', fontsize=12)
plt.show()

utils.print_metrics(pd.Series(demand), pd.Series(over) )

# Test pytorch R loss function
t_demand = torch.tensor(demand.astype(np.float32)).view(-1,1)
t_over = torch.tensor(over.astype(np.float32)).view(-1,1)
t_r = loss_1mr(t_demand, t_over)
print(t_r)

late_demand = np.sin(t-0.5)
discharge_late = utils.discharge_pattern(battery, late_demand)
modified_late = demand + discharge_late
plt.plot(t, demand, color='red', linestyle='solid', label='Actual Demand')
plt.plot(t, late_demand, color='red', linestyle='dotted', label='late predicted demand')
plt.plot(t, discharge, color='green', linestyle='solid', label='Battery Discharge')
plt.plot(t, discharge_late, color='green', linestyle='dotted', label='Battery Discharge for forecast')
plt.plot(t, modified, color='blue', linestyle='solid', label='actual demand modified ')
plt.plot(t, modified_late, color='blue', linestyle='dotted', label='actual demand modified by discharge based on forecast')
plt.title('Late Prediction of demand')
plt.xlabel('Time')
plt.ylabel('Demand')
plt.legend(loc='lower right', fontsize=12)
plt.show()

utils.print_metrics(pd.Series(demand), pd.Series(late_demand))

t_late = torch.tensor(late_demand.astype(np.float32)).view(-1,1)
t_r = loss_1mr(t_demand, t_late)
print(t_r)
