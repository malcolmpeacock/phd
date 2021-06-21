# Plot EV charging hourly profile.

# library stuff
import sys
import pandas as pd
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np

# calculate ev profile
ev = np.full(24,0.0)
hour = np.full(24,0)
for i in range(24):
    hour[i] = i
# night charging at 1am, 2am, 3am
ev[1] = 0.2
ev[2] = 0.2
ev[3] = 0.2
# peak daily charging
ev[14] = 0.1
ev[15] = 0.1
ev[16] = 0.1
ev[17] = 0.1
print(hour)
print(ev)

plt.plot(hour,ev)
plt.title('Hourly EV charging profile')
plt.xlabel('hour of the day', fontsize=15)
plt.ylabel('Fraction of daily value', fontsize=15)
plt.show()

