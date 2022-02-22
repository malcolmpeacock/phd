# library stuff
import sys
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
import numpy as np

# custom code
import stats
import readers

years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
cops = []
# for each weather year ...
for year in years:
    file_base = 'Brhpp'
    demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(year, 2018, file_base)
    cop = readers.read_copheat(demand_filename,['ASHP_radiator'])
    print('Year {} Mean Annual COP {}'.format(year, cop.mean()) )
    cops.append(cop.mean())

mean_cop = np.mean(np.array(cops))
print('Mean cop {} with 0.8 {} with 0.67 {}'.format(mean_cop, mean_cop*0.8, mean_cop*0.67))
