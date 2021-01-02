# Output daily GB temp from ERA 5 data.
# the mean of all the grid squares (mapped by population)

# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse
from scipy.stats import ttest_ind

# Custom scripts
import scripts.download as download 
import scripts.read as read
import scripts.preprocess as preprocess
import scripts.demand as demand
import scripts.cop as cop
import scripts.write as write
import scripts.metadata as metadata

home_path = os.path.realpath('../')

input_path = os.path.join(home_path, 'input')
interim_path = os.path.join(home_path, 'interim')
output_path = os.path.join(home_path, 'output', 'avtemp')

os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
os.environ["ECMWF_API_KEY"] = "45a818c5dfcb79f44a3a5dd3217a2866"
os.environ["ECMWF_API_EMAIL"] = "MPeacock2@uclan.ac.uk"

year = 2018

# weather
download.temperatures(input_path, year, year)

# population
download.population(input_path)

print('Mapping population ... ')

mapped_population = preprocess.map_population(input_path, interim_path)


print('Processing temp ... ')

temperature = preprocess.temperature(input_path, year, mapped_population, interim_path)

num_previous = 1
reference_temperature = demand.reference_temperature(temperature['air'],num_previous)

print(reference_temperature.head())

# mean temperature of the grid points (time series)

t = reference_temperature.mean(axis=1) - 273.15

print(t.head())

output_file = output_path + '/' + str(year) + '.csv'
t.to_csv(output_file, sep=',', decimal='.', float_format='%g')

# mean temperature of the year for each weather grid square.

t_grid = reference_temperature.mean(axis=0) - 273.15
df = pd.concat([t_grid, mapped_population], axis=1, keys=['temperature', 'population'])
print(df)
t_stat, pval = ttest_ind(df['temperature'], df['population'])
print('T Statistic {} P-Value {}'.format(t_stat, pval) )
# plt.scatter(df['temperature'], df['population'])
df.plot(x='temperature', y='population', style='o')
plt.show()
c = t_grid.corr(mapped_population)
print(c)
