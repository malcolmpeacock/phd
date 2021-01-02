# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt

# Custom scripts
import scripts.download as download 
import scripts.read as read
import scripts.preprocess as preprocess
import scripts.demand as demand
import scripts.cop as cop
import scripts.write as write
import scripts.metadata as metadata

# version = '2020-03-12'
version = '2020-04-02'
changes = 'First Attempt'

home_path = os.path.realpath('../')

input_path = os.path.join(home_path, 'input')
interim_path = os.path.join(home_path, 'interim')
output_path = os.path.join(home_path, 'output', version)

countries = ['GB', 'LU']  # selected for calculation

methods = { "R" : "Ruhnau", "W" : "Watson", "S" : "HDD 15.5", "H" : "HDD 12.8" }
method = "R"

year_start = 2008
year_end = 2018
 
os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
os.environ["ECMWF_API_KEY"] = "45a818c5dfcb79f44a3a5dd3217a2866"
os.environ["ECMWF_API_EMAIL"] = "MPeacock2@uclan.ac.uk"

# weather
# download.wind(input_path)
# download.temperatures(input_path, year_start, year_end)

# population
# download.population(input_path)

print('Mapping population ... ')

mapped_population = preprocess.map_population(input_path, countries, interim_path)

mapped_population['LU']

print('Processing wind ... ')

wind = preprocess.wind(input_path, mapped_population)

print('Processing temp ... ')

temperature = preprocess.temperature(input_path, year_start, year_end, mapped_population, interim_path)

print('Reference temp ... ')

reference_temperature = demand.reference_temperature(temperature['air'],4)

print('Daily parms ... ')
daily_parameters = read.daily_parameters(input_path)

print('Daily heat and water ... ')
if method == 'R':
    daily_heat = demand.daily_heat(reference_temperature, wind, daily_parameters)
    daily_water = demand.daily_water(reference_temperature, wind, daily_parameters)

if method == 'W':
    daily_heat = demand.watson_daily_heat(reference_temperature, wind, daily_parameters)
    daily_water = demand.watson_daily_water(reference_temperature, wind, daily_parameters)

if method == 'H':
    daily_heat = demand.hdd_daily_heat(reference_temperature, wind, daily_parameters, 12.8)
    daily_water = demand.hdd_daily_heat(reference_temperature, wind, daily_parameters, 12.8)

if method == 'S':
    daily_heat = demand.hdd_daily_heat(reference_temperature, wind, daily_parameters, 15.5)
    daily_water = demand.hdd_daily_heat(reference_temperature, wind, daily_parameters, 15.5)
 
print(daily_heat)


# Localize Timestamps (including daylight saving time correction)
# df_country = localize(df[country], country)
# Weighting
# df_cb = df_country[building_type] * population
# Scaling to building database
# Scaling to 1 TWh/a
# Change index to UCT
# results.append(country_results.tz_convert('utc'))
