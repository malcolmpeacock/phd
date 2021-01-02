# Python modules
import os
import shutil
import pandas as pd
from time import time
from datetime import date
import matplotlib.pyplot as plt
import argparse

## Script to work out number of HDD for a given year from the weather.

# Custom scripts
import scripts.download as download 
import scripts.read as read
import scripts.preprocess as preprocess
import scripts.demand as demand
import scripts.cop as cop
import scripts.write as write
import scripts.metadata as metadata

all_countries = ['AT', 'BE', 'BG', 'CZ', 'DE', 'FR', 'GB', 'HR', 
                 'HU', 'IE', 'LU', 'NL', 'PL', 'RO', 'SI', 'SK'] # available
countries = ['GB', 'LU']  # selected for calculation

methods = { "R" : "Ruhnau", "W" : "Watson", "S" : "HDD 15.5", "H" : "HDD 12.8" }

# verify command line

parser = argparse.ArgumentParser(description='Calculate HDD from a weather year.')
parser.add_argument('weather', type=int, help='Weather year')
parser.add_argument('--version', action="store", dest="version", help='Version - subdirectory to store output in, defaults to year', default=None )

args = parser.parse_args()
year = args.weather
if args.version:
    version = args.version
else:
    version = str(year)
print(' weather {} '.format( year))

home_path = os.path.realpath('../')

input_path = os.path.join(home_path, 'input')
interim_path = os.path.join(home_path, 'interim')
output_path = os.path.join(home_path, 'output', version)

for path in [input_path, interim_path, output_path]:
    os.makedirs(path, exist_ok=True)

year_start = year
year_end = year
 
os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
os.environ["ECMWF_API_KEY"] = "45a818c5dfcb79f44a3a5dd3217a2866"
os.environ["ECMWF_API_EMAIL"] = "MPeacock2@uclan.ac.uk"

# weather
download.wind(input_path)
download.temperatures(input_path, year_start, year_end)

# population
download.population(input_path)

print('Mapping population ... ')

mapped_population = preprocess.map_population(input_path, countries, interim_path)

mapped_population['LU']

print('Processing wind ... ')

wind = preprocess.wind(input_path, mapped_population)

print('Processing temp ... ')

temperature = preprocess.temperature(input_path, year_start, year_end, mapped_population, interim_path)

print('Reference temp ... ')

reference_temperature = demand.reference_temperature(temperature['air'],4)

hdd = demand.hdd(reference_temperature['GB'], mapped_population['GB'], base_temp=15.5)

print('Heading degree days for {0} {1:.2f}'. format(year, hdd))
