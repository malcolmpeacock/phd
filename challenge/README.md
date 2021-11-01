# Presumed Open Data (POD) Challenge Solution - team u_cvml
Python programs to:
* clean and augment the data.
* to create PV and demand forecasts. 
* generate battery charging solution.

## Author
 Malcolm Peacock 2020 

## Programs

The programs would be run in the following order to create a solution
* **weather_clean.py** converts weather from hourly to half hourly. adds weighted averages and means
* **demand_clean.py** replaces odd low or high values by linear interpolation. drops whole days if many low or high values.
* **pv_clean.py** replaces low, high or missing values by interpolation or drops whole days if too many values to replace.
* **merge_data.py** reads the output from the above 3 programs, replaces missing days by similar days forecasting, merges the data together and augments with additional values.
* **pv_forecast.py** creates a PV forecast.
* **demand_forecast.py** creates a demand forecast
* **create_solution.py** creates a solution file from the demand and pv forecasts
* **solution_validate.py** validates a solution file - checking that the battery is not overcharged etc.
* **how_we_did.py** calculates the score from the next set of data and creates a plot to see how it performed.
* **utils.py** common utility functions

## Example of use
  python demand_forecast.py set3 --method regs --alg lgbm
  python create_solution.py set3
  python how_we_did.py 3 --sf --plot
