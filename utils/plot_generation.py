# read and plot electric generation time series 
#

import sys
import matplotlib.pyplot as plt
import stats

import readers

year = '2018'

electric_filename = '/home/malcolm/uclan/data/ElectricityDemandData_' + year + '.csv'

electric = readers.read_electric(electric_filename)

electric['EMBEDDED_WIND_GENERATION'].plot(label='Wind')
electric['EMBEDDED_SOLAR_GENERATION'].plot(label='Solar')
plt.title('Renewable generation 2018')
plt.xlabel('Day of the year')
plt.ylabel('Generation (MWh)')
plt.legend(loc='upper right')
plt.show()
