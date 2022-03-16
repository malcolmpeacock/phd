# add electric heat from gas onto historic electric and compare ramp rates etc
# with different methods and profiles.

import matplotlib.pyplot as plt
import argparse

# custom code
import stats
import readers

# peaks and ramps
def peaks(method, profile, series):
    print('{:9s} {:9s}  {:.2f}    {:.2f}       {:.2f}       {:.2f}        {:.2f}'.format(method, profile, series.max(), series.min(), series.diff().max(), series.diff().min(), series.sum() * 1e-3 ) )

def peaks_header():
    print('Method    Profile    Peak     Minimum     Ramp up  Ramp down  Annual')
 
# process command line

parser = argparse.ArgumentParser(description='Impact of eletric heat on electric.')
parser.add_argument('--year', action="store", dest="year", help='Year', default='2018' )
parser.add_argument('--method', action="store", dest="method", help='Method', default='B' )
parser.add_argument('--profile', action="store", dest="profile", help='Profile', default='bdew' )
parser.add_argument('--plot', action="store_true", dest="plot", help='Show diagnostic plots', default=False)
args = parser.parse_args()

# gas % of heat
gas_heat = { '2016' : 0.696, '2017' : 0.695, '2018' : 0.703, '2019' : 0.70 }

demand_filename = '/home/malcolm/uclan/data/electricity/espeni.csv'
demand = readers.read_espeni(demand_filename, args.year)
# convert from MWh to GWh
electric = demand * 1e-3

# read electric heat

demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{}/GBRef{}Weather{}I-{}{}.csv'.format(args.year, args.year, args.year, args.method, args.profile)
heat_demand = readers.read_copheat(demand_filename, ['electricity', 'temperature'])
# convert from MWh to GWh
electric_heat = heat_demand['electricity'] * 1e-3

# factor electric heat to all current gas heating
electric_heat = electric_heat * gas_heat[args.year]

# add electric heat to historic
new_electric = electric + electric_heat

# look at ramp rates etc and annual demands and peak
print('HOURLY')
peaks_header()
peaks(args.method, args.profile, new_electric)
peaks('Historic', 'Historic', electric)

# convert to daily
daily_electric = electric.resample('D').sum()
daily_new_electric = new_electric.resample('D').sum()

# daily plots
if args.plot:

    daily_electric.plot(color='blue', label='Daily Electricity demand {}'.format(args.year))
    daily_new_electric.plot(color='red', label='Daily Electric with heat 2018')
    plt.title('Impact of heat electrification on daily electricty demand series')
    plt.xlabel('day', fontsize=15)
    plt.ylabel('Energy (Twh) per week', fontsize=15)
    plt.legend(loc='upper center')
    plt.show()

# look at ramp rates etc and annual demands and peak
print('DAILY')
peaks_header()
peaks(args.method, args.profile, daily_new_electric)
peaks('Historic', 'Historic', daily_electric)
