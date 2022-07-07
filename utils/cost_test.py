import pandas as pd
import storage

# test values from table 2
#
num_years = 9
annual_demand = 335 * 1e6     # MWh
wind_fraction = 0.83          # %  
wind_energy = 2736.7 * 1e6    # MWh
pv_energy = 560.52 * 1e6      # MWh
storage_size = 116.48 * 1e9   # kwh
max_charge = 103.32 * 1e6     # kW
max_discharge = 61.26 * 1e6   # kW
# TCoE = 94.62 £/MWh

alpha = 3             # £/kWh
beta  = 300           # £/kW
life_time = 30        # years
gen_offshore = 57.5   # £/MWh
gen_onshore = 46      # £/MWh
offshore_f = 0.467    # %
onshore_f = 0.533     # %
gen_solar = 60        # £/MWh
store_cap = alpha * storage_size
store_pow = beta * max(max_charge, max_discharge)
c_store = store_cap + store_pow
Cf = 0.0
Cw = wind_energy * ( gen_offshore * offshore_f + gen_onshore * onshore_f)
Cs = pv_energy * gen_solar
Ed = annual_demand * num_years      # MWh
Cb = c_store * (num_years / life_time)
TC = Cf + Cw + Cs + Cb
TCoE = TC / Ed
print("Cstore {} store_cap {} store_pow {} Cw {} Cs {} TCoE {}".format(c_store, store_cap/c_store, store_pow/c_store, Cw, Cs, TCoE))

print("Ed {} WindP {} SolarP {} StoreP {} ".format(Ed, Cw/TC, Cs/TC, Cb/TC ))

num_days = num_years * 365.25
normalise_factor = annual_demand / 365.25
size_days = storage_size / normalise_factor
wind_days = wind_energy / (normalise_factor * num_days)
pv_days = pv_energy/ (normalise_factor * num_days)
discharge_rate = max_discharge / normalise_factor
charge_rate =  max_charge / normalise_factor
data = { 'storage' : [size_days], 'wind_energy' : [wind_days], 'pv_energy': [pv_days], 'variable_energy' : [0], 'base' : [0], 'charge_rate' : [charge_rate], 'discharge_rate': [discharge_rate], 'variable': [0] }

df = pd.DataFrame(data)
print(df)
storage.generation_cost(df, 'caes', normalise_factor, num_years, False, 'both', 'B')
print(df)
