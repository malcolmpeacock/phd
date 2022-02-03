import pandas as pd
import matplotlib.pyplot as plt
# custom code
import stats
import readers

year = 2018
reference = 2018
file_base = 'Brhpp_copS'
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(year, reference, file_base)
demand_r = readers.read_copheat(demand_filename,['electricity','heat','temperature'])
file_base = 'Brhpp_copR'
demand_filename = '/home/malcolm/uclan/tools/python/scripts/heat/output/{0:}/GBRef{1:}Weather{0:}I-{2:}.csv'.format(year, reference, file_base)
demand_s = readers.read_copheat(demand_filename,['electricity','heat','temperature'])

stats.print_stats_header()
stats.print_stats(demand_r['electricity'], demand_s['electricity'], 'BDEW', 2, False)


dailyr = demand_r['electricity'].resample('D').sum() * 1e-6
dailys = demand_s['electricity'].resample('D').sum() * 1e-6

print('Total electric heat: Ruhnau {:.2f} Staffel {:.2f}'.format(dailyr.sum(), dailys.sum() ) )

dailyr.plot(label='Electric Heat Ruhnau COP')
dailys.plot(label='Electric Heat Staffel COP')
plt.title('Comparison of using different COP curve for electric heat')
plt.xlabel('Day of the year', fontsize=15)
plt.ylabel('Demand (kWh)', fontsize=15)
plt.legend(loc='upper right', fontsize=15)
plt.show()

