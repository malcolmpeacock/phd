# Plot energy for different proportions of wind and solar

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

cf_wind = 0.28
cf_pv = 0.116

data = { 'SF' : [], 'CW' : [], 'CS': [], 'pv' : [], 'wind': [] }
for isf in range(17,21,1):
    SF = isf / 10.0
    for icw in range(0,105,5):
        CW= icw / 100.0
        CS = 1 - CW
        e_wind = CW * cf_wind * SF
        e_pv = CS * cf_pv * SF
        data['SF'].append(SF)
        data['CW'].append(CW)
        data['CS'].append(CS)
        data['pv'].append(e_pv)
        data['wind'].append(e_wind)

df = pd.DataFrame(data)
df['energy'] = df['pv'] + df['wind']
print(df)
df.plot.scatter(x='CW', y='CS', c='energy', colormap='viridis')
plt.xlabel('Proportion of wind')
plt.ylabel('Porportion of solar')
plt.title('different proportions of wind and solar.')
plt.show()

df.plot.scatter(x='wind', y='pv', c='energy', colormap='viridis')
plt.xlabel('Proportion of wind')
plt.ylabel('Porportion of solar')
plt.title('different energy of wind and solar.')
plt.show()
