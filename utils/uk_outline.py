# plot era interim grid on uk outline.

import cartopy
import matplotlib.pyplot as plt

ax = plt.axes(projection=cartopy.crs.PlateCarree())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
# ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
# ax.add_feature(cartopy.feature.RIVERS)

# x0, x1, y0, y1
# ax.set_extent([-20, 10, 45, 60])
ax.set_extent([-10, 4, 48, 62])
# (-7.57216793459, 49.959999905, 1.68153079591, 58.6350001085)

# vertical lines

lat = 50.25
while lat < 61.0:
    plt.plot( [-7.5, 1.5], [lat, lat], color='gray', transform=cartopy.crs.Geodetic())
    lat = lat + 0.75

# horizontal lines

lon = -7.5
while lon < 2.0:
    plt.plot( [lon, lon], [50.25, 60.75], color='gray', transform=cartopy.crs.Geodetic())
    lon = lon + 0.75

plt.show()
