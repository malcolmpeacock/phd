import cdsapi
 
c = cdsapi.Client()
 
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': 'temperature',
        'pressure_level': '1000',
        'year': '2008',
        'month': '01',
        'day': '01',
        'time': '12:00',
        'format': 'netcdf', # Supported format: grib and netcdf. Default: grib
    },
    'download.nc')          # Output file. Adapt as you wish.
