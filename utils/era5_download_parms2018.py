import cdsapi

# Download various ERA5 weather parameters for 2018 for EU
#  

def get_era5_parms(filename, year):
    
    # Create list of times
    hours = 1             # hourly
    months=[]
    for month in range(12):
            months.append("{:02d}".format(month+1))
    times=[]
    for i in range(0,24,hours):
        times.append('{:02d}:00'.format(i))

    # Create dictionary defining the request
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', '2m_dewpoint_temperature', 'total_precipitation', 'surface_pressure', 'total_cloud_cover', 'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards', 'surface_solar_radiation_downward_clear_sky'],
        'year': year,
        'month': months,
        'day': [
        '01', '02', '03',
        '04', '05', '06',
        '07', '08', '09',
        '10', '11', '12',
        '13', '14', '15',
        '16', '17', '18',
        '19', '20', '21',
        '22', '23', '24',
        '25', '26', '27',
        '28', '29', '30',
        '31',
        ],
        'time': times,
        'area': [ 72, -10.5, 36.75, 25.5, ],
#       'area': [ 61, -2.0, 50.0, 8.0, ],
#       'grid': ['0.25', '0.25'],
        'grid': ['0.75', '0.75'],
    }

    print('Downloading wind to {}'.format(filename))
    print(request)
    # Download ERA5 weather 
    c = cdsapi.Client()
    c.retrieve( 'reanalysis-era5-single-levels', request, filename)

#year = '2018'
year = '2009'
get_era5_parms("/home/malcolm/uclan/tools/python/scripts/heat/input/weather/ERA5_parms{}.nc".format(year), year)
