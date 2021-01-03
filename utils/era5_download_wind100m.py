import cdsapi

def get_era4_wind(filename):
    
    # Create list of times
    hours = 1
    year = '2010'
    months=['12']
    times=[]
    for i in range(0,24,hours):
        times.append('{:02d}:00'.format(i))

    # Create dictionary defining the request
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '100m_u_component_of_wind', '100m_v_component_of_wind'],
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
#       'area': [ 72, -10.5, 36.75, 25.5, ],
        'area': [ 53, 4.0, 52.0, 5.0, ],
        'grid': ['0.25', '0.25'],
    }

    print('Downloading wind to {}'.format(filename))
    print(request)
    # Download ERA5 weather 
    c = cdsapi.Client()
    c.retrieve( 'reanalysis-era5-single-levels', request, filename)

get_era4_wind("highwind.nc")
