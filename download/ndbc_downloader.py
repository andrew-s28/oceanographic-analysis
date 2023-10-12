# input/output, html parsing, requests, and os functionality
import io
import os
import requests
from tqdm import tqdm as tq
import numpy as np
import xarray as xr
from pycoare.coare import c35
import util
import warnings

# get nitrate data
print("Newport South Beach - NWPO3")
print("Oregon Shelf - 46097")
print("Oregon Offshore - 46050")
print("Enter NDBC site code: ", end='')
site = input()

if site == '46050':
    zt = 3.7
    zu = 4.1
elif site == '46097':
    zt = 4.0
    zu = 4.5
elif site == 'NWPO3':
    zt = 6.4+9.1
    zu = 9.4+9.1
else:
    print('Bad site selection.')
    exit(1)

# setup defaults to use in subsequent data queries
url = "https://dods.ndbc.noaa.gov/thredds/catalog/data/stdmet/" + site + "/catalog.html"

tag = r'[1-2][0-9][0-9][0-9].*\.nc$'  # setup regex for files we want (*=anything,$=end of line)
nc_files = util.list_files(url, tag)
file_url = 'https://dods.ndbc.noaa.gov/thredds/fileServer/'
nc_url = [file_url + i + '#mode=bytes' for i in nc_files]  # combine files, add mode to ensure download works

# load datasets
ds = []  # empty arrays for datasets
for i, f in (enumerate(tq(nc_url, desc='Downloading datasets'))):
    r = requests.get(f, timeout=(3.05, 120))
    # ensure request worked
    if r.ok:
        ds.append(xr.load_dataset(io.BytesIO(r.content)))
        ds[i].load()

# some renaming and new variables
for i, d in enumerate(ds):
    ds[i] = ds[i].squeeze()
    ds[i]['rh'] = util.relative_humidity_from_dewpoint(
        ds[i].air_temperature, ds[i].dewpt_temperature
    )
    ds[i]['rh'] = ds[i]['rh'].where(
        ds[i]['rh'] < 500, 75)
    ds[i]['air_temperature'] = ds[i]['air_temperature'].where(
        ds[i]['air_temperature'] < 500, 12.5)
    ds[i]['sea_surface_temperature'] = ds[i]['sea_surface_temperature'].where(
        ds[i]['sea_surface_temperature'] < 500, 12.5)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in power")
    for i, d in enumerate(ds):
        ds[i]['wind_east'], ds[i]['wind_north'] = util.uv_from_spddir(ds[i]['wind_spd'],
                                                                      ds[i]['wind_dir'])
        coare_mag = c35(ds[i]['wind_spd'].values,
                        t=ds[i]['air_temperature'].values,
                        rh=ds[i]['rh'],
                        ts=ds[i]['sea_surface_temperature'],
                        lat=ds[i]['latitude'],
                        zu=zu, zt=zt, zq=zt, out='tau')  # type: ignore <- pylance doesn't like zq/zt as float
        ds[i]['coare_mag'] = (['time'], coare_mag)
        ds[i]['coare_east'], ds[i]['coare_north'] = util.uv_from_spddir(ds[i]['coare_mag'],
                                                                        ds[i]['wind_dir'])
# flatten array to determine principal axis based on all available data from data start to data end
east = np.array([item for sublist in ds for item in sublist['wind_east'].values])
north = np.array([item for sublist in ds for item in sublist['wind_north'].values])

theta, major, minor = util.princax(east, north)
for i, d in enumerate(ds):
    ds[i]['cs'], ds[i]['as'] = util.rot(d['wind_east'], d['wind_north'], theta)
    ds[i]['coare_x'], ds[i]['coare_y'] = util.rot(d['coare_east'], d['coare_north'], theta)

for i, d in enumerate(ds):
    ds[i]['wind_east'].attrs = {'comment': 'Eastwards wind velocity',
                                'units': 'm/s'}
    ds[i]['wind_north'].attrs = {'comment': 'Northwards wind velocity',
                                 'units': 'm/s'}
    ds[i]['coare_mag'].attrs = {'comment': 'Magnitude of wind stress computed by COARE v3.5',
                                'units': 'N/m^2'}
    ds[i]['coare_east'].attrs = {'comment': 'Eastwards wind stress computed by COARE v3.5',
                                 'units': 'N/m^2'}
    ds[i]['coare_north'].attrs = {'comment': 'Northwards wind stress computed by COARE v3.5',
                                  'units': 'N/m^2'}
    ds[i]['rh'].attrs = {'comment': 'Relative humidity computed from air and dewpoint temperature',
                         'units': '%'}
    ds[i]['cs'].attrs = {'comment': 'Cross-shelf component of wind velocity computed by principal axis',
                         'units': 'm/s'}
    ds[i]['as'].attrs = {'comment': 'Along-shelf component of wind velocity computed by principal axis',
                         'units': 'm/s'}
    ds[i]['coare_x'].attrs = {'comment': 'Cross-shelf component of wind stress computed by principal axis',
                              'units': 'm/s'}
    ds[i]['coare_y'].attrs = {'comment': 'Along-shelf component of wind stress computed by principal axis',
                              'units': 'm/s'}
    ds[i].attrs = {'Principal axis angle': f'{theta + 270} deg CW of true north'}

# resample to 1D time intervals with mean method
datasets = []
for i, d in enumerate(ds):
    datasets.append(d.resample(time="1D").mean(skipna=True))

ds_bin = xr.concat(datasets, dim='time')

print("Enter name for output nc file: ", end='')
out_name = input()

if not os.path.exists('./output/'):
    os.mkdir('./output/')

ds_bin.to_netcdf('./output/' + out_name + '.nc')

for i, d in enumerate(ds):
    d.to_netcdf('./output/' + nc_files[i][-13:-3] + '.nc')
