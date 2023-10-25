"""
A script to download NDBC data, compute wind stress, rotate into principal axes, and daily average.
"""
import argparse
import io
import numpy as np
import os
from pycoare.coare import c35
import requests
from tqdm import tqdm as tq
import util
import warnings
import xarray as xr

# parse command line arguments
parser = argparse.ArgumentParser(description="Download NDBC data, compute wind stress,"
                                 + " rotate into principal axes, and daily average.")
parser.add_argument('station', metavar='station', type=str, nargs=1,
                    help="station ID to download data from")
parser.add_argument('-p', '--path', metavar='path', type=str, nargs='?',
                    help="path to folder for output files",
                    default='./output/')
parser.add_argument('-f', '--file', metavar='file', type=str, nargs='?',
                    help="file name for output file, do not include .nc",
                    default=None)
args = parser.parse_args()
args = vars(args)
site = args['station'][0].lower()
out_path = args['path']
out_file = args['file']

# get ndbc site metadata for instrument elevation
ndbc_site_url = 'https://www.ndbc.noaa.gov/station_page.php?station=' + site
elev, zt, zu, zb, zt_sea, depth, radius = util.ndbc_heights(ndbc_site_url)
zt = zt + elev
zu = zu + elev
zb = zb + elev

# get all available files
url = "https://dods.ndbc.noaa.gov/thredds/catalog/data/stdmet/" + site + "/catalog.html"
tag = r'[1-2][0-9][0-9][0-9].*\.nc$'
nc_files = util.list_files(url, tag)
file_url = 'https://dods.ndbc.noaa.gov/thredds/fileServer/'
nc_url = [file_url + i + '#mode=bytes' for i in nc_files]

# load datasets
ds = []
for i, f in (enumerate(tq(nc_url, desc='Downloading datasets'))):
    r = requests.get(f, timeout=(3.05, 120))
    # ensure request worked
    if r.ok:
        ds.append(xr.load_dataset(io.BytesIO(r.content)))
        ds[i].load()

# replace bad values
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

# compute wind stress
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", message="invalid value encountered in power")
    warnings.filterwarnings("ignore", message="overflow encountered in exp")
    warnings.filterwarnings("ignore", message="invalid value encountered in log")
    for i, d in enumerate(ds):
        ds[i]['wind_east'], ds[i]['wind_north'] = util.uv_from_spddir(ds[i]['wind_spd'], ds[i]['wind_dir'])
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

# rotate wind velocity and wind stress into principal axis
theta, major, minor = util.princax(east, north)
for i, d in enumerate(ds):
    ds[i]['cs'], ds[i]['as'] = util.rot(d['wind_east'], d['wind_north'], theta)
    ds[i]['coare_x'], ds[i]['coare_y'] = util.rot(d['coare_east'], d['coare_north'], theta)

# add metadata
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
    ds[i].attrs = {'Site Elevation (m)': f'{elev:.02f}',
                   'Air temp height (m)': f'{zt:.02f}',
                   'Anemometer height (m)': f'{zu:.02f}',
                   'Barometer height (m)': f'{zb:.02f}',
                   'Sea temp depth (m)': f'{zt_sea:.02f}',
                   'Water depth (m)': f'{depth:.02f}',
                   'Watch radius (m)': f'{radius:02f}',
                   'Principal axis (deg CW of true north)': f'{theta + 270:.02f}'}

# resample to 1D time intervals with mean method and concatenate datasets
datasets = []
for i, d in enumerate(ds):
    datasets.append(d.resample(time="1D").mean(skipna=True))
ds_bin = xr.concat(datasets, dim='time')

# setup output folders
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(os.path.join(out_path, 'raw')):
    os.mkdir(os.path.join(out_path, 'raw'))
if out_file is None:
    out_file = site + '_wind_binned.nc'
else:
    out_file = out_file + '.nc'

# save output files
ds_bin.to_netcdf(os.path.join(out_path, out_file))
for i, d in enumerate(ds):
    d.to_netcdf(os.path.join(out_path, 'raw', nc_files[i][-13:-3] + '.nc'))
