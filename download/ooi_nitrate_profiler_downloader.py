# input/output, html parsing, requests, and os functionality
import io
import os
import requests
from tqdm import tqdm as tq
import numpy as np
import xarray as xr
import gsw
import util

# get nitrate data
print("Oregon Inshore - CE01ISSP")
print("Oregon Shelf   - CE02SHSP")
print("Enter site for nitrate: ", end='')
site = input()
# setup defaults to use in subsequent data queries
refdes = site + "-SP001-06-NUTNRJ000"
method = "recovered_cspp"
stream = "nutnr_j_cspp_instrument_recovered"

# construct the OOI Gold Copy THREDDS catalog URL for this data set
base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
url = base_url + ('-').join([refdes, method, stream]) + '/catalog.html'

tag = r'NUTNRJ000.*.nc$'  # setup regex for files we want (*=anything,$=end of line)
nc_files = util.list_files(url, tag)
file_url = 'https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/'
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
    ds[i] = ds[i].swap_dims({'obs': 'time'})
    ds[i] = ds[i].rename({'ctdpf_j_cspp_instrument_recovered-sea_water_temperature': 'temperature',
                          'ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity': 'salinity'})
    ds[i]['abs_sal'] = gsw.conversions.SA_from_SP(ds[i]['salinity'], ds[i]['int_ctd_pressure'],
                                                  ds[i]['lon'], ds[i]['lat'])
    ds[i]['abs_sal'].attrs = {'comment': 'Absolute salinity calculated from pressure and practical salinity.',
                              'units': 'g/kg'}
    ds[i]['density'] = gsw.density.rho_t_exact(ds[i]['abs_sal'], ds[i]['temperature'],
                                               ds[i]['int_ctd_pressure'])
    ds[i]['density'].attrs = {'comment': 'In-situ density from salinity, temperature, and pressure.',
                              'units': 'kg/m^3'}
    ds[i]['density_anom'] = ds[i]['density'] - 1000
    ds[i]['density_anom'].attrs = {'comment': 'Density anomaly - density - 1000 kg/m^3',
                                   'units': 'kg/m^3'}

# find minimum and maximum depth bins over all nitrate data
sur = np.min(xr.concat(ds, dim='time')['depth'])
bot = np.max(xr.concat(ds, dim='time')['depth'])

step = 1
sur = np.floor(sur)
bot = np.ceil(bot)
# pressure_grid = centers of bins
pressure_grid = np.arange(sur+step/2, bot+step, step)
# pressure_bins = edges of bins
pressure_bins = np.nan*np.empty(len(pressure_grid)+1)
pressure_bins[0] = pressure_grid[0] - step/2
pressure_bins[-1] = pressure_grid[-1] + step/2
for i in range(len(pressure_bins)-2):
    pressure_bins[i+1] = np.average([pressure_grid[i], pressure_grid[i+1]])

# create regular time bins based on nitrate time range
# need to cast to daily data to round down then recast to ns for consistency
begin = ds[0]['time'].values[0].astype('datetime64[D]').astype('datetime64[ns]')
# same as above, but add one extra day since arange is endpoint exclusive
end = (ds[-1]['time'].values[-1] + np.timedelta64(1, 'D')).astype('datetime64[D]').astype('datetime64[ns]')
time_grid = np.arange(begin, end, step=np.timedelta64(1, 'D'), dtype='datetime64')

nitrate_grid = np.nan*np.empty([len(pressure_grid), len(time_grid)])
density_grid = np.nan*np.empty([len(pressure_grid), len(time_grid)])
sal_grid = np.nan*np.empty([len(pressure_grid), len(time_grid)])
temp_grid = np.nan*np.empty([len(pressure_grid), len(time_grid)])

datasets = []

for d in tq(ds, desc='Binning datasets'):
    datasets.append(util.profiler_binning(d, pressure_bins, offset=0.5))

ds_bin = xr.concat(datasets, dim='time')
ds_bin = ds_bin.drop_duplicates('time', 'first')

print("Enter name for output nc file: ", end='')
out_name = input()

if not os.path.exists('./output/'):
    os.mkdir('./output/')

ds_bin.to_netcdf('./output/' + out_name + '.nc')

for d in ds:
    d.to_netcdf(f'./output/dep{d.deployment[0]:02d}_' + d.attrs['source'] + '.nc')
