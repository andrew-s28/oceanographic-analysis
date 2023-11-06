"""
Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save.
"""
import argparse
import gsw
import io
import numpy as np
import os
import requests
from tqdm import tqdm as tq
from ocean_data_utilities import util
import xarray as xr

# parse command line arguments
parser = argparse.ArgumentParser(description="Download OOI Endurance Array profiler nitrate data,"
                                 + " QC fits, bin, and save.")
parser.add_argument('site', metavar='site', type=str, nargs=1,
                    help="site to download from")
parser.add_argument('-p', '--path', metavar='path', type=str, nargs='?',
                    help="path to folder for output files",
                    default='./output/')
parser.add_argument('-f', '--file', metavar='file', type=str, nargs='?',
                    help="file name for output file, do not include .nc",
                    default=None)
args = parser.parse_args()
args = vars(args)
site = args['site'][0].upper()
out_path = args['path']
out_file = args['file']

# setup defaults to use in subsequent data queries
refdes = site + "-SP001-06-NUTNRJ000"
method = "recovered_cspp"
stream = "nutnr_j_cspp_instrument_recovered"

# construct the OOI Gold Copy THREDDS catalog URL for this data set
base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
url = base_url + ('-').join([refdes, method, stream]) + '/catalog.html'
tag = r'NUTNRJ000.*.nc$'  # setup regex for files we want
nc_files = util.list_files(url, tag)
base_url = 'https://thredds.dataexplorer.oceanobservatories.org/thredds/fileServer/'
nc_url = [base_url + i + '#mode=bytes' for i in nc_files]  # create urls for download

# load datasets
ds = []
for i, f in (enumerate(tq(nc_url, desc='Downloading datasets'))):
    r = requests.get(f, timeout=(3.05, 120))
    if r.ok:
        ds.append(xr.load_dataset(io.BytesIO(r.content)))
        ds[i].load()

# some renaming and new variables
for i, d in enumerate(ds):
    ds[i] = ds[i].swap_dims({'obs': 'time'})
    ds[i] = ds[i].rename({'ctdpf_j_cspp_instrument_recovered-sea_water_temperature': 'temperature',
                          'ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity': 'salinity',
                          'nutnr_dark_value_used_for_fit': 'dark_val'})
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

# QC nitrate data and remove short datasets
mask = []
for i, d in enumerate(tq(ds, desc='QC')):
    ds[i] = util.nutnr_qc(ds[i])
    if len(ds[i].time) > 10:
        mask.append(i)
ds = [ds[i] for i in mask]

# find minimum and maximum depth bins over all nitrate data
sur = np.min(xr.concat(ds, dim='time')['depth'])
bot = np.max(xr.concat(ds, dim='time')['depth'])

# setup depth/pressure bins
step = 1
sur = np.floor(sur)
bot = np.ceil(bot)
# pressure_grid is centers of bins
pressure_grid = np.arange(sur+step/2, bot+step, step)
# pressure_bins is edges of bins
pressure_bins = np.nan*np.empty(len(pressure_grid)+1)
pressure_bins[0] = pressure_grid[0] - step/2
pressure_bins[-1] = pressure_grid[-1] + step/2
for i in range(len(pressure_bins)-2):
    pressure_bins[i+1] = np.average([pressure_grid[i], pressure_grid[i+1]])

# bin datasets into new list
datasets = []
for d in tq(ds, desc='Binning datasets'):
    datasets.append(util.profiler_binning(d, pressure_bins, offset=0.5))

# concatenate binned datasets
ds_bin = xr.concat(datasets, dim='time')
ds_bin = ds_bin.drop_duplicates('time', 'first')
ds_bin = ds_bin.where(~np.isinf(ds_bin.salinity_corrected_nitrate))

# setup output folders
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(os.path.join(out_path, 'raw')):
    os.mkdir(os.path.join(out_path, 'raw'))
if out_file is None:
    out_file = site + '_nitrate_binned.nc'
else:
    out_file = out_file + '.nc'

# save output files
ds_bin.to_netcdf(os.path.join(out_path, out_file))
for d in ds:
    d.to_netcdf(os.path.join(out_path, f'raw/dep{d.deployment[0]:02.0f}_' + d.attrs['source'] + '.nc'))
