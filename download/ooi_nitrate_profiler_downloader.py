"""
Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save.

"""
# Python Standard Library
import argparse
import io
import os
import re
import warnings

# External Modules
from bs4 import BeautifulSoup
import gsw
from joblib import Parallel, delayed
import numpy as np
import requests
from scipy.optimize import curve_fit
from tqdm import tqdm as tq
import xarray as xr
from flox.xarray import xarray_reduce


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar
    From https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: int | None = None,
        desc: str | None = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tq(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def list_files(url, tag=r'.*\.nc$') -> list[str]:
    """
    Function to create a list of the netCDF data files in the THREDDS catalog
    created by a request to the M2M system. Obtained from 2022 OOIFB workshop

    Args:
        url (str): URL to a THREDDS catalog specific to a data request
        tag (regexp, optional): Regex pattern used to distinguish files of interest. Defaults to r'.*\\.nc$'.

    Returns:
        array: list of files in the catalog with the URL path set relative to the catalog
    """
    with requests.session() as s:
        page = s.get(url).text

    soup = BeautifulSoup(page, 'html.parser')
    pattern = re.compile(tag)
    nc_files = [node.get('href') for node in soup.find_all('a', string=pattern)]
    nc_files = [re.sub('catalog.html\\?dataset=', '', file) for file in nc_files]
    return nc_files


def nutnr_qc(ds, rmse_lim=1000) -> xr.Dataset:
    """
    Remove bad fits in OOI nutnr datasets

    Args:
        ds (Dataset): OOI nutnr dataset
        rmse_lim (int, optional): Maximum RMSE for fit to be kept. Defaults to 1000.
    """
    # covariance issues are explicitly handled by checking if pcov is finite
    warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")

    temp = ds.sel({'wavelength': slice(217, 240)})
    mask = np.full(ds.time.shape, True, dtype=bool)
    for i in range(len(temp.time)):
        # remove fits if any values are nan or inf
        if np.any(~np.isfinite(temp.spectral_channels[i] - temp.dark_val[i])):
            mask[i] = False
        # remove anomalously low salinity values
        elif ds.salinity[i] <= 20:
            mask[i] = False
        # remove fits where mean is near zero
        elif (ds.spectral_channels[i].mean() > 1000):
            (a, b), pcov = curve_fit(lambda x, a, b: a*x + b,
                                     temp.wavelength,
                                     temp.spectral_channels[i] - temp.dark_val[i],
                                     p0=[-100, 10000], ftol=0.01, xtol=0.01)
            residuals = temp.spectral_channels[i] - temp.dark_val[i] - temp.wavelength*a - b
            rmse = ((np.sum(residuals**2)/(residuals.size-2))**0.5).values
            # remove fits with high rmse for linear fit in wavelength range
            if rmse > rmse_lim:
                mask[i] = False
            # remove fits with any negative values in wavelength range
            elif np.any(temp.spectral_channels[i] - temp.dark_val[i] < 0):
                mask[i] = False
            # remove fits that did not converge
            elif np.any(~np.isfinite(pcov)):
                mask[i] = False
    ds = ds.where(xr.DataArray(mask, coords={'time': ds.time.values}), drop=True)
    return ds


def split_profiles(ds):
    """
    Split the data set into individual profiles, where each profile is a
    collection of data from a single deployment and profile sequence. The
    resulting data sets are returned in a list.

    :param ds: data set containing the profile data
    :return: a list of data sets, one for each profile
    """
    # split the data into profiles, assuming at least 120 seconds between profiles
    dt = ds.where(ds['time'].diff('time') > np.timedelta64(120, 's'), drop=True).get_index('time')

    # process each profile, adding the results to a list of profiles
    profiles = []
    jback = np.timedelta64(30, 's')  # 30 second jump back to avoid collecting data from the following profile
    for i, d in enumerate(dt):
        # pull out the profile
        if i == 0:
            profile = ds.sel(time=slice(ds['time'].values[0], d - jback))
        else:
            profile = ds.sel(time=slice(dt[i - 1], d - jback))

        # add the profile to the list
        profiles.append(profile)

    # grab the last profile and append it to the list
    profile = ds.sel(time=slice(d, ds['time'].values[-1]))
    profiles.append(profile)
    return profiles


def profiler_binning(d, z, z_lab='depth', t_lab='time', offset=0.5):
    """
    Bins a profiler time series into daily bins and depth bins.
    Removes any non-numeric data types, including any time types,
    outside of the coordinates.

    input:
    d = xr.Dataset with coordinates depth and time
    z = depth bins array
    z_lab, t_lab = labels for depth, time in d

    returns:
    Binned xr.Dataset
    Args:
        d (xr.dataset): OOI profiler dataset
        z (array): edges of depth/pressure bins
        z_lab (str, optional): name of depth/pressure in dataset. Defaults to 'depth'.
        t_lab (str, optional): name of time in dataset. Defaults to 'time'.
        offset (float, optional): Distance from location to CTD (positive when CTD is higher).
            Defaults to 0.5.

    Returns:
        xr.dataset: binned dataset
    """
    types = [d[i].dtype for i in d]
    vars = list(d.keys())
    exclude = []
    for i, t in enumerate(types):
        if not (np.issubdtype(t, np.number)):
            exclude.append(vars[i])
    d = d.drop_vars(exclude)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        out = xarray_reduce(
            d,
            d[t_lab],
            d[z_lab],
            func='nanmean',
            expected_groups=(None, z),
            isbin=[False, True],
            method='map-reduce',
            skipna=True,
        )

    depth = np.array([x.mid + offset for x in out.depth_bins.values])
    out[z_lab] = ([z_lab + '_bins'], depth)
    out = out.swap_dims({z_lab + '_bins': z_lab})
    out = out.drop_vars([z_lab + '_bins'])

    return out


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Download OOI Endurance Array profiler nitrate data, QC fits, bin, and save."
    )
    parser.add_argument('site', metavar='site', type=str, nargs=1,
                        help="site to download from, either CE01ISSP or CE02SHSP")
    parser.add_argument('-p', '--path', metavar='path', type=str, nargs='?',
                        help="path to folder for output files",
                        default='./output/')
    parser.add_argument('-f', '--file', metavar='file', type=str, nargs='?',
                        help="file name for output file, do not include .nc file extension",
                        default=None)
    parser.add_argument('-n', '--njobs', metavar='njobs', type=int, nargs='?',
                        help="number of jobs to run in parallel - see joblib.Parallel docs for more details",
                        default=-2)
    args = parser.parse_args()
    args = vars(args)
    site = args['site'][0].upper()
    out_path = args['path']
    out_file = args['file']
    n_jobs = args['njobs']

    if site == 'CE01ISSP':
        # setup defaults to use in subsequent data queries
        refdes = site + "-SP001-06-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    elif site == 'CE02SHSP':
        refdes = site + "-SP001-05-NUTNRJ000"
        method = "recovered_cspp"
        stream = "nutnr_j_cspp_instrument_recovered"
    else:
        raise ValueError("Invalid site selection, choose one of CE01ISSP (inshore) or CE02SHSP (midshelf).")

    # construct the OOI Gold Copy THREDDS catalog URL for this data set
    base_url = "https://thredds.dataexplorer.oceanobservatories.org/thredds/catalog/ooigoldcopy/public/"
    url = base_url + ('-').join([refdes, method, stream]) + '/catalog.html'
    tag = r'NUTNRJ000.*.nc$'  # setup regex for files we want
    nc_files = list_files(url, tag)
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
        ds[i] = ds[i].rename({
            'ctdpf_j_cspp_instrument_recovered-sea_water_temperature': 'temperature',
            'ctdpf_j_cspp_instrument_recovered-sea_water_practical_salinity': 'salinity',
            'nutnr_dark_value_used_for_fit': 'dark_val'
            })
        ds[i]['abs_sal'] = gsw.conversions.SA_from_SP(
            SP=ds[i]['salinity'],
            p=ds[i]['int_ctd_pressure'],
            lon=ds[i]['lon'],
            lat=ds[i]['lat']
        )
        ds[i]['abs_sal'].attrs = {
            'comment': 'Absolute salinity',
            'units': 'g/kg'
        }
        ds[i]['cons_temp'] = gsw.conversions.CT_from_t(
            SA=ds[i]['abs_sal'],
            t=ds[i]['temperature'],
            p=ds[i]['int_ctd_pressure']
        )
        ds[i]['density'] = gsw.density.sigma0(
            SA=ds[i]['abs_sal'],
            CT=ds[i]['cons_temp']
        )
        ds[i]['density'].attrs = {
            'comment': 'Potential density anomaly referenced to 0 dbar',
            'units': 'kg/m^3'}

    # QC nitrate data and remove short datasets
    ds = ParallelTqdm(n_jobs=n_jobs, desc='QC')(delayed(nutnr_qc)(d) for d in ds)
    mask = [i for i, d in enumerate(ds) if len(d.time) > 10]
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
    deployments = []
    times = []
    for d in tq(ds, desc='Splitting and binning profiles'):
        deployments.append(split_profiles(d))
        profiles = []
        profile_times = []
        for profile in deployments[-1]:
            profile_binned = profiler_binning(profile, pressure_bins, offset=0.5)
            profiles.append(profile_binned.mean(dim='time'))
            profile_times.append(profile_binned.time.mean().values)
        deployments[-1] = profiles
        times.append(xr.DataArray(profile_times, dims='time'))
    deployments = [xr.concat(profiles, profile_times) for profiles, profile_times in zip(deployments, times)]

    # concatenate binned datasets
    ds_bin = xr.concat(deployments, dim='time')
    ds_bin = ds_bin.drop_duplicates('time', keep='first')
    ds_bin = ds_bin.where(~np.isinf(ds_bin.salinity_corrected_nitrate))

    # setup output folders
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(os.path.join(out_path, 'raw')):
        os.mkdir(os.path.join(out_path, 'raw'))
    if not os.path.exists(os.path.join(out_path, 'deps')):
        os.mkdir(os.path.join(out_path, 'deps'))
    if out_file is None:
        out_file = site + '_nitrate_binned.nc'
    else:
        out_file = out_file + '.nc'

    # save output files
    ds_bin.to_netcdf(os.path.join(out_path, out_file))
    ds_bin.close()
    for d in ds:
        d.to_netcdf(os.path.join(out_path, f'raw/dep{d.deployment[0]:02.0f}_' + d.attrs['source'] + '.nc'), mode='w')
    for d in deployments:
        deployment = np.unique(d.deployment)[0]
        d.to_netcdf(os.path.join(out_path, f'deps/dep{deployment:02.0f}_binned.nc'), mode='w')
