import numpy as np


def align_yaxis(ax1, v1, ax2, v2):
    """
    Adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1

    Args:
        ax1 (pyplot.axis): left axis
        v1 (scalar): value to align from left axis
        ax2 (pyplot.axis): right axis
        v2 (scalar): value to align from right axis
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


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
    from flox.xarray import xarray_reduce
    import warnings

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
            d[t_lab + '.date'],
            d[z_lab],
            func='nansum',
            expected_groups=(None, z),
            isbin=[False, True],
            method='map-reduce',
            skipna=True,
        )

    time = np.array([np.datetime64(t).astype('datetime64[ns]') for t in out.date.values])
    depth = np.array([x.mid + 0.5 for x in out.depth_bins.values])
    out[z_lab] = ([z_lab + '_bins'], depth)
    out[t_lab] = (['date'], time)
    out = out.swap_dims({z_lab + '_bins': z_lab, 'date': t_lab})
    out = out.drop_vars([z_lab + '_bins', 'date'])

    return out


def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Args:
        dt (array of datetime64): datetimes to convert

    Returns:
        array: calendar array with last axis representing year, month, day, hour,
            minute, second, microsecond
    """
    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970                   # Gregorian Year
    out[..., 1] = (M - Y) + 1                # month
    out[..., 2] = (D - M) + 1                # date
    out[..., 3] = (dt - D).astype("m8[h]")   # hour
    out[..., 4] = (dt - h).astype("m8[m]")   # minute
    out[..., 5] = (dt - m).astype("m8[s]")   # second
    out[..., 6] = (dt - s).astype("m8[us]")  # microsecond
    return out


def find_nearest(array, value):
    if np.all(np.isnan(array)):
        idx = np.nan
    else:
        array = np.asarray(array)
        idx = np.nanargmin((np.abs(array - value)))
    return idx


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Args:
        lon1 (scalar): longitude of first point
        lat1 (scalar): latitude of first point
        lon2 (scalar): longitude of second point
        lat2 (scalar): latitude of second point

    Returns:
        scalar: distance in km between (lon1, lat1) and (lon2, lat2)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371*c
    return km


def list_files(url, tag=r'.*\.nc$'):
    """
    Function to create a list of the netCDF data files in the THREDDS catalog
    created by a request to the M2M system. Obtained from 2022 OOIFB workshop

    Args:
        url (str): URL to a THREDDS catalog specific to a data request
        tag (regexp, optional): Regex pattern used to distinguish files of interest. Defaults to r'.*\\.nc$'.

    Returns:
        array: list of files in the catalog with the URL path set relative to the catalog
    """
    from bs4 import BeautifulSoup
    import re
    import requests

    with requests.session() as s:
        page = s.get(url).text

    soup = BeautifulSoup(page, 'html.parser')
    pattern = re.compile(tag)
    nc_files = [node.get('href') for node in soup.find_all('a', text=pattern)]
    nc_files = [re.sub('catalog.html\\?dataset=', '', file) for file in nc_files]
    return nc_files


def princax(u, v):
    """
    Determines the principal axis of variance for the east and north velocities defined by u and v

    Args:
        u (scalar or array): east velocity
        v (scalar or array): north velocity

    Returns:
        tuple of scalar: (theta, major, minor) - the angle of the principal axis CW from north,
            the variance along the major axis, and the variance along the minor axis
    """
    u = np.array(u)
    v = np.array(v)

    # only use finite values for covariance matrix
    ii = np.isfinite(u+v)
    uf = u[ii]
    vf = v[ii]

    # compute covariance matrix
    C = np.cov(uf, vf)

    # calculate principal axis angle (ET, Equation 4.3.23b)
    # > 0 CCW from east axis, < 0 CW from east axis
    theta = 0.5*np.rad2deg(np.arctan2(2.*C[0, 1], (C[0, 0] - C[1, 1])))
    # switch to > 0 CW from north axis, < 0 CCW from north axis
    if theta >= 0:
        theta = 90 - theta
    elif theta < 0:
        theta = -(90 + theta)

    # calculate variance along major and minor axes (Equation 4.3.24)
    term1 = C[0, 0] + C[1, 1]
    term2 = ((C[0, 0] - C[1, 1])**2 + 4*(C[0, 1]**2))**0.5
    major = np.sqrt(0.5*(term1 + term2))
    minor = np.sqrt(0.5*(term1 - term2))

    return theta, major, minor


def pycno(x, zf, r, h=125):
    """
    Function for an idealized representation of the 25.8 kg/m^3 isopycnal.
    See Austin and Barth, 2002

    Args:
        x (scalar or array): cross-shelf distance in km
        zf (scalar): z intercept of the 25.8 kg/m^3 isopycnal in m
        r (scalar): radius of deformation in km
        h (int, optional): Offshore decay depth of the pycnocline. Defaults to 125.

    Returns:
        scalar or array: cross-shelf depth of the 25.8 kg/m^3 isopycnal
    """
    return -h+(zf+h)*np.exp(x/r)


def rot(u, v, theta):
    """
    Rotates a vector counter clockwise or a coordinate system clockwise
    Designed to be used with theta output from princax(u, v)

    Args:
        u (scalar or array): x-component of vector
        v (scalar or array): y-component of vector
        theta (scalar): rotation angle (CCW > 0, CW < 0)

    Returns:
        tuple of scalar or array: (ur, vr) - x and y components of vector in rotated coordinate system
    """
    w = u + 1j*v
    ang = np.deg2rad(theta)
    wr = w*np.exp(1j*ang)
    ur = np.real(wr)
    vr = np.imag(wr)
    return ur, vr


def uv_from_spddir(spd, dir, which='from'):
    """
    Computes east and west vectors of velocity vector

    Args:
        spd (scalar or array): Velocity magnitude.
        dir (scalar or array): Direction of velocity, CW from true north. Behavior controlled by which.
        which ({"from", "to"}, default: "from"): Determines if dir defines the velocity coming "from" dir
            (common for wind) or going "to" dir (common for currents).

    Returns:
        tuple of scalar or array: (u, v) - east velocity "u" and north velocity "v"
    """
    theta = np.array(dir)
    theta = np.deg2rad(theta)
    if which == 'from':
        u = -spd*np.sin(theta)
        v = -spd*np.cos(theta)
    elif which == 'to':
        u = spd*np.sin(theta)
        v = spd*np.cos(theta)
    else:
        raise ValueError("Invalid argument for 'which'.")
    return (u, v)


def ws_integrand(tp, t, tau, k, rho=1000):
    """
    Integrand for computation of 8-day exponentially weighted integral of 
    wind stress. See Austin and Barth, 2002.

    Args:
        tp (array): integration variable, time
        t (scalar): upper limit of integration, time
        tau (array): wind stress array with same lenth as times tp
        k (scalar): relaxation timescale, same units as time
        rho (scalar, optional): Density of sea water. Defaults to 1000.

    Returns:
        array: integrand for use in scipy.integrate and computation of W8d
    """
    return tau[:t+1]/rho*np.exp((tp[:t+1]-t)/k)


def relative_humidity_from_dewpoint(t, t_dew):
    """
    Relative humidity as a function of air temp. and dew point temp.

    Args:
        t (scalar or array): air temperature (degC)
        t_dew (scalar or array): dew point temperature (degC)

    Returns:
        scalar or array: relative humidity as a percent (0->100)
    """
    e = 610.94*np.exp(17.625*t_dew/(t_dew+243.04))
    es = 610.94*np.exp(17.625*t/(t+243.04))
    rh = e/es*100
    return rh
