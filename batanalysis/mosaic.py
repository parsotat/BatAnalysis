"""
This file is meant to hold the functions that allow users to create mosaic-ed images for survey data
"""

import shutil
import h5py
from pathlib import Path

import numpy as np
import pkg_resources
import scipy.spatial.qhull as qhull
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from reproject import reproject_to_healpix, reproject_from_healpix
from astropy.table import Table
from joblib import Parallel, delayed
import healpy as hp

from .bat_survey import MosaicBatSurvey, BatSurvey
from .batlib import dirtest, met2utc
from .skyproject import ReprojectSurvey, MosaicSummary

# for python>3.6
try:
    import heasoftpy.swift as hsp
    import heasoftpy.utils as hsp_util
    import heasoftpy
except ModuleNotFoundError as err:
    # Error handling
    print(err)

import swiftbat.swutil as sbu

# Off-axis flux correction file
_cimgfile = "offaxiscorr_8bin_20061221.img"
_chilothresh = 0.50  # Minimum chi-square for any energy band
_chihithresh = 1.15  # Maximum chi-square for any energy band
_chiscobump = 0.35  # Additional bump of chi-square threshold around Sco X-1 (band 0)
_chiscotheta = 30  # Approximate angular scale of bump around Sco X-1 (deg)
_pcodethresh = 0.01  # Minimum image partial coding
_minexpo = 150  # Minimum image exposure
_nskyimg = 6  # Number of facets to sky image
_nebands = 8  # Number of energy bands to process
_proj = "ZEA"  # projection from idl code that is used
_order = 10  # Healpix order parameter
_nside = 2**_order  # Healpix nside parameter

# also information to create the skygrids if the user wants
_gcenters = np.array(
    [
        [90, 0],  # Galactic equatorial belt 1
        [0, 0],  # Galactic equatorial belt 2
        [-90, 0],  # Galactic equatorial belt 3
        [180, 0],  # Galactic equatorial belt 4
        [0, 90],  # North galactic polar cap
        [0, -90],  # South galactic polar cap
    ],
    dtype="float64",
)

# Sco X-1 ra and dec
_scox1_ra = 245.100
_scox1_dec = -15.600
_sco_coord = SkyCoord(_scox1_ra, _scox1_dec, frame="icrs", unit="deg")


def interp_weights(xyz, uvw, d=2):
    """
    This is a function to calculate the weights for each vertex on a grid that will be interpolated
    over. See https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    :param xyz: The x,y,z points that will be interpolated over on the new grid
    :param uvw: The new grid that we will interpolation to be conducted over
    :param d: dimension of the grid, default is 2D grids
    :return: returns the vertices of the interpolation funciton and the weights at each vertex
    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum("njk,nk->nj", temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, fill_value=np.nan):
    """
    Function that conducts the interpolation for a set of values at points x,y,z and interpolates their corresponding
    values on a new grid that was passed into the interp_weights function to get the correponsing verticies and weights.

    :param values: The values that will be interpolated over at the points of the xyz grid (see interp_weights)
    :param vtx: The verticies obtained from the interp_weights function
    :param wts: The weights obtained from the interp_weights function
    :param fill_value: The default fill value for interpolation points outside the grid of interest (where points have
        to be extrapolated)
    :return: Returns the interpolated values at the uvw points (see interp_weights)
    """
    ret = np.einsum("nj,nj->n", np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


def convert_time_to_decimal_day(times):
    """
    Converts an array of astropy Time objects to decimal days

    :param times: an array of astropy Time objects
    :return: an array of decimal days corresponding to the input times
    """
    # get the ymdhms values for each time object
    # First decode the object shape
    nelements = len(times.shape)
    if nelements == 0:
        times = Time([times])
    ymdhms = times.ymdhms

    # calculate the decimal day values
    year = ymdhms["year"].astype(str)
    month = [str(i).zfill(2) for i in ymdhms["month"]]
    # day = [str(i).zfill(2) for i in ymdhms["day"]]
    decimal_day = np.round(
        ymdhms["day"]
        + ymdhms["hour"] / 24
        + ymdhms["minute"] / 1440
        + ymdhms["second"] / 86400,
        3,
    ).astype(str)
    output_str = [f"{y}_{m}_{d}" for y, m, d in zip(year, month, decimal_day)]
    if nelements == 0:
        return output_str[0]
    else:
        return output_str


def make_skygrids(
    center_resolution=2.8, galactic_boundaries=[48, 48], savedirectory=None
):
    """
    Creates a skygrid based on galactic coordinates being split up into 6 facets. The skygrids are created using the
    Zenith Equal Angle projection.

    These facets are given by (galatic latitude, galactic longitude) coordinates being at the center of each facet.
    The coordinates are:
    gcenters=np.array([[90,0], #Galactic equatorial belt 1
          [0,0], #Galactic equatorial belt 2
          [-90,0], #Galactic equatorial belt 3
          [180,0], #Galactic equatorial belt 4
          [0,90], #North galactic polar cap
          [0,-90]], #South galactic polar cap
          )

    The coordinates are converted to RA/DEC and are used to create the mosaiced images.

    :param center_resolution: The angular resolution of the central pixel of the sky facets.
        The value must be in units of arcminutes.
    :param galactic_boundaries: The maximum size of each facet for the galactic latitude and longitude coordinates.
        The units for this parameter is in degrees
    :param savedirectory: Default None or Pathlib object that points to a directory that the skygrids will be saved to.
        Default of None uses the default BATAnalysis data directory with the package.
    :return: None
    """

    deg_resolution = center_resolution / 60  # pixel spacing
    nimages = len(_gcenters)  # number of sky facets to make
    thxmax = galactic_boundaries[0]
    thymax = galactic_boundaries[1]

    # set up the save directory
    if savedirectory is None:
        savedirectory = Path(__file__).parent.joinpath("data")
    else:
        savedirectory = Path(savedirectory)

    if "ZEA" in _proj:
        p_dth = 2 * np.rad2deg(
            np.sin(np.deg2rad(deg_resolution / 2))
        )  # projected pixel spacing, includes 360/pi factor in front
        p_thxmax = 2 * np.rad2deg(
            np.sin(np.deg2rad(thxmax / 2))
        )  # Projected X boundary
        p_thymax = 2 * np.rad2deg(
            np.sin(np.deg2rad(thymax / 2))
        )  # Projected Y boundary
    elif "TAN" in _proj:
        p_dth = np.tan(np.deg2rad(deg_resolution))
        p_thxmax = np.tan(np.deg2rad(thxmax))
        p_thymax = np.tan(np.deg2rad(thymax))

    # get the size of the x,y grid
    naxis1 = int(np.ceil(2 * p_thxmax / p_dth))
    naxis2 = int(np.ceil(2 * p_thymax / p_dth))

    for i in range(nimages):
        # get the facet corresponding to the galactic coordinates of interest being at the center
        gl = _gcenters[i, 0]
        gb = _gcenters[i, 1]

        # save these to be used as header keywords
        crval1 = gl
        crval2 = gb
        crpix1 = naxis1 / 2 + 1
        crpix2 = naxis2 / 2 + 1
        cdelt1 = -deg_resolution
        cdelt2 = +deg_resolution

        # create the pixel arrays and the grids
        x = np.arange(naxis1)
        y = np.arange(naxis2)
        xx, yy = np.meshgrid(x, y)

        header = fits.Header()
        header["NAXIS"] = (2, " number of data axes")
        header["NAXIS1"] = (naxis1, " length of data axis 1")
        header["NAXIS2"] = (naxis2, " length of data axis 2")
        header["BITPIX"] = (-32, " number of bits per data pixel")
        header["CTYPE1"] = ("GLON-" + _proj, " Name of coordinate")
        header["CTYPE2"] = ("GLAT-" + _proj, " Name of coordinate")
        header["CUNIT1"] = ("deg", " Units of coordinate axis")
        header["CUNIT2"] = ("deg", " Units of coordinate axis")
        header["CRPIX1"] = (crpix1, " Reference pixel position")
        header["CRPIX2"] = (crpix2, " Reference pixel position")
        header["CRVAL1"] = (crval1, " Coordinate value at reference pixel position")
        header["CRVAL2"] = (crval2, " Coordinate value at reference pixel position")
        header["CDELT1"] = (cdelt1, " Pixel spacing in physical units")
        header["CDELT2"] = (cdelt2, " Pixel spacing in physical units")
        header["LONPOLE"] = (180.0, " Longitude of native pole")

        # convert coordinates to glat/glon and then to ra/dec
        ra, dec = convert_xy2radec(xx, yy, header)

        for coord, val in zip(["ra", "dec"], [ra, dec]):
            file = f"{coord}_c{i}_{_proj}.img"
            save_file = savedirectory.joinpath(file)
            fits.writeto(save_file, val, header)

    return 0


def make_all_sky_stg_grids(resolution=2.8, savedirectory=None):
    """Helper function to save WCS to a FITS file. This 2D sky projection uses the
    Stereographic (STG) projection in Galactic coordinates. The tile centers are
    hardcoded for an 8-tile configuration covering the sky. The centers are at
    (l, b) = (0°, 0°), (90°, 0°), (180°, 0°), (270°, 0°), (45°, 50°), (135°, 50°),
    (225°, -50°), and (315°, -50°).

    Parameters
    ----------
    resolution : float
        Pixel scale in arcminutes per pixel. Default is 2.8 arcmin/pixel.
    savedirectory : str
        Directory to which the output FITS filename is written.
    """
    # Format: (longitude, latitude) in Galactic coordinates
    TILE_CENTERS = [
        # 4 equatorial tiles (full longitude coverage)
        (0, 0),
        (90, 0),
        (180, 0),
        (270, 0),
        # 2 polar caps
        (0, 90),
        (0, -90),
    ]

    # Configuration
    PIXEL_SCALE = resolution / 60  # degrees per pixel (2.8 arcmin)
    TILE_SIZE = 95.0  # degrees
    NPIX = int(TILE_SIZE / PIXEL_SCALE)  # ~2000 pixels

    # set up the save directory
    if savedirectory is None:
        savedirectory = Path(__file__).parent.joinpath("data")
    else:
        savedirectory = Path(savedirectory)

    all_sky_wcs_grids = {}

    for i, (l, b) in enumerate(TILE_CENTERS):
        # Create WCS
        # Save to FITS file
        # Print summary
        filename = f"sky_grid_{i+1:02d}_l_{l:03.0f}_b_{b:+03.0f}_STG.fits"

        """Create a WCS object for a tile centered at (l_center, b_center)."""
        """Save WCS to a FITS header file."""

        # Create a minimal FITS HDU with the WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crval = [l, b]  # Reference sky position
        wcs.wcs.crpix = [NPIX / 2 + 0.5, NPIX / 2 + 0.5]  # Reference pixel
        wcs.wcs.cdelt = [-PIXEL_SCALE, PIXEL_SCALE]  # Pixel scale
        wcs.wcs.ctype = ["GLON-STG", "GLAT-STG"]  # Stereographic projection
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.pixel_shape = (NPIX, NPIX)

        # save them
        all_sky_wcs_grids[filename] = wcs
    return all_sky_wcs_grids


def make_healpix_grids():
    """Helper function that creates the healpix grids that will be used for mosaicing.
    By defualt it uses _nside parameter to decide the resolution of the grid. The morale is to
    project the image onto a healpix map so that storage can be efficient
    """
    pix_ids = np.arange(hp.nside2npix(_nside)).astype(int)
    theta, phi = hp.pix2ang(_nside, pix_ids)
    ra = np.rad2deg(phi)
    dec = 90 - np.rad2deg(theta)
    return ra, dec


def divide_with_respect(a, b):
    """Helper function that divides two arrays but takes care of divide by zero errors

    Args:
        a (np.ndarray): The numerator array
        b (np.ndarray): The denominator array

    Returns:
        np.ndarray: The result of the division with divide by zero handled
    """
    result = np.divide(a, b, out=np.zeros_like(b), where=b != 0)
    return result


def get_common_healpix_grid(hdr):
    """By defualt the healpiz grid is all sky, but we dont want that. We want what is common
    to the image, so given a header file, make a WCS and get the common area.

    Args:
        hdr (astropy.io.fits.header.Header): The header file
    """
    wcs = WCS(hdr)
    ra_hp, dec_hp = make_healpix_grids()
    coords = SkyCoord(ra_hp, dec_hp, unit="deg")
    mask = wcs.footprint_contains(coord=coords)
    commom_coords = coords[mask]

    # get common pixels
    phi = commom_coords.ra.rad
    theta = np.pi / 2 - commom_coords.dec.rad
    pix_ids = hp.ang2pix(_nside, theta, phi)

    # Instantiate values
    vals = np.zeros(len(ra_hp))
    return commom_coords, pix_ids, vals


def convert_time_to_decimal_day(times):
    """
    Converts an array of astropy Time objects to decimal days

    :param times: an array of astropy Time objects
    :return: an array of decimal days corresponding to the input times
    """
    # get the ymdhms values for each time object
    # First decode the object shape
    nelements = len(times.shape)
    if nelements == 0:
        times = Time([times])
    ymdhms = times.ymdhms

    # calculate the decimal day values
    year = ymdhms["year"].astype(str)
    month = [str(i).zfill(2) for i in ymdhms["month"]]
    # day = [str(i).zfill(2) for i in ymdhms["day"]]
    decimal_day = np.round(
        ymdhms["day"]
        + ymdhms["hour"] / 24
        + ymdhms["minute"] / 1440
        + ymdhms["second"] / 86400,
        3,
    ).astype(str)
    output_str = [f"{y}_{m}_{d}" for y, m, d in zip(year, month, decimal_day)]
    if nelements == 0:
        return output_str[0]
    else:
        return output_str


def merge_outventory(survey_list, savedir=None):
    """
    Creates a merged outventory file in the savedir parameter which lists all the BAT surveys that will be
    combined into the mosaiced image.

    :param survey_list: a list of BATSurvey objects with observations that will eventually be combined into mosaiced
        images
    :param savedir: Default None or a Path object that points to a directory where the merged outventory file will be
        saved. This is also the directory where the mosaiced images will be saved.
    :return: A pathlib object of the created outventory file
    """

    if type(survey_list) is not list:
        raise ValueError("The input needs to be a list of BatSurvey objects.")

    if savedir is None:
        savedir = survey_list[0].result_dir.parent.joinpath("mosaiced_surveyresults")
    else:
        savedir = Path(savedir)
    dirtest(savedir, clean_dir=False)

    # Below IS HEASOFT STUFF
    # need to find all the statfiles.lis and merge them and sort by time
    # input_files = ""
    # for obs in survey_list:
    #    input_files += f'{obs.result_dir.joinpath("stats_point.fits")},'
    # input_files = input_files[:-1]  # get rid of last comma

    # create the pfile directory
    # local_pfile_dir = savedir.joinpath(".local_pfile")
    # local_pfile_dir.mkdir(parents=True, exist_ok=True)
    # try:
    #    hsp.local_pfiles(pfiles_dir=str(local_pfile_dir))
    # except AttributeError:
    #    hsp_util.local_pfiles(par_dir=str(local_pfile_dir))

    # output_file = savedir.joinpath(
    #    "outventory_all_unsrt.fits"
    # )  # os.path.join(savedir, "outventory_all_unsrt.fits")

    # input_filename = savedir.joinpath(
    #    "input_files.txt"
    # )  # os.path.join(savedir, "input_files.txt")
    # write input files to a text file for convience
    # with open(str(input_filename), "w") as text_file:
    #    text_file.write(input_files.replace(",", "\n"))

    # input_dict = dict(
    #    infile="@" + str(input_filename), outfile=str(output_file), clobber="YES"
    # )

    # merge files
    # hsp.ftmerge(**input_dict)

    # outventory_file = str(output_file).replace("_unsrt", "")
    # input_dict = dict(
    #    infile=str(output_file),
    #    outfile=outventory_file,
    #    columns="TSTART",
    #    clobber="YES",
    # )

    # sort file by time
    # hsp.ftmergesort(**input_dict)

    # get rid of the unsorted file
    # output_file.unlink()
    # input_filename.unlink()
    # Above IS HEASOFT STUFF

    output_file = savedir.joinpath("outventory_all.fits")

    shutil.copy(survey_list[0].result_dir.joinpath("stats_point.fits"), output_file)
    for i in survey_list[1:]:
        with fits.open(output_file) as hdul1:
            with fits.open(i.result_dir.joinpath("stats_point.fits")) as hdul2:
                nrows1 = hdul1[1].data.shape[0]
                nrows2 = hdul2[1].data.shape[0]
                nrows = nrows1 + nrows2
                hdu = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
                for colname in hdul1[1].columns.names:
                    hdu.data[colname][nrows1:] = hdul2[1].data[colname]
                hdu.writeto(output_file, overwrite=True)

    # now sort the file by time
    with fits.open(output_file, mode="update") as hdul:
        hdu = hdul[1]
        idx = np.argsort(hdu.data["TSTART"])
        hdu.data = hdu.data[idx]
        hdul.flush()

    return Path(output_file)


def select_outventory(outventory_file, start_met, end_met):
    """
    Function that selects observations listed in a given outventory file based on a start and end time
    in MET units. This function produces a fits file with the observations that fall between the start and end times.

    :param outventory_file: a Path object that points to the outventory file with all the observations of interest.
    :param start_met: The start time in MET to select observations from the outventory file. This can be an array of
        start bin edges.
    :param end_met: The end time in MET to select observations from the outventory file. This can be an array of
        start bin edges.
    :return: None
    """

    # replace the ftselect with astropy fits operations
    output_file = str(outventory_file).replace(".fits", "_sel.fits")
    with fits.open(outventory_file) as f:
        # identify the lines with the times of interest/images that are good and have the comparison be broadcastable
        idx = np.where(
            (f[1].data["TSTART"][..., None] >= start_met)
            & (f[1].data["TSTART"][..., None] < end_met)
            & (f[1].data["IMAGE_STATUS"][..., None] == True)
        )

        # save the headers of the original outventory file and then copy them to the new output file
        hdu = f[1]
        hdu.data = hdu.data[idx[0]]
        hdu.writeto(output_file)


def group_outventory(
    outventory_file,
    binning_timedelta=None,
    start_datetime=None,
    end_datetime=None,
    recalc=False,
    custom_timebins=None,
    subdirs=False,
    save_group_outventory=True,
):
    """
    This function groups the observations listed in an outventory file together based on time bins that each observation
    may fall within. this function creates a "grouped_outventory" directory in the folder that the outventory folder will

    :param outventory_file: a Path object that points to the outventory file with all the observations of interest.
    :param binning_timedelta:  a numpy delta64 object that denotes the width of each time bin of interest. In typical
        BAT survey papers this is a month but different time bins can be used.
    :param start_datetime: An astropy Time object that denotes the start date to start binning the observations
    :param end_datetime: An astropy Time object that denotes the end date to stop binning the observations
    :param recalc: Boolean to denote if the directoruy at is created or not. Also denotes if the
    :param custom_timebins: None OR
        an array of astropy Time values denoting the timebin edges for which mosaicing will take place.
            ie if custom_timebins=astropy.Time(["2022-10-08","2022-10-10", "2022-10-12"]) then there will be 2 grouped
            outventory files (and thus mosaics) will be created.
        OR a list of N astropy Time arrays each with shape (2 x T) where N grouped outventory files will be created
        (and thus mosaics will be created) where the selected observation times correspond to the time bin edges
        denoted by the single (2 x T) array. The astropy Time array should be formatted such that row 0
        (indexed as [0,:]) contains all the start times of the edges of the timebins of interest which will be mosaiced
        together. The row 1 of the astropy Time array (indexed as [1,:]) contains all the end times of the edges of the
        timebins of interest which will be mosaiced together.
            ie if tbins=[
                Time([["2022-10-08","2022-10-11"],
                    ["2022-10-10", "2022-10-12"]]),
                Time([["2022-10-10"],
                    ["2022-10-11"]])]

                then there will be 2 mosaics created. Mosaic 1 will combine observations from 2022-10-08 to 2022-10-10
                AND observations from 2022-10-11 to 2022-10-12.
                While mosaic 2 will combine observations from 2022-10-10 to 022-10-11.
    :param subdirs: a Boolean that denotes whether each time bin will have its own subdirectory created to hold the
        grouped outventory file and eventually the mosaiced results. Deafults to False. This would mean that all 1 day mosaics
        will be saved under a folder called 1_day_mosaics
    :param save_group_outventory: a Boolean that denotes whether the grouped outventory files for each time bin and the
        associated directories to hold the mosaic results for the time bins will be created. If this is set to False,
        these will not be created but the calculated time_bins will be returned
    :return: astropy Time array of the time bin edges that are created based on the user specification. This can be
        passed directly to the create_mosaic function.
    """

    # need to group the observations based on the time binning that the user wants this is given by binning_timedelta
    # we can start from the first entry of the time sorted outventory file (or the start datetime that the user
    # specifies) and go until the end of the last observation of the outventory file (or the end datetime that the
    # user specifies)

    # error checking

    # if the  custom_timebins variable is set to none (the default) then we defualt to using the start/end_datetimes
    # so need to do input checks here also set this time_bins_is_list switch to false by default
    time_bins_is_list = False
    if custom_timebins is None:
        if type(binning_timedelta) is not np.timedelta64:
            raise ValueError(
                "The binning_timedelta variable needs to be a numpy timedelta64 object."
            )

        if start_datetime is not None:
            if type(start_datetime) is not Time:
                raise ValueError(
                    "The start_datetime variable needs to be an astropy Time object."
                )

        if end_datetime is not None:
            if type(end_datetime) is not Time:
                raise ValueError(
                    "The end_datetime variable needs to be an astropy Time object."
                )

    else:
        if type(custom_timebins) is not Time and type(custom_timebins) is not list:
            raise ValueError(
                "The custom_timebins variable needs to be an astropy Time object or a list of astropy Time objects."
            )
        # make sure that all elements of list are astropy time objects and set a switch for later processing
        if type(custom_timebins) is list:
            for i in custom_timebins:
                if type(i) is not Time:
                    raise ValueError(
                        "All the list elements of the  custom_timebins variable needs to be an astropy Time object of \
                        dimension 2 x T, where T is the number of timebins of interest."
                    )
            time_bins_is_list = True

    # initalize the reference time for the Swift MET time (starts from 2001), used to calculate MET
    reference_time = Time("2001-01-01")

    # make sure its a path object
    outventory_file = Path(outventory_file)

    # if we dont have the actual time bins passed in we need to calcualte them
    if custom_timebins is None:
        # by default use the earliest date of outventory file
        if start_datetime is None:
            # use the swift launch date
            # launch_time = Time("2004-12-01")
            # start_datetime=launch_time
            with fits.open(str(outventory_file)) as file:
                t = [
                    Time(i, format="isot", scale="utc")
                    for i in file[1].data["DATE_OBS"]
                ]

            # get the min date and get the ymdhms to modify
            t = Time(t).min()
            tholder = t.min().ymdhms

            # get the date to start at the beginning of the day
            for i in range(3, len(tholder)):
                tholder[i] = 0
            start_datetime = Time(tholder)

        # by default use the last entry of the outventory_file rounded to the nearest timedelta that the user is
        # interested in
        if end_datetime is None:
            with fits.open(outventory_file) as f:
                end_datetime = Time(met2utc(f[1].data["TSTART"].max()))

        if np.dtype(binning_timedelta) == np.dtype(
            np.timedelta64(1, "M")
        ) and binning_timedelta == np.timedelta64(1, "M"):
            # if the user wants months, need to specify each year, month and the number of days
            years = [
                i
                for i in range(
                    start_datetime.ymdhms["year"], end_datetime.ymdhms["year"] + 1
                )
            ]
            months = np.arange(1, 13)
            months_list = []
            for y in years:
                for m in months:
                    if start_datetime.ymdhms["year"] == end_datetime.ymdhms["year"]:
                        if (
                            m >= start_datetime.ymdhms["month"]
                            and m <= end_datetime.ymdhms["month"] + 1
                            and y == start_datetime.ymdhms["year"]
                        ):
                            months_list.append("%d-%02d" % (y, m))
                    else:
                        if (
                            m >= start_datetime.ymdhms["month"]
                            and y == start_datetime.ymdhms["year"]
                        ):
                            months_list.append("%d-%02d" % (y, m))
                        elif (
                            m <= end_datetime.ymdhms["month"] + 1
                            and y == end_datetime.ymdhms["year"]
                        ):
                            months_list.append("%d-%02d" % (y, m))  # for the edge case
                        elif (
                            y > start_datetime.ymdhms["year"]
                            and y < end_datetime.ymdhms["year"]
                        ):
                            months_list.append("%d-%02d" % (y, m))

            # days_per_month = [31, febdays, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            time_bins = np.array(
                months_list, dtype="datetime64[M]"
            )  # create array with the months of interest
        else:
            time_bins = np.arange(
                start_datetime.datetime64,
                end_datetime.datetime64 + binning_timedelta,
                binning_timedelta,
            )

        # convert to astropy time objects
        time_bins = Time(time_bins)
    else:
        # need to convert the  custom_timebins to the time_bins format which is trivial since they are already set to
        # be that
        time_bins = custom_timebins

    # need to see if time_bins is a 1D Time array or a list of size N where there are N arrays of dimension 2xT where
    # there are T time bins of interest that will be combined into a grouped outventory file. The index 0 of the T
    # times should be the start of the time bin(s) of interest while the index 1 of the T times should be the end of
    # the time bins

    # if we want to create the timebin mosaic directories and the associated group outventory do so, otherwise just
    # return the time_bins for the user to check them
    if save_group_outventory:
        # create the folder that will hold the outventory files for each time range of interest
        savedir = outventory_file.parent.joinpath("grouped_outventory")

        # see if the savedir exists, if it does, then we dont have to do all of these calculations again
        if not savedir.exists() or recalc:
            # clear the directory
            dirtest(savedir)

            # get the number of iterations we need to do in teh loop below
            if not time_bins_is_list:
                # this is to account for the fact that we have the array consisting of the start/end edges all in one
                loop_iters = len(time_bins) - 1
            else:
                # this accounts for having the list of just the starting bin edges or the end bin edges
                loop_iters = len(time_bins)

            # loop over time bins to select the appropriate outventory enteries
            # And add in placeholder directories
            # if subdir=True, make a global subdirectory to hold all mosaics
            if subdirs:
                subdir_dt = np.round(
                    TimeDelta(
                        binning_timedelta.astype("timedelta64[s]"), format="sec"
                    ).jd,
                    1,
                )
                if subdir_dt % 1 == 0:
                    subdir_dt = int(subdir_dt)
                subdir_name = outventory_file.parent.joinpath(
                    f"{subdir_dt}_day_mosaics"
                )
                if not subdir_name.exists():
                    subdir_name.mkdir(parents=True, exist_ok=True)
            for i in range(loop_iters):
                # print(i)
                if not time_bins_is_list:
                    start = time_bins[i]
                    end = time_bins[i + 1]

                    # convert from utc times to mjd and then from mjd to MET
                    start_met = sbu.datetime2met(start.datetime, correct=True)

                    end_met = sbu.datetime2met(end.datetime, correct=True)
                else:
                    start = time_bins[i][0, 0]
                    end = time_bins[i][1, 0]

                    # convert the start time bins edges to met times
                    start_met = [
                        sbu.datetime2met(j.datetime, correct=True)
                        for j in time_bins[i][0, :]
                    ]

                    # convert the end time bins edges to met times
                    end_met = [
                        sbu.datetime2met(j.datetime, correct=True)
                        for j in time_bins[i][1, :]
                    ]

                select_outventory(outventory_file, start_met, end_met)

                if time_bins_is_list:
                    # after selecting the array of times for the grouped outventory when we have a list passed in
                    # we need to set start_met and end_met to a single value for updating the header values
                    start_met = start_met[0]
                    end_met = end_met[0]

                # move the outventory file to the folder where we will keep them
                output_file = Path(str(outventory_file).replace(".fits", "_sel.fits"))

                start_time_str = convert_time_to_decimal_day(start)
                end_time_str = convert_time_to_decimal_day(end)
                savefile = savedir.joinpath(
                    output_file.name.replace(
                        "_sel.fits", f"_{start_time_str}_{end_time_str}.fits"
                    )
                )

                output_file.rename(savefile)

                with fits.open(str(savefile), mode="update") as file:
                    file[1].header["S_TBIN"] = (
                        float(start_met),
                        "Mosaicing Start of Time Bin (MET)",
                    )
                    file[1].header["E_TBIN"] = (
                        float(end_met),
                        "Mosaicing End of Time Bin (MET)",
                    )
                    file.flush()

                # create the directories that will hold all the mosaiced images within a given time bin
                if subdirs:
                    binned_savedir = subdir_name.joinpath(
                        f"mosaic_{start_time_str}_{end_time_str}"
                    )
                    dirtest(binned_savedir)
                else:
                    binned_savedir = outventory_file.parent.joinpath(
                        f"mosaic_{start_time_str}_{end_time_str}"
                    )

                    dirtest(binned_savedir)

    return time_bins


def read_skygrids(savedirectory=None):
    """
    Reads the skygrids that the user may have made using the make_skygrids function.

    :param savedirectory: Default None or a Path object to the location of the directory that contains all the skygrids
        that will be read in
    :return: numpy arrays of the ra/dec coordinates in degrees of the skygrid facets that are read in. the shape is (n,m,n_facets),
        where nxm is the size of each facet and n_facet corresponds to the number of facets that has been created.
    """
    # reads the skygrids and output numpy array that contains all the data

    # get the directory that the data directory is located in
    if savedirectory is None:
        dir = Path(__file__).parent
    else:
        dir = Path(savedirectory)

    nimages = len(_gcenters)  # number of sky facets

    # get the size of the first one for us to allocate an array for
    string = "c%d_%s" % (0, _proj)
    ra_string = "ra_" + string + ".img"
    ra_file = dir.joinpath("data").joinpath(ra_string)
    with fits.open(str(ra_file)) as file:
        grid_shape = file[0].data.shape

    # allocate arrays to hold data, already know sizes of the skygrids from lookng at them before
    ra_skygrid = np.zeros((grid_shape[0], grid_shape[1], nimages))
    dec_skygrid = np.zeros_like(ra_skygrid)

    # create the filenames and read in the data
    for i in range(_nskyimg):
        string = "c%d_%s" % (i, _proj)
        ra_string = "ra_" + string + ".img"
        dec_string = "dec_" + string + ".img"

        ra_file = dir.joinpath("data").joinpath(ra_string)
        dec_file = dir.joinpath("data").joinpath(dec_string)

        file = fits.open(str(ra_file))
        ra_skygrid[:, :, i] = file[0].data
        file.close()

        file = fits.open(str(dec_file))
        dec_skygrid[:, :, i] = file[0].data
        file.close()

    return ra_skygrid, dec_skygrid


def convert_radec2xy(ra, dec, header):
    """
    Converts RA.DEC coordinates to pixel coordinates based on astrometric header keywords.

    :param ra: numpy array of ra coordinates in degrees
    :param dec: numpy array of dec coordinates in degrees
    :param header: The header that will be used to extract astrometric keywords to convert RA/DEC to detector pixel
        coordinates or a WCS header object
    :return: numpy arrays of x and y in detector pixel coordinates
    """
    # use the astropy WCS object to convert from ra and dec to x,y

    # make the WCS object
    if not isinstance(header, WCS):
        w = WCS(header)
    else:
        w = header

    # calculate the xy values, think I need to use origin=0 because in fits header says that initial pixel is 0? not
    # sure need to double-check against idl code. When comparing wcs_world2pix to heasarc sky2xy, the results of
    # sky2xy matches with wcs_world2pix if we use origin=1, therefore we probably need this since the codes had
    # typically used heasarc scripts
    xy = w.wcs_world2pix(np.array([ra.flatten(), dec.flatten()], dtype="float64").T, 0)

    # reshape to be the original dimensions of the ra/dec arrays
    x = xy[:, 0].reshape(ra.shape)
    y = xy[:, 1].reshape(dec.shape)

    return x, y


def convert_xy2radec(x, y, header):
    """
    Converts pixel coordinates to RA.DEC coordinates based on astrometric header keywords.

    :param x: numpy array of pixel x coordinates
    :param y: numpy array of pixel y coordinates
    :param header: The header that will be used to extract astrometric keywords to convert RA/DEC to detector pixel
        coordinates
    :return: numpy arrays of RA/DEC in degrees
    """
    # use the astropy WCS object to convert from x,y to ra and dec

    # make the WCS object
    w = WCS(header)

    # calculate the ra/dec values, think I need to use origin=0 because in fits header says that initial pixel is 0?
    # not sure
    ra_dec = w.wcs_pix2world(np.array([x.flatten(), y.flatten()], dtype="float64").T, 0)

    # reshape to be the original dimensions of the ra/dec arrays
    ra = ra_dec[:, 0].reshape(x.shape)
    dec = ra_dec[:, 1].reshape(y.shape)

    if "galactic" in w.world_axis_physical_types[0]:
        # converted coordinates in galactic coordinates and need to convert to RA/DEC
        c = SkyCoord(l=ra, b=dec, frame="galactic", unit="deg")
        ra = c.fk5.ra.value
        dec = c.fk5.dec.value

    return ra, dec


def read_correctionsmap():
    """
    Reads the BAT coded mask energy-dependent off axis corrections mask which accounts for the fact that the mask has a
    finite width which affects the propagation of photons at some angle relative to the boresight.

    :return: numpy array of (954, 1760, _nebands) where _nebands=8, which is the number of energy bands in the BAT survey
    """
    # reads the correction map for correcting off-axis effects

    # get the directory that the data directory is located in
    dir = Path(__file__).parent

    file_string = dir.joinpath("data").joinpath(_cimgfile)

    # create array to hold data, already know sizes of grids from looking at file
    corrections_map = np.zeros((954, 1760, _nebands))

    # open file and read contents
    with fits.open(str(file_string)) as file:
        for i in range(_nebands):
            corrections_map[:, :, i] = file[i].data

    return corrections_map


def scox1_slop(ang_sep):
    """
    This calculaates the additional chi squared values that are added to the statistical fit of the survey
    observation image based on whether the pointing of the survey observation is near Sco X-1. This correction is
    applied at the lowest energy bin and attempts to account for the brightness of Sco X-1 in the survey images.

    :param ang_sep: numpy array of angular separation between the BAT survey observation RA/DEC pointing and the
        locaiton of Sco X-1
    :return: numpy array of the additional reduced chi squared values that should be used for determining if the
        low energy image is acceptable to include in the total mosaiced image.
    """
    # This seems to add some amount of chi squared value to the region near Sco X-1 so it gets cut out later on, this is
    # meant to be done only in the first energy band (14-20 keV)

    f = _chihithresh + _chiscobump / (1 + (ang_sep / _chiscotheta) ** 2)

    return f


def compute_statistics_map(chi_sq, nbatdet, ra_pnt, dec_pnt, pa_pnt, tstart):
    """
    Determines whether the statistics in a given BAT survey observation is sufficient to be added to the total mosaiced
    image. This function also exludes observations that are pointed at/near Sco X-1.

    :param chi_sq: numpy array of chi squared values for a set of BAT survey observations
    :param nbatdet: numpy array of the nbatdet values for a set of BAT survey observations (same order as above)
    :param ra_pnt: numpy array of the RA pointing values for a set of BAT survey observations (same order as above)
    :param dec_pnt: numpy array of the DEC pointing values for a set of BAT survey observations (same order as above)
    :param pa_pnt: numpy array of the pointing angle valules for a set of BAT survey observations (same order as above)
    :param tstart: numpy array of the pointing observations' start time in MET (same order as above)
    :return: numpy array mask of good and bad survey observations (0=bad observation that will be excluded)
    """
    # computes the stastics map based on chi squared values and angular separation from Sco X-1
    # found that comparing the original mosaic code reduced chisq can vary from the current reduced chisq value
    # by ~50% at low energy range and ~6% at highest energy range
    # This fudge may not be necessary when using the proper noise map for each day
    fudge = 1.5

    # reduced chisq
    red_chi2 = chi_sq / nbatdet[:, np.newaxis]

    # calculate angular separation between the pointings and Sco X-1
    coord_array = SkyCoord(ra_pnt, dec_pnt, frame="icrs", unit="deg")
    ang_sep = coord_array.separation(_sco_coord)  # these are in degrees

    # calculate the extra chisq value added around Sco X-1 for the lowest energy band
    sco_xtra_chi2 = scox1_slop(ang_sep.value)

    # stop

    # create the mask (1=good; 0=bad) based on if the reduced chisq values in each energy bin meet the requirements
    mask = np.zeros_like(chi_sq[:, 0])
    for i in range(_nebands):
        if i == 0:
            mask = (red_chi2[:, i] < fudge * sco_xtra_chi2) & (
                red_chi2[:, i] > _chilothresh
            )
        else:
            mask = (
                mask & (red_chi2[:, i] > _chilothresh) & (red_chi2[:, i] < _chihithresh)
            )

    # include whether Sco is the object corresponding to the pointing. If it is, we want to exclude this pointing ID,
    # therefore set mask=0
    idx = np.where(
        (ra_pnt > 245)
        & (ra_pnt < 246)
        & (dec_pnt > -18)
        & (dec_pnt < -17)
        & (pa_pnt > 100)
        & (pa_pnt < 110)
        & (tstart > 0)
    )
    mask[idx] = 0

    return np.array(mask, dtype=np.int64)


def write_mosaic(
    img,
    header,
    filename_base,
    emin=[14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 14.0],
    emax=[20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0, 195.0],
):
    """
    Write out the intermediate mosaic images to fits files.

    :param img: numpy array of the image that will be saved.
    :param header: The astropy header object that contains image specific information. This header will be appended to
        the header of the associated skygrid facet, and additional standard keywords (see the add_header variable
        within this function).
    :param filename_base: Path object that denotes the directory where the images will be saved.
    :param emin: The lower energy values for each survey energy bin that is created for each image. This should not need
        to be modified.
    :param emax: The upper energy values for each survey energy bin that is created for each image. This should not need
        to be modified.
    :return: None
    """
    # actually writes out the files that we produced in create mosaics

    filename_base = Path(filename_base)

    # get the directory that the data directory is located in
    direc = Path(Path(__file__).parent)

    # get the current date_time
    time_now = str(np.datetime64("now"))

    # create header with additional keywords that we want to add
    add_header = fits.Header()
    add_header["TIMESYS"] = ("TT", " Time system")
    add_header["MJDREFI"] = (51910.0, " Reference MJD Integer part")
    add_header["MJDREFF"] = (0.00074287037, " Reference MJD fractional")
    add_header["TIMEREF"] = ("LOCAL", " Time reference (barycenter/local)")
    add_header["TASSIGN"] = ("SATELLITE", " Time assigned by clock")
    add_header["TIMEUNIT"] = ("s", " Time unit")
    add_header["TIERRELA"] = (1.0e-8, " [s/s] relative errors expressed as rate")
    add_header["TIERABSO"] = (1.0, " [s] timing precision in seconds")
    add_header["CLOCKAPP"] = ("F", "Is mission time corrected for clock drift?")
    add_header["TELESCOP"] = ("SWIFT", " Telescope (mission) name")
    add_header["INSTRUME"] = ("BAT", " Instrument name")
    add_header["EQUINOX"] = (2000.0, " Equinox")
    add_header["RADECSYS"] = ("FK5", " Coordinate system")
    add_header["ORIGIN"] = ("SWIFT BAT TEAM", " Source of FITS file")
    add_header["CREATOR"] = ("BatAnalysis", " Program that created FITS file")
    add_header["DATE"] = time_now
    add_header["BACKAPP"] = ("T", " Was background subtracted?")
    add_header["FLUXMETH"] = ("WEIGHTED", " Flux extraction method")
    add_header["HDUCLASS"] = ("OGIP", " Conforms to OGIP/GSFC standards")
    add_header["HDUCLAS1"] = ("IMAGE", " Contains image data")

    # get the type of image that we are saving to denote the name
    hdu_comment = header.comments["HDUCLAS2"]
    if "PCODE" in hdu_comment:
        file_start = "pcode_"
    elif "EXPMAP" in hdu_comment:
        file_start = "expmap_"
    elif "VARIANCE" in hdu_comment:
        file_start = "var_"
    elif "SKY_WT_FLUX" in hdu_comment:
        file_start = "flux_"

    # read in the appropriate headers and create new headers and save files
    for i in range(_nskyimg):
        string = "c%d_%s" % (i, _proj)
        ra_string = "ra_" + string + ".img"

        ra_file = direc.joinpath("data").joinpath(
            ra_string
        )  # os.path.join(direc, "data", ra_string)
        with fits.open(str(ra_file)) as file:
            skygrid_header = file[0].header

        total_header = header + add_header + skygrid_header
        total_header["BSKYPLAN"] = (string, "BAT mosaic ZEA sky plane ID (0-5)")

        # construct the name of the file that we will be saving
        savefile = filename_base.joinpath(file_start + string + ".img")

        if img.ndim == 3:
            # if this the pimg or eimg
            if i == 0:
                fits.writeto(str(savefile), img[:, :, i], total_header)
            else:
                fits.append(str(savefile), img[:, :, i], total_header)
        else:
            for j in range(len(emin)):
                # if this is the variance or sky flux image need ot add energy related header keys
                total_header["BENRGYBN"] = (
                    f"E_{int(emin[j]):03}_{int(emax[j]):03}",
                    "BAT mosaic energy bin (keV)",
                )
                total_header["E_MIN"] = (emin[j], " [keV] Lower energy bin edge")
                total_header["E_MAX"] = (emax[j], " [keV] Upper energy bin edge")

                if j == 0:
                    total_header["EXTEND"] = ("T", "File contains extensions")
                    total_header["HDUNAME"] = (
                        f"E_{int(emin[j]):03}_{int(emax[j]):03}",
                        "BAT mosaic energy bin (keV)",
                    )
                    fits.writeto(str(savefile), img[:, :, i, j], total_header)
                else:
                    if j == 1:
                        total_header.remove("EXTEND")
                        total_header.remove("HDUNAME")
                    total_header["EXTNAME"] = (
                        f"E_{int(emin[j]):03}_{int(emax[j]):03}",
                        "BAT mosaic energy bin (keV)",
                    )
                    fits.append(str(savefile), img[:, :, i, j], total_header)


def finalize_mosaic(intermediate_mosaic_directory):
    """
    Converts the intermediate mosaic images, located in intermediate_mosaic_directory, to physical units and saves them.

    :param intermediate_mosaic_directory: Path object that points to the directory where the intermediate mosaic images
        are located.
    :return: None
    """
    # takes the intermediate mosaic files and applies proper units

    intermediate_mosaic_directory = Path(intermediate_mosaic_directory)

    # loop over the sky images
    for i in range(_nskyimg):
        string = "c%d_%s" % (i, _proj)

        # copy the expmap to the new name and remove a few header keywords
        file_name = intermediate_mosaic_directory.joinpath(f"expmap_{string}.img")
        output_name = intermediate_mosaic_directory.joinpath(
            f"swiftbat_flatexp_c{i}.img"
        )
        shutil.copy(file_name, output_name)

        with fits.open(output_name, mode="update") as file:
            header = file[0].header
            header.remove("BLSTOBS")
            header.remove("BLSTOUTP")
            header.remove("BLSTPNT")
            header.remove("BMOSMON")
            file.flush()

        # do the same for the pcode file
        file_name = intermediate_mosaic_directory.joinpath(f"pcode_{string}.img")
        output_name = intermediate_mosaic_directory.joinpath(
            f"swiftbat_exposure_c{i}.img"
        )
        shutil.copy(file_name, output_name)

        with fits.open(output_name, mode="update") as file:
            header = file[0].header
            header.remove("BLSTOBS")
            header.remove("BLSTOUTP")
            header.remove("BLSTPNT")
            header.remove("BMOSMON")
            file.flush()

        # for the flux, need to do simg/vimg and save this for each sky image and energy band
        # and change some of the header keywords
        flux_file_name = intermediate_mosaic_directory.joinpath(f"flux_{string}.img")
        flux_output_name = intermediate_mosaic_directory.joinpath(
            f"swiftbat_flux_c{i}.img"
        )
        shutil.copy(flux_file_name, flux_output_name)

        # for the SNR, need to do simg/sqrt(vimg) and save this for each sky image and energy band
        # and change some of the header keywords
        snr_file_name = intermediate_mosaic_directory.joinpath(f"flux_{string}.img")
        snr_output_name = intermediate_mosaic_directory.joinpath(
            f"swiftbat_snr_c{i}.img"
        )
        shutil.copy(snr_file_name, snr_output_name)

        # after calculating the flux and the SNR, for the variance, need to convert from units of (1/cts/s)^2) to cts/s
        # and modify the header
        var_file_name = intermediate_mosaic_directory.joinpath(f"var_{string}.img")
        var_output_name = intermediate_mosaic_directory.joinpath(
            f"swiftbat_var_c{i}.img"
        )
        shutil.copy(var_file_name, var_output_name)

        # open all the files in update mode
        flux_file = fits.open(str(flux_output_name), mode="update")
        snr_file = fits.open(str(snr_output_name), mode="update")
        var_file = fits.open(str(var_output_name), mode="update")

        for j in range(len(flux_file)):
            # make modifications for flux
            flux_file[j].header.remove("BLSTOBS")
            flux_file[j].header.remove("BLSTOUTP")
            flux_file[j].header.remove("BLSTPNT")
            flux_file[j].header.remove("BMOSMON")
            flux_file[j].header["BUNIT"] = ("count/s", " Flux level")
            flux_file[j].header["HDUCLAS2"] = ("NET", " Contains net flux map <== FLUX")
            flux_file[j].header["IMATYPE"] = ("INTENSITY", " Contains net flux map")

            flux_file[j].data = flux_file[j].data / var_file[j].data

            flux_file.flush()

            # make modifications for var
            var_file[j].header.remove("BLSTOBS")
            var_file[j].header.remove("BLSTOUTP")
            var_file[j].header.remove("BLSTPNT")
            var_file[j].header.remove("BMOSMON")
            var_file[j].header["BUNIT"] = ("count/s", " Image variance flux level")
            var_file[j].header["HDUCLAS2"] = (
                "BKG_STDDEV",
                " Contains std. deviation map <== NOISE",
            )
            var_file[j].header["HDUCLAS3"] = (
                "PREDICTED",
                " Predicted standard deviation",
            )
            var_file[j].header["IMATYPE"] = ("ERROR", " Contains std. deviation map")
            var_file[j].data = 1 / np.sqrt(var_file[j].data)
            var_file.flush()

            # make modifications for SNR
            snr_file[j].header.remove("BLSTOBS")
            snr_file[j].header.remove("BLSTOUTP")
            snr_file[j].header.remove("BLSTPNT")
            snr_file[j].header.remove("BMOSMON")
            snr_file[j].header["BUNIT"] = ("sigma", " Image significance (sigma)")
            snr_file[j].header["HDUCLAS2"] = (
                "SIGNIFICANCE",
                " Contains significance map <== SNR",
            )
            snr_file[j].header["IMATYPE"] = (
                "SIGNIFICANCE",
                " Contains significance map",
            )
            # remember that the data in snr file is the old flux data, so we just overwrite it entirely
            snr_file[j].data = flux_file[j].data / var_file[j].data
            snr_file.flush()

        flux_file.close()
        snr_file.close()
        var_file.close()


def create_mosaics(
    outventory_file,
    time_bins,
    survey_list,
    catalog_file=None,
    total_mosaic_savedir=None,
    recalc=False,
    verbose=True,
):
    """
    Creates the mosaiced images for specified time bins and a total mosaic image that is "time-integrated" across all
    time bins.

    :param outventory_file: Path object of the outventory file that contains all the BAT survey observations that will
        be used to create the mosaiced images.
    :param time_bins: The time bins that the observatons in outventory file have been grouped into
    :param survey_list: The list of BATSurvey objects that correpond to the observations listed in the outventory file
    :param catalog_file: A Path object of the catalog file that should be used to identify sources in the mosaic images.
        This will default to using the catalog file that is included with the BatAnalysis package.
    :param total_mosaic_savedir: Default None or a Path object that denotes the directory that the total
        "time-integrated" images will be saved to. The default is to place the total mosaic image in a directory called
        "total_mosaic" located in the same directory as the outventory file.
    :param recalc: Boolean False by default. If this calculation was done previously, do not try to load the results of
        prior calculations. Instead recalculate the mosaiced images. The default, will cause the function to try to load
        a save file to save on computational time.
    :param verbose: Boolean True by default. Tells the code to print progress/diagnostic information.
    :return: a list of MosaicBatSurvey objects correponding to each time bin that was requested, and a single M
        osaicBatSurvey corresponding to the total mosaiced image across all time bins.
    """
    # This function actually creates the mosaic-ed files, NOTE there is no usco inclusion here, but this can be
    # easily added if its really necessary. In the idl code, it didnt seem like this was used, but not sure.

    # make sure its a path object
    outventory_file = Path(outventory_file)

    # get the corections map and the skygrids
    corrections_map = read_correctionsmap()
    ra_skygrid, dec_skygrid = read_skygrids()

    # get the correct catalog file
    if catalog_file is None:
        catalog_file = Path(__file__).parent.joinpath("data/survey6b_2.cat")

    # determine format of the time_bins, ie an astropy Time array or a list of astropy Time arrays
    # no error checking here since it should be taken care of in group_outventory function
    time_bins_is_list = False
    if type(time_bins) is list:
        time_bins_is_list = True

    intermediate_mosaic_dir_list = []
    all_mosaic_survey = []

    # get the number of iterations we need to do in teh loop below
    if not time_bins_is_list:
        # this is to account for the fact that we have the array consisting of the start/end edges all in one
        loop_iters = len(time_bins) - 1
    else:
        # this accounts for having the list of just the starting bin edges or the end bin edges
        loop_iters = len(time_bins)

    # loop over the time bins
    for i in range(loop_iters):
        if not time_bins_is_list:
            start = time_bins[i]
            end = time_bins[i + 1]
        else:
            start = time_bins[i][0, 0]
            end = time_bins[i][1, 0]

        if verbose:
            print(f"Working on time bins from {start} to {end}.\n")

        mosaic_obj = _mosaic_loop(
            outventory_file,
            start,
            end,
            corrections_map,
            ra_skygrid,
            dec_skygrid,
            survey_list,
            recalc=recalc,
            verbose=not verbose,
        )
        if mosaic_obj is not None:
            mosaic_obj.detect_sources(catalog_file=catalog_file)
            all_mosaic_survey.append(mosaic_obj)

    intermediate_mosaic_dir_list = [i.result_dir for i in all_mosaic_survey]

    # see if the total mosaic has been created and saved (ie there is a .batsurvey file in that directory) if there
    # isnt, then do the full calculation or if we set recalc=True then also do the full calculation
    if total_mosaic_savedir is None:
        total_mosaic_savedir = intermediate_mosaic_dir_list[0].parent.joinpath(
            "total_mosaic"
        )
    else:
        total_mosaic_savedir = Path(total_mosaic_savedir)

    if not total_mosaic_savedir.joinpath("batsurvey.pickle").exists() or recalc:
        # merge all the mosaics together to get the full 'time integrated' images and convert to final files with
        # proper units
        total_dir = merge_mosaics(
            intermediate_mosaic_dir_list, savedir=total_mosaic_savedir
        )
        finalize_mosaic(total_dir)
        total_mosaic = MosaicBatSurvey(total_dir)
        total_mosaic.detect_sources(catalog_file=catalog_file)
        total_mosaic.save()
    else:
        total_mosaic = MosaicBatSurvey(total_mosaic_savedir)

    return all_mosaic_survey, total_mosaic


def _mosaic_loop(
    outventory_file,
    start,
    end,
    corrections_map,
    ra_skygrid,
    dec_skygrid,
    survey_list,
    recalc=False,
    verbose=True,
):
    """
    The loop that computes the mosaiced images for a time bin of interest. It sums up all the BAT survey observations
    where:
     the partial coding images are multiplied by the exposure time of each image and summed
     the exposure images are directly summed
     the flux images are weighted by the inverse variance of the image and summed
     and the inverse variance images are summed together.
    Then, these intermediate images are converted to physical units where:
     the flux is multiplied by the summed inverse variance image
     the inverse variance image is converted back to normal variance.

    :param outventory_file: Path object that provides the full outventory file of the BAT survey observations that will
        be used to calculate the mosaiced images.
    :param start: astropy Time of the start time of the time bin that survey observations need to be made to be included
        in that time bin's mosaiced image.
    :param end: astropy Time of the end time of the time bin that survey observations need to be made to be included
        in that time bin's mosaiced image.
    :param corrections_map: numpy array with the energy dependent off-axis corrections map
    :param ra_skygrid: numpy array of the skygrid facets' RA values in degrees
    :param dec_skygrid:numpy array of the skygrid facets' DEC values in degrees
    :param survey_list: list of BAT survey objects that should have been used to create the full outventory file passed
        into the outventory_file parameter.
    :param recalc: Boolean False by default. If this calculation was done previously, do not try to load the results of
        prior calculations. Instead recalculate the mosaiced images. The default, will cause the function to try to load
        a save file to save on computational time.
    :param verbose: Boolean True by default. Tells the code to print progress/diagnostic information.
    :return: a MosaicBatSurvey object correponding to the time bin that was requested
    """

    if verbose:
        print(f"Working on time bins from {start} to {end}.\n")

    # get the name of the file with binned outventory info and where its saved
    savedir = outventory_file.parent.joinpath("grouped_outventory")
    output_file = savedir.joinpath(
        outventory_file.name.replace(
            ".fits", f"_{start.datetime64.astype('datetime64[D]')}.fits"
        )
    )
    # see if we need to use the mjd time format
    if not output_file.exists():
        output_file = savedir.joinpath(
            outventory_file.name.replace(".fits", f"_{start.mjd}.fits")
        )

    # this is the directory of the time bin where the images will be saved
    img_dir = outventory_file.parent.joinpath(
        f"mosaic_{start.datetime64.astype('datetime64[D]')}"
    )
    if not img_dir.exists():
        img_dir = outventory_file.parent.joinpath(f"mosaic_{start.mjd}")

    # see if there is a .batsurvey file, if it doesnt exist or if we want to recalc things then go through the full loop
    if not img_dir.joinpath("batsurvey.pickle").exists() or recalc:
        # loop over the survey list to get the observation IDs for reference later
        survey_obids = [i.obs_id for i in survey_list]

        # read the fits file for the date/time of interest
        with fits.open(str(output_file)) as file:
            grouped_outventory_data = file[1].data

        # if there are survey observations within the time bin of interest do all this stuff
        if grouped_outventory_data["NBATDETS"].size > 0:
            # calculate the mask of which points we should use based on good image statistics
            chi_mask = compute_statistics_map(
                grouped_outventory_data["CHI2"],
                grouped_outventory_data["NBATDETS"],
                grouped_outventory_data["RA_PNT"],
                grouped_outventory_data["DEC_PNT"],
                grouped_outventory_data["PA_PNT"],
                grouped_outventory_data["TSTART"],
            )

            # create the arays that will hold the binned data
            eimg = np.zeros_like(
                ra_skygrid
            )  # exposure map, has the same dimensions as the skygrid
            pimg = np.zeros_like(ra_skygrid)  # Partial coding map
            nx, ny, nz = ra_skygrid.shape
            vimg = np.zeros(
                (nx, ny, nz, _nebands + 1)
            )  # Variance map, size of skygrid with extra enegy dimension (+1 for total 14-195 band)
            simg = np.zeros_like(vimg)  # Sky flux  image

        total_binned_exposure = 0  # tally up the total exposure
        total_tstart = []
        total_tstop = []
        total_dateobs_start = []
        total_dateobs_end = []
        total_headers = []

        # this holds the merged files in the next loop
        merged_pointing_dir = []
        obsids = []
        data_directories = []

        # loop over the observation IDs and the pointings that are outlined in the
        for j in range(grouped_outventory_data["NBATDETS"].size):
            obsid = grouped_outventory_data["OBS_ID"][j]
            pointing_id = grouped_outventory_data["IMAGE_ID"][j]

            # test that we have good image statistics
            if (
                (chi_mask[j] == 0)
                or (grouped_outventory_data["NBATDETS"][j] <= 0)
                or (grouped_outventory_data["IMAGE_STATUS"][j] == False)
                or (grouped_outventory_data["EXPOSURE"][j] <= 0)
            ):
                if verbose:
                    print(
                        "Bad image Statistics. Skipping observation ID/Pointing: %s/%s\n"
                        % (obsid, pointing_id)
                    )
            else:
                if verbose:
                    print(
                        "Good image Statistics. Working on observation ID/Pointing: %s/%s\n"
                        % (obsid, pointing_id)
                    )

                # get the inde of the appropriate survey object in the list
                surveylist_idx = survey_obids.index(obsid)

                # need to also make sure that the pointing ID is actually valid and it has all the files necessary
                pointing_id_number = pointing_id.split("_")[-1]
                if pointing_id_number in survey_list[surveylist_idx].pointing_ids:
                    # get the directory of the observation ID where the survey result lives
                    batsurvey_result_dir = survey_list[surveylist_idx].result_dir

                    data_directory = batsurvey_result_dir.joinpath(pointing_id)

                    ncleaniter = survey_list[surveylist_idx].batsurvey_result.params[
                        "ncleaniter"
                    ]

                    # read the partial coding map, variance map, sky flux map for the pointing
                    pointing_pimg_str = data_directory.joinpath(
                        f"{pointing_id}_{ncleaniter}.img"
                    )
                    with fits.open(str(pointing_pimg_str)) as file:
                        # read the partial coding map
                        pointing_pimg = file["BAT_PCODE_1"].data

                        # get the image size and create array to hold the sky flux at each channel
                        sz = pointing_pimg.shape
                        pointing_simg = np.zeros(
                            (sz[0], sz[1], _nebands + 1)
                        )  # plus 1 for the total energy
                        for k in range(_nebands):
                            pointing_simg[:, :, k] = file[k].data

                        # get other header information
                        survey_ver = file["BAT_PCODE_1"].header["BSURVER"]
                        # survey_eq=file['BAT_PCODE_1'].header['BSURSEQ'] #not in there? it is 8b in the headers on the machines
                        pointing_exposure = file["BAT_PCODE_1"].header["EXPOSURE"]
                        pointing_tstart = file["BAT_PCODE_1"].header["TSTART"]
                        pointing_tstop = file["BAT_PCODE_1"].header["TSTOP"]
                        pointing_dateobs_start = file["BAT_PCODE_1"].header["DATE-OBS"]
                        pointing_dateobs_end = file["BAT_PCODE_1"].header["DATE-END"]

                        # save the header for use later
                        pointing_pimg_header = file[0].header

                    # make sure that the exposure is over the minimum limit
                    if pointing_exposure >= _minexpo:
                        # read in the variance images at each energy
                        pointing_vimg = np.zeros_like(pointing_simg)
                        pointing_vimg_str = data_directory.joinpath(
                            f"{pointing_id}_{ncleaniter}.var"
                        )
                        with fits.open(str(pointing_vimg_str)) as file:
                            for k in range(_nebands):
                                pointing_vimg[:, :, k] = file[k].data

                        # correct for off axis effects
                        pointing_vimg_corr = np.zeros_like(pointing_vimg)
                        pointing_simg_corr = np.zeros_like(pointing_vimg)
                        pointing_vimg_corr[:, :, :-1] = (
                            pointing_vimg[:, :, :-1] / corrections_map
                        )
                        pointing_simg_corr[:, :, :-1] = (
                            pointing_simg[:, :, :-1] / corrections_map
                        )

                        # construct the total energy images for variance and flux, the zeros in last array dont affect
                        # calculations of the total values
                        pointing_vimg_corr[:, :, -1] = np.sqrt(
                            np.sum(pointing_vimg_corr**2, axis=2)
                        )
                        pointing_simg_corr[:, :, -1] = pointing_simg_corr.sum(axis=2)

                        # construct the quality map for each energy and for the total energy images
                        energy_quality_mask = np.zeros_like(pointing_vimg_corr)
                        good_idx = np.where(
                            (
                                np.repeat(
                                    pointing_pimg[:, :, np.newaxis],
                                    pointing_vimg_corr.shape[-1],
                                    axis=2,
                                )
                                > _pcodethresh
                            )
                            & (pointing_vimg_corr > 0)
                            & np.isfinite(pointing_simg_corr)
                            & np.isfinite(pointing_vimg_corr)
                        )
                        energy_quality_mask[good_idx] = 1

                        # make the intermediate maps for each energy and for the total energy
                        interm_pointing_eimg = (
                            energy_quality_mask * pointing_exposure
                        )  # Exposure map
                        interm_pointing_pimg = (
                            pointing_pimg[:, :, np.newaxis]
                            * energy_quality_mask
                            * pointing_exposure
                        )  # partial coding map
                        interm_pointing_vimg = (
                            energy_quality_mask / (pointing_vimg_corr + 1e-10) ** 2
                        )  # Convert to 1 / variance
                        interm_pointing_simg = (
                            pointing_simg_corr
                            * energy_quality_mask
                            * interm_pointing_vimg
                        )  # variance weighted sky flux

                        # need to compute the x/y position for each RA/DEC point in the sky map using the new
                        # file for the pointing of interest
                        pixel_x, pixel_y = convert_radec2xy(
                            ra_skygrid, dec_skygrid, pointing_pimg_header
                        )

                        # get the good values, in the idl file the shape of pointing_pimg is reversed, not sure if
                        # this is correct here.
                        pixel_idx = np.where(
                            (pixel_y <= pointing_pimg.shape[0])
                            & (pixel_x <= pointing_pimg.shape[1])
                            & (pixel_x >= -1)
                            & (pixel_y >= -1)
                            & np.isfinite(pixel_x)
                            & np.isfinite(pixel_y)
                        )
                        chosen_pixel_x = pixel_x[pixel_idx]
                        chosen_pixel_y = pixel_y[pixel_idx]

                        # need to interpolate the survey sky image onto the all sky image
                        # need to verify that the eimg and pimg maps are energy independent, in idl code only does this
                        # for te first energy iteration
                        grid_x, grid_y = np.mgrid[
                            0 : pointing_pimg.shape[0], 0 : pointing_pimg.shape[1]
                        ]
                        points = np.array([grid_x.flatten(), grid_y.flatten()])
                        values = interm_pointing_eimg[:, :, 0]
                        values[np.isnan(values)] = 0

                        # before had: #np.array([chosen_pixel_x, chosen_pixel_y]) for the below line but the results
                        # werent consistent with the idl code results. changing the x and y pixel coordinates here works
                        interp_at_points = np.array([chosen_pixel_y, chosen_pixel_x])

                        # see if thie other method works,
                        # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
                        vtx, wts = interp_weights(points.T, interp_at_points.T)
                        test = interpolate(values.flatten(), vtx, wts, fill_value=0)
                        eimg[
                            pixel_idx
                        ] += test  # new interpolate dir, took 722.239518339 s

                        # tried if method here works
                        # https://stackoverflow.com/questions/51858194/storing-the-weights-used-by-scipy-griddata-for-re-use/51937990#51937990
                        # found that it took 3247.2622033880034 s versus 722.239518339 s

                        values = interm_pointing_pimg[:, :, 0]
                        values[np.isnan(values)] = (
                            0  # if there are nan values in the images, this can mess up the interpolation
                        )

                        test = interpolate(values.flatten(), vtx, wts, fill_value=0)
                        pimg[pixel_idx] += test

                        for k in range(_nebands + 1):
                            values = interm_pointing_simg[:, :, k]
                            values[np.isnan(values)] = 0

                            test = interpolate(values.flatten(), vtx, wts, fill_value=0)
                            simg[:, :, :, k][pixel_idx] += test  # new interpolate dir

                            values = interm_pointing_vimg[:, :, k]
                            values[np.isnan(values)] = 0

                            test = interpolate(values.flatten(), vtx, wts, fill_value=0)
                            vimg[:, :, :, k][pixel_idx] += test  # new interpolate dir

                        # keep track of exposure and times
                        total_binned_exposure += pointing_exposure
                        total_tstart.append(pointing_tstart)
                        total_tstop.append(pointing_tstop)
                        total_dateobs_start.append(pointing_dateobs_start)
                        total_dateobs_end.append(pointing_dateobs_end)
                        merged_pointing_dir.append(batsurvey_result_dir)
                        obsids.append(obsid)
                        data_directories.append(data_directory)

        # only do this stuff if there were files that needed to be mosaiced
        # if there were no files that were mosaiced for the time interval dont copy any of the template fits files
        # to save space, also dont include these time bins in the total mosaic calculation
        if len(merged_pointing_dir) > 0:

            # need to write the outputs after combining datasets that fall within a time bin
            # create a model header
            model_hdr = fits.Header()
            model_hdr["BSURSEQ"] = (
                "8b",
                " BAT survey sequence id",
            )  # was 8b in most recent survey mosaics
            model_hdr["BSURVER"] = (
                heasoftpy.__version__,
                " BAT survey processing version",
            )  # was 6.16 in  hsp version 0.1.22
            model_hdr["BMOSVER"] = (
                "py" + pkg_resources.require("BatAnalysis")[0].version,
                " BAT mosaic processing version",
            )
            model_hdr["BMOSMON"] = (
                str(start.datetime64.astype("datetime64[D]")),
                " BAT mosaic processing date",
            )

            model_hdr["BLSTOUTP"] = (
                str(merged_pointing_dir[-1]),
                " BAT archive for last pointing written to mosaic file",
            )
            model_hdr["BLSTOBS"] = (
                obsids[-1],
                " BAT observation for last pointing written to mosaic file",
            )
            model_hdr["BLSTPNT"] = (
                data_directories[-1].name,
                " BAT last pointing written to mosaic file",
            )
            model_hdr["TSTART"] = (np.min(total_tstart), " start time of image")
            model_hdr["TSTOP"] = (np.max(total_tstop), " stop time of image")
            model_hdr["TELAPSE"] = (
                np.max(total_tstop) - np.min(total_tstart),
                "  elapsed time of image (= TSTOP-TSTART)",
            )
            model_hdr["DATE-OBS"] = (
                total_dateobs_start[np.argmin(total_tstart)],
                "  TSTART, expressed in UTC",
            )
            model_hdr["DATE-END"] = (
                total_dateobs_end[np.argmax(total_tstop)],
                "  TSTOP, expressed in UTC",
            )

            model_hdr["EXPOSURE"] = (
                total_binned_exposure,
                "[sec.] Sum of pointing exposures used",
            )

            # Add info about the user specified TBIN that was used to create the mosaic
            start_met = sbu.datetime2met(start.datetime)

            end_met = sbu.datetime2met(end.datetime)

            model_hdr["S_TBIN"] = (start_met, "Mosaicing Start of Time Bin (MET)")
            model_hdr["E_TBIN"] = (end_met, "Mosaicing End of Time Bin (MET)")

            # add/modify extra stuff for pcoding*exp image
            model_hdr["HDUCLAS2"] = (
                "VIGNETTING",
                " Contains partial coding map <== PCODE*EXP",
            )
            model_hdr["IMATYPE"] = ("EXPOSURE", " Contains partial coding map ")
            model_hdr["BUNIT"] = ("s ", " Exposure map")
            write_mosaic(pimg, model_hdr, img_dir)

            # add/modify extra stuff for exposure image
            model_hdr["HDUCLAS2"] = ("FLAT_EXP", " Contains exposure map <== EXPMAP")
            model_hdr["IMATYPE"] = ("EXPOSURE", " Contains partial coding map ")
            model_hdr["BUNIT"] = ("s ", " Exposure map")
            write_mosaic(eimg, model_hdr, img_dir)

            # add/modify extra stuff for variance image
            model_hdr["HDUCLAS2"] = (
                "VAR_WEIGHTS",
                " Contains sum of weights <== 1/VARIANCE",
            )
            model_hdr["IMATYPE"] = ("VARIANCE", " Contains sum of weights")
            model_hdr["BUNIT"] = (
                "1/(counts/sec)^2",
                " Physical units for sum-of-weights image",
            )
            write_mosaic(vimg, model_hdr, img_dir)

            # add/modify extra stuff for sky flux image
            model_hdr["HDUCLAS2"] = (
                "SKY_WT_FLUX",
                " Contains var. weighted sky flux <== SKY_WT_FLUX",
            )
            model_hdr["IMATYPE"] = ("INTENSITY", " Contains sky flux flux map")
            model_hdr["BUNIT"] = (
                "1/(counts/sec)",
                " Physical units for weighted-flux image",
            )
            write_mosaic(simg, model_hdr, img_dir)

            # in idl code the mosaic_wrt_outventory routine is called but this seems to reproduce the table that we call
            # in the beginning of this function to determine which pointings to merge. NEED TO DETERMINE IF THIS IS CORRECT
            # AND NECESSARY

            # Convert intermediate files to final files with proper units
            finalize_mosaic(img_dir)

            # create a mosaic survey object to hold all the information and allow the user to
            mosaic_survey = MosaicBatSurvey(img_dir)
            mosaic_survey.save()
        else:
            mosaic_survey = None
    else:
        # otherwise load the .batsurvey file
        mosaic_survey = MosaicBatSurvey(img_dir)

    return mosaic_survey


def _add_hpx_mosaics(hpx_file_list, verbose=True, altname=False, ebins=8):
    """
    Helper function to add healpix mosaics together.

    :param hpx_file_list: list of Path objects that provide the healpix mosaic files to be added together.
    :param verbose: Boolean True by default. Tells the code to print progress/diagnostic information.
    :param altname: Boolean False by default. If True, use alternative naming convention for energy bands.
        By default they are names e0 to e8. with e8 being total energy, if alternate naming convention is used
        they will be names e1 to e8, with e_tot being total energy band.
    :param ebins: Number of energy bins. In case it is missing from header, this will be used.
    :return: a h5py file
    """

    # Now read each file and add them
    # Instatiate an empty healpix grid
    # To do this read a sample h5py file
    dummy_hpx_data = h5py.File(hpx_file_list[0], "r")
    metadata = dict(dummy_hpx_data["mosaic"].attrs)
    dummy_hpx_data.close()

    npix = hp.nside2npix(nside=metadata["nside"])
    try:
        en_chans = metadata["EBINS"]
    except KeyError:
        en_chans = ebins
        metadata["EBINS"] = en_chans

    # Make empty sky images and variance maps
    sky_images = np.zeros((en_chans + 1, npix))  # +1 for total energy band
    var_images = np.zeros_like(sky_images)

    sky_exposure = np.zeros(npix)
    for i in range(len(hpx_file_list)):
        if verbose:
            print(f"Adding pointing {i+1} of {len(hpx_file_list)}")
        ind_pointing_obj = h5py.File(hpx_file_list[i], "r")
        ind_pointing_data = ind_pointing_obj["mosaic"][()]

        # Sky images are named sky_e1 and so on

        if not altname:
            for ein in range(en_chans + 1):
                sky_images[ein, :] = np.nansum(
                    [
                        sky_images[ein, :],
                        divide_with_respect(
                            ind_pointing_data[f"sky_e{ein}"],
                            ind_pointing_data[f"var_e{ein}"] ** 2,
                        ),
                    ],
                    axis=0,
                )
                var_images[ein, :] = np.nansum(
                    [
                        var_images[ein, :],
                        divide_with_respect(1, ind_pointing_data[f"var_e{ein}"] ** 2),
                    ],
                    axis=0,
                )
        else:
            for ein in range(en_chans):
                sky_images[ein, :] = np.nansum(
                    [
                        sky_images[ein, :],
                        divide_with_respect(
                            ind_pointing_data[f"sky_e{ein+1}"],
                            ind_pointing_data[f"var_e{ein+1}"] ** 2,
                        ),
                    ],
                    axis=0,
                )
                var_images[ein, :] = np.nansum(
                    [
                        var_images[ein, :],
                        divide_with_respect(1, ind_pointing_data[f"var_e{ein+1}"] ** 2),
                    ],
                    axis=0,
                )

            # Once this is done, do the total energy band
            sky_images[-1, :] = np.nansum(
                [
                    sky_images[-1, :],
                    divide_with_respect(
                        ind_pointing_data["sky_etot"],
                        ind_pointing_data["var_etot"] ** 2,
                    ),
                ],
                axis=0,
            )
            var_images[-1, :] = np.nansum(
                [
                    var_images[-1, :],
                    divide_with_respect(1, ind_pointing_data["var_etot"] ** 2),
                ],
                axis=0,
            )

        # Next get the net exposure
        sky_exposure = np.nansum([sky_exposure, ind_pointing_data["exposure"]], axis=0)

    if verbose:
        print("Mosaicking them")
    # Now conver them all to proper units
    sky_images = divide_with_respect(sky_images, var_images)
    var_images = np.sqrt(divide_with_respect(1, var_images))

    # Normalize the net exposure
    # sky_exposure *= 1 / np.nanmax(sky_exposure)

    if verbose:
        print("Creating final healpix mosaic table")

    # Next is to make the final table
    healpix_mosaic_table = Table(
        sky_images.T,
        names=[f"sky_e{i+1}" for i in range(en_chans)] + ["sky_etot"],
    )
    healpix_mosaic_table.add_columns(
        var_images,
        names=[f"var_e{i+1}" for i in range(en_chans)] + ["var_etot"],
    )
    healpix_mosaic_table.add_column(sky_exposure, name="exposure")
    healpix_mosaic_table.add_column(np.arange(npix), name="hpix_ind", index=0)

    return healpix_mosaic_table, metadata


def make_healpix_mosaics(
    outventory_file,
    start,
    end,
    dt=None,
    datadir=None,
    recalc=False,
    make_image=False,
    wcs=None,
    use_intermediate=False,
    parallel_grids=False,
    verbose=True,
):
    """
    Helper function to create mosaiced images on a healpix grid.
    The results are stored based on the requested time binning, so all 1 day mosaics are stored in
    a directory called 1_day_mosaics, and so on.
    It sums up all the BAT survey observations
    where:
     the partial coding images are multiplied by the exposure time of each image and summed
     the exposure images are directly summed
     the flux images are weighted by the inverse variance of the image and summed
     and the inverse variance images are summed together.

    :param outventory_file: Path object that provides the full outventory file of the BAT survey observations that will
        be used to calculate the mosaiced images.
    :param start: astropy Time of the start time of the time bin that survey observations need to be made to be included
        in that time bin's mosaiced image.
    :param end: astropy Time of the end time of the time bin that survey observations need to be made to be included
        in that time bin's mosaiced image.
    :param dt: number of days (in float) for which the mosaic is made, if None is given, it will calculate it from the
        start and end times. This will decide which folder to save the mosaics in.
    :param recalc: Boolean False by default. If this calculation was done previously, do not try to load the results of
        prior calculations. Instead recalculate the mosaiced images. The default, will cause the function to try to load
        a save file to save on computational time.
    :param make_image: Convert the healpix mosaics to image files (FITS) after making the mosaics on healpix grid.
        Default is False. If False, need to run convert_healpix_to_image separately.
    :param wcs: WCS object. Required if make_image is True. The WCS object provides the world coordinate system
    :param use_intermediate: Boolean False by default. If True, will use the exisitng mosaiced images if the existing
        mosaiced images are on a timescale that is integral multiple of the requested time bin. For example, if the user
        requests a 3 day mosaic, and there are existing 1 day mosaics, then those will be used to create the 3 day.
    :param parallel_grids: Boolean False by default. If True, will use parallel processing to convert healpix mosaics
        and project them onto multiple 2D sky images.
    :param verbose: Boolean True by default. Tells the code to print progress/diagnostic information.
    """

    if verbose:
        print(f"Working on time bins from {start} to {end}.\n")

    # get the name of the file with binned outventory info and where its saved

    start_time_str = convert_time_to_decimal_day(start)
    end_time_str = convert_time_to_decimal_day(end)

    savedir = outventory_file.parent.joinpath("grouped_outventory")
    output_file = savedir.joinpath(
        outventory_file.name.replace(".fits", f"_{start_time_str}_{end_time_str}.fits")
    )

    # Decide the directory where to store
    if dt is None:
        dt = np.round(end.mjd - start.mjd, 1)
        if dt % 1 == 0:
            dt = int(dt)

    # And look for this directory
    img_base_path = outventory_file.parent.joinpath(f"{dt}_day_mosaics")
    if not img_base_path.exists():
        img_base_path.mkdir(parents=True, exist_ok=True)

    # Now add day specific folder to this
    # this is the directory of the time bin where the images will be saved
    img_dir = img_base_path.joinpath(f"mosaic_{start_time_str}_{end_time_str}")
    if not img_dir.exists():
        img_dir.mkdir(parents=True, exist_ok=True)

    # see if there is a .batreproject file, if it doesnt exist or if we want to recalc things then go through the full loop
    if not img_dir.joinpath(".batreproject").exists() or recalc:

        if not use_intermediate:
            # Build mosaic images from scratch, i.e using 300s BAT exposures

            # Get all the obsids and pointing IDs that fall within the time bin
            grouped_outventory_data = Table.read(output_file)
            obsids = np.unique(grouped_outventory_data["OBS_ID"]).astype(
                str
            )  # But this is numpy str (unicode)
            # So convert them to python strings
            obsids = [str(i) for i in obsids]
            pointing_ids = [
                i.replace("point_", "") for i in grouped_outventory_data["IMAGE_ID"]
            ]

            # Get some meta data information
            tstart = np.min(grouped_outventory_data["TSTART"].data)
            tstop = np.max(grouped_outventory_data["TSTOP"].data)

            dateobs_start = Time(
                np.min(Time(grouped_outventory_data["DATE_OBS"]).mjd), format="mjd"
            ).isot
            dateobs_end = Time(
                np.max(Time(grouped_outventory_data["DATE_END"]).mjd), format="mjd"
            ).isot

            net_hdr_exposure = tstop - tstart
            # For each obsids, get the reprojected image
            # Which entails making a ReprojectSurvey object
            reproj_obj_list = [
                ReprojectSurvey(obsid, obs_dir=datadir) for obsid in obsids
            ]

            # Get the number of pointings
            # Only select those pointings that fall in the time bin
            reproj_pointings = []
            reproj_images = []
            for i in range(len(reproj_obj_list)):
                point_mask = np.isin(
                    reproj_obj_list[i].reprojected_pointings, pointing_ids
                )
                reproj_pointings = np.concatenate(
                    (
                        reproj_pointings,
                        np.array(reproj_obj_list[i].reprojected_pointings)[point_mask],
                    )
                )
                reproj_images = np.concatenate(
                    (
                        reproj_images,
                        np.array(reproj_obj_list[i].reprojected_images)[point_mask],
                    )
                )

            if verbose:
                print(
                    f"Found {len(reproj_pointings)} reprojected pointings, mosaicking them\n"
                )
            healpix_mosaic_table, metadata = _add_hpx_mosaics(
                reproj_images, verbose=verbose
            )
            # Now save the file
            outfile = img_dir.joinpath(
                f"mosaic_{start_time_str}_{end_time_str}_hybrid_healpix_projected.h5"
            )
            with h5py.File(outfile, "w") as f:
                # Store data
                if verbose:
                    print(f"Writing reprojected data to {outfile}")
                f.create_dataset(
                    "mosaic",
                    data=healpix_mosaic_table,
                    compression="gzip",
                    compression_opts=4,
                )

                # Store header as attributes
                f["mosaic"].attrs["nside"] = _nside
                f["mosaic"].attrs["TSTART"] = tstart
                f["mosaic"].attrs["TSTOP"] = tstop
                f["mosaic"].attrs["DATE-OBS"] = dateobs_start
                f["mosaic"].attrs["DATE-END"] = dateobs_end
                f["mosaic"].attrs["EXPOSURE"] = net_hdr_exposure
                f["mosaic"].attrs["EBINS"] = metadata["EBINS"]
                f.close()

        else:
            # Use existing intermediate mosaics to make the healpix mosaic
            if verbose:
                print("Using existing intermediate mosaics to make the healpix mosaic")
            existing_mosiac_dir = [
                i
                for i in img_base_path.parent.glob(f"*day_mosaics")
                if f"{dt}_day_mosaics" not in i.name
            ]
            existing_mosiac_dts = np.array(
                [d.name.split("_")[0] for d in existing_mosiac_dir]
            )
            if np.any(dt % existing_mosiac_dts.astype(float) == 0):

                dt_index = np.max(
                    np.where(dt % existing_mosiac_dts.astype(float) == 0)[0]
                )
                # Find the largest factor dt
                intermediate_dt = existing_mosiac_dts[dt_index]
                intermediate_mosaic_dir = existing_mosiac_dir[dt_index]
                if verbose:
                    print(
                        f"Using existing {intermediate_dt} day mosaics to make the {dt} day mosaic"
                    )
                # Select all the mosaics in this directory that fall within the time bin
                all_intermediate_mosaics = [
                    i
                    for i in intermediate_mosaic_dir.glob(
                        f"mosaic_*/mosaic*sky_image.fits"
                    )
                ]
                all_intermediate_mosaic_hdrs = [
                    fits.getheader(i, 1) for i in all_intermediate_mosaics
                ]
                all_intermediate_mosaic_tstarts = Time(
                    [h["DATE-OBS"] for h in all_intermediate_mosaic_hdrs], scale="tai"
                )
                all_intermediate_mosaic_tstops = Time(
                    [h["DATE-END"] for h in all_intermediate_mosaic_hdrs]
                )
                # Now select the ones that fall within the time bin
                mask = (all_intermediate_mosaic_tstarts >= start) & (
                    all_intermediate_mosaic_tstops <= end
                )

                intermediate_mosaics_to_use = np.array(all_intermediate_mosaics)[mask]

                # Only process these mosaics
                # intermediate_mosaic_hpx_files = [
                #     i.as_posix().replace(
                #         "_sky_image.fits", "_hybrid_healpix_projected.h5"
                #     )
                #     for i in intermediate_mosaics_to_use
                # ]
                intermediate_mosaic_hpx_files = []
                for i in intermediate_mosaics_to_use:
                    basename = i.as_posix()[: i.as_posix().index("_sky_grid_")]
                    intermediate_mosaic_hpx_files.append(
                        f"{basename}_hybrid_healpix_projected.h5"
                    )

                # Now start adding these intermediate mosaics
                final_mosaic_table, metadata = _add_hpx_mosaics(
                    intermediate_mosaic_hpx_files, verbose=verbose, altname=True
                )
                # Now save the file
                outfile = img_dir.joinpath(
                    f"mosaic_{start_time_str}_{end_time_str}_hybrid_healpix_projected.h5"
                )
                with h5py.File(outfile, "w") as f:
                    # Store data
                    if verbose:
                        print(f"Writing reprojected data to {outfile}")
                    f.create_dataset(
                        "mosaic",
                        data=final_mosaic_table,
                        compression="gzip",
                        compression_opts=4,
                    )

                    # Store header as attributes
                    f["mosaic"].attrs["nside"] = _nside
                    f["mosaic"].attrs["DATE-OBS"] = np.min(
                        all_intermediate_mosaic_tstarts[mask]
                    ).isot
                    f["mosaic"].attrs["DATE-END"] = np.max(
                        all_intermediate_mosaic_tstops[mask]
                    ).isot
                    f["mosaic"].attrs["EBINS"] = metadata["EBINS"]
                    f.close()
            else:
                raise ValueError(
                    "No existing intermediate mosaics found that are integral factors of the requested time bin"
                )

        # Then decide whether to make the final sky image
        if make_image:
            if wcs is None:
                raise ValueError("WCS object must be provided if make_image is True")
            else:
                convert_mosaic_to_sky_image_loop(
                    hpx_file=outfile, wcs=wcs, parallel=parallel_grids
                )

        img_dir.joinpath(".batreproject").touch()
    else:
        if verbose:
            print(f"Reprojected mosaic already exists in {img_dir}, not recalculating")


def convert_mosaic_to_sky_image(hpx_data, w, basepath, meta=None):
    """Helper function to convert the healpix map to a WCS sky image. The WCS
    need to have a defined coordinate system (equitorial/galactic), a defined projection
    sysmtem (TAN/ZEA/SIN), defined reference pixel and value and define projection matrix

    Args:
        hpx_data (h5py.file.File): The healpix map data that was created by sky projection.
        w (astropy.wcs.WCS): WCS object defining the target projection.
        basepath (str): The output file path where the sky image will be saved (this is just the basename)
        meta (dict, optional): Header keywords to be used for the output images.

    Returns:
        tuple: A tuple containing three astropy.io.fits.HDUList objects:
            - The first HDUList contains the sky images for each energy band and total energy.
            - The second HDUList contains the variance images for each energy band and total energy.
            - The third HDUList contains the normalized exposure image.
    """

    # Create an empyty primary header
    # Add all the key words back
    hdr = w.to_header()
    ebins = meta["EBINS"]
    # Next get the mean exposure for the sky map
    mean_exp = np.nanmedian(hpx_data["exposure"])

    for i in meta.keys():
        if i != "EBINS":
            hdr[i] = meta[i]
    hdr["EXPOSURE"] = (mean_exp, "Mean Exposure of the Healpix Mosaic [s]")

    sky_hdul = []
    var_hdul = []

    pri_hdu = fits.PrimaryHDU()
    sky_hdul.append(pri_hdu)
    var_hdul.append(pri_hdu)

    for i in range(ebins):
        # Make sky image for each band
        sky_image, _ = reproject_from_healpix(
            (hpx_data[f"sky_e{i+1}"], "icrs"), w, nested=False
        )
        sky_image_hdu = fits.ImageHDU(data=sky_image, header=hdr, name=f"BATIMAGE{i+1}")
        sky_hdul.append(sky_image_hdu)

        # Similarly make variance images
        var_image, _ = reproject_from_healpix(
            (hpx_data[f"var_e{i+1}"], "icrs"), w, nested=False
        )
        var_image_hdu = fits.ImageHDU(data=var_image, header=hdr, name=f"BATIMAGE{i+1}")
        var_hdul.append(var_image_hdu)

    # Make one for total energy
    sky_image, _ = reproject_from_healpix(
        (hpx_data["sky_etot"], "icrs"), w, nested=False
    )
    sky_image_hdu = fits.ImageHDU(data=sky_image, header=hdr, name=f"BATIMAGE_TOT")
    sky_hdul.append(sky_image_hdu)

    var_image, _ = reproject_from_healpix(
        (hpx_data[f"var_etot"], "icrs"), w, nested=False
    )
    var_image_hdu = fits.ImageHDU(data=var_image, header=hdr, name=f"BATIMAGE_TOT")
    var_hdul.append(var_image_hdu)

    # Next make one for exposure, but normalize it so that it can be used as a mask for
    # cell detections
    exp_image, _ = reproject_from_healpix(
        (hpx_data["exposure"] / np.nanmax(hpx_data["exposure"]), "icrs"),
        w,
        nested=False,
    )
    exp_hdul = [fits.PrimaryHDU(header=hdr, data=exp_image)]

    # Save sky images
    fits.HDUList(sky_hdul).writeto(
        basepath + "_sky_image.fits",
        overwrite=True,
    )

    # Save variance images
    fits.HDUList(var_hdul).writeto(
        basepath + "_var_image.fits",
        overwrite=True,
    )

    # Save exposure map
    fits.HDUList(exp_hdul).writeto(
        basepath + "_exp_image.fits",
        overwrite=True,
    )


def convert_mosaic_to_sky_image_loop(hpx_file, wcs, parallel=False):
    """Helper function to convert the healpix map to a WCS sky image. The WCS
    need to have a defined coordinate system (equitorial/galactic), a defined projection
    sysmtem (TAN/ZEA/SIN), defined reference pixel and value and define projection matrix

    Args:
        hpx_file (str): The healpix map that was created by sky projection.
        wcs (astropy.wcs.WCS): WCS object defining the target projection.
        parallel (bool, optional): Whether to use parallel processing. Defaults to False.

    Returns:
        numpy.ndarray: 2D array representing the projected sky image.
    """
    f = h5py.File(hpx_file, "r")
    data = f["mosaic"][()]

    meta = dict(f["mosaic"].attrs)

    # Now start reprojecting
    if type(wcs) == WCS:
        # Only a single wcs is passed
        # If not this means multiple sky wcs objects are passed, so we want to
        # reproject to each of them and save separate files
        # If this is the case, it should be a dict, with name that can identify
        # the grid and corresposnding wcs object
        wcs = {"global": wcs}
    # Make sky image, variance image, and exposure image for each wcs

    ngrids = len(wcs)

    # Make output file names
    if ngrids == 1:
        # dont worry about name, this is a global skygrid
        # Save all these files
        basepath = Path(hpx_file).expanduser().resolve().as_posix()
        filenames = basepath.replace("_hybrid_healpix_projected.h5", "")
    else:
        basepath = Path(hpx_file).expanduser().resolve().as_posix()
        filenames = [
            basepath.replace("hybrid_healpix_projected.h5", i.replace(".fits", ""))
            for i in wcs.keys()
        ]
    filenames = dict(zip(list(wcs.keys()), filenames))

    if (ngrids > 1) and parallel:
        # Run the grids in parallel using python joblib
        Parallel(n_jobs=ngrids)(
            delayed(convert_mosaic_to_sky_image)(
                data,
                wcs[sgrid],
                filenames[sgrid],
                meta,
            )
            for sgrid in list(wcs.keys())
        )
    else:
        for sgrid in list(wcs.keys()):
            convert_mosaic_to_sky_image(
                data,
                wcs[sgrid],
                filenames[sgrid],
                meta,
            )


def merge_mosaics(intermediate_mosaic_dir_list, savedir=None):
    """
    Merges the intermediate mosaic images from a number of previously calculated mosaic images for a set of time bins.
    The intermediate mosaic images must exist for this function to work.

    :param intermediate_mosaic_dir_list: A list of the directories with mosaic images that will be added together.
    :param savedir: None or a Path object. None creates a cirectory called "total_mosaic" in the parent directory of the
        directory given by intermediate_mosaic_dir_list[0]
    :return: Path object of the directory that holds the resulting intermediate mosaic images
    """
    # this goes through the various intermediate mosaic files and adds them up

    # create the directory that will hold the total mosaiced images,
    # get the directory to create the folder that we will put the images
    if savedir is None:
        savedir = intermediate_mosaic_dir_list[0].parent
        total_dir = savedir.joinpath("total_mosaic")
    else:
        total_dir = savedir

    dirtest(total_dir)

    # create the arrays that will hold all the data
    ra_skygrid = read_skygrids()[0]
    eimg = np.zeros_like(
        ra_skygrid
    )  # exposure map, has the same dimensions as the skygrid
    pimg = np.zeros_like(ra_skygrid)  # Partial coding map
    nx, ny, nz = ra_skygrid.shape
    vimg = np.zeros(
        (nx, ny, nz, _nebands + 1)
    )  # Variance map, size of skygrid with extra enegy dimension (+1 for total 14-195 band)
    simg = np.zeros_like(vimg)  # Sky flux  image
    total_binned_exposure = 0  # tally up the total exposure
    total_tstart = []
    total_tstop = []
    total_dateobs_start = []
    total_dateobs_end = []
    user_met_tbin_start = []
    user_met_tbin_end = []

    # loop over the directories to read files and add them
    for i in intermediate_mosaic_dir_list:
        # loop over each sky facet
        for j in range(nz):
            string = "c%d_%s" % (j, _proj)

            # open the pimg and add it to the array and accumulate the exposure and other header info
            pimg_file = i.joinpath("pcode_" + string + ".img")
            with fits.open(str(pimg_file)) as file:
                # read the partial coding map
                pimg[:, :, j] += file[0].data
                # for the first sky facet obtain this info. all sky facets have this info so we only need it from one
                # sky facet
                if j == 0:
                    total_binned_exposure += file[0].header["EXPOSURE"]
                    total_tstart.append(file[0].header["TSTART"])
                    total_tstop.append(file[0].header["TSTOP"])
                    total_dateobs_start.append(file[0].header["DATE-OBS"])
                    total_dateobs_end.append(file[0].header["DATE-END"])
                    user_met_tbin_start.append(file[0].header["S_TBIN"])
                    user_met_tbin_end.append(file[0].header["E_TBIN"])

            # open the eimg and add it to the array and accumulate the exposure
            eimg_file = i.joinpath("expmap_" + string + ".img")
            with fits.open(str(eimg_file)) as file:
                # read the flat exposure map
                eimg[:, :, j] += file[0].data

            # open the vimg and flux files
            simg_file_name = i.joinpath("flux_" + string + ".img")
            vimg_file_name = i.joinpath("var_" + string + ".img")

            simg_file = fits.open(str(simg_file_name))
            vimg_file = fits.open(str(vimg_file_name))

            # loop over the enegy bands for variance and flux
            for k in range(vimg.shape[-1]):
                # add the fluxes and the variances
                vimg[:, :, j, k] += vimg_file[k].data
                simg[:, :, j, k] += simg_file[k].data

            simg_file.close()
            vimg_file.close()

    # after adding everything up, need to save the data to save time/effort just copy over some of the intermediate
    # files from the 0th directory of the list that is passed in, into the total_mosaic directory and update these
    # data/header values

    tmin = np.min(total_tstart)
    tmax = np.max(total_tstop)
    dt = np.max(total_tstop) - np.min(total_tstart)
    obs_min = total_dateobs_start[np.argmin(total_tstart)]
    obs_max = total_dateobs_end[np.argmax(total_tstop)]
    user_tbin_start = np.min(user_met_tbin_start)
    user_tbin_end = np.max(user_met_tbin_end)

    # loop over each sky facet
    for j in range(nz):
        string = "c%d_%s" % (j, _proj)

        # do this for the exposure
        file_name = intermediate_mosaic_dir_list[0].joinpath(f"expmap_{string}.img")
        output_name = total_dir.joinpath(f"expmap_{string}.img")
        shutil.copy(file_name, output_name)

        with fits.open(str(output_name), mode="update") as file:
            file[0].data = eimg[:, :, j]
            header = file[0].header
            header["TSTART"] = (tmin, " start time of image")
            header["TSTOP"] = (tmax, " stop time of image")
            header["TELAPSE"] = (dt, "  elapsed time of image (= TSTOP-TSTART)")
            header["DATE-OBS"] = (obs_min, "  TSTART, expressed in UTC")
            header["DATE-END"] = (obs_max, "  TSTOP, expressed in UTC")
            header["EXPOSURE"] = (
                total_binned_exposure,
                "[sec.] Sum of pointing exposures used",
            )
            header["S_TBIN"] = (user_tbin_start, "Mosaicing Start of Time Bin (MET)")
            header["E_TBIN"] = (user_tbin_end, "Mosaicing End of Time Bin (MET)")
            file.flush()

        # do this for the pcode
        file_name = intermediate_mosaic_dir_list[0].joinpath(f"pcode_{string}.img")
        output_name = total_dir.joinpath(f"pcode_{string}.img")
        shutil.copy(file_name, output_name)

        with fits.open(str(output_name), mode="update") as file:
            file[0].data = pimg[:, :, j]
            header = file[0].header
            header["TSTART"] = (tmin, " start time of image")
            header["TSTOP"] = (tmax, " stop time of image")
            header["TELAPSE"] = (dt, "  elapsed time of image (= TSTOP-TSTART)")
            header["DATE-OBS"] = (obs_min, "  TSTART, expressed in UTC")
            header["DATE-END"] = (obs_max, "  TSTOP, expressed in UTC")
            header["EXPOSURE"] = (
                total_binned_exposure,
                "[sec.] Sum of pointing exposures used",
            )
            header["S_TBIN"] = (user_tbin_start, "Mosaicing Start of Time Bin (MET)")
            header["E_TBIN"] = (user_tbin_end, "Mosaicing End of Time Bin (MET)")
            file.flush()

        # copy files for the variability and flux
        file_name = intermediate_mosaic_dir_list[0].joinpath(f"var_{string}.img")
        var_output_name = total_dir.joinpath(f"var_{string}.img")
        shutil.copy(file_name, var_output_name)

        file_name = intermediate_mosaic_dir_list[0].joinpath(f"flux_{string}.img")
        flux_output_name = total_dir.joinpath(f"flux_{string}.img")
        shutil.copy(file_name, flux_output_name)

        # open all the files in update mode
        flux_file = fits.open(str(flux_output_name), mode="update")
        var_file = fits.open(str(var_output_name), mode="update")

        # iterating over the energies
        for k in range(len(flux_file)):
            # update the flux
            flux_file[k].data = simg[:, :, j, k]
            header = flux_file[k].header
            header["TSTART"] = (tmin, " start time of image")
            header["TSTOP"] = (tmax, " stop time of image")
            header["TELAPSE"] = (dt, "  elapsed time of image (= TSTOP-TSTART)")
            header["DATE-OBS"] = (obs_min, "  TSTART, expressed in UTC")
            header["DATE-END"] = (obs_max, "  TSTOP, expressed in UTC")
            header["EXPOSURE"] = (
                total_binned_exposure,
                "[sec.] Sum of pointing exposures used",
            )
            header["S_TBIN"] = (user_tbin_start, "Mosaicing Start of Time Bin (MET)")
            header["E_TBIN"] = (user_tbin_end, "Mosaicing End of Time Bin (MET)")
            flux_file.flush()

            # update the variance
            var_file[k].data = vimg[:, :, j, k]
            header = var_file[k].header
            header["TSTART"] = (tmin, " start time of image")
            header["TSTOP"] = (tmax, " stop time of image")
            header["TELAPSE"] = (dt, "  elapsed time of image (= TSTOP-TSTART)")
            header["DATE-OBS"] = (obs_min, "  TSTART, expressed in UTC")
            header["DATE-END"] = (obs_max, "  TSTOP, expressed in UTC")
            header["EXPOSURE"] = (
                total_binned_exposure,
                "[sec.] Sum of pointing exposures used",
            )
            header["S_TBIN"] = (user_tbin_start, "Mosaicing Start of Time Bin (MET)")
            header["E_TBIN"] = (user_tbin_end, "Mosaicing End of Time Bin (MET)")
            var_file.flush()

        flux_file.close()
        var_file.close()

    return total_dir


# Need to make sure that the sky facets have been created when this is imported
# if this is the first time its imported need to create the default ones
# if it's not the first time this module is imported (ie that batanalysis is imported) then dont redo this calculation

package_data_dir = Path(__file__).parent.joinpath("data")
zea_proj = sorted(package_data_dir.glob("*_ZEA.img"))
grid_completion = (
    False if (len(zea_proj) < 2 * _nskyimg) else True
)  # the 2*_nskyimg is for the fact that we need skyfacets for RA and for Dec

if not grid_completion:
    print("Initalizing the BatAnalysis package")
    make_skygrids()  # For ZEA projection
    print("Completed initalizing the package")
