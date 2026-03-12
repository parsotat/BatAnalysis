"""
This file is meant to hold the functions that allow users to reproject the survey images on to a healpix grid
"""

import sys
import h5py
import pickle
import shutil
import numpy as np
import healpy as hp
from pathlib import Path
import subprocess as subp
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from sklearn.cluster import DBSCAN
from astropy.nddata import Cutout2D
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, Column
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_pdf import PdfPages


from .mosaic import (
    interp_weights,
    interpolate,
    read_correctionsmap,
    compute_statistics_map,
)

from .bat_survey import BatSurvey
from .batlib import datadir, calculate_effective_snr

# for python>3.6
try:
    import heasoftpy.swift as hsp
    import heasoftpy.utils as hsp_util
    from heasoftpy import heatools
except ModuleNotFoundError as err:
    # Error handling
    print(err)

import swiftbat.swutil as sbu

# Off-axis flux correction file
# _cimgfile = "offaxiscorr_8bin_20061221.img"
# _chilothresh = 0.50  # Minimum chi-square for any energy band
# _chihithresh = 1.15  # Maximum chi-square for any energy band
# _chiscobump = 0.35  # Additional bump of chi-square threshold around Sco X-1 (band 0)
# _chiscotheta = 30  # Approximate angular scale of bump around Sco X-1 (deg)
_pcodethresh = 0.05  # Minimum image partial coding
_minexpo = 150  # Minimum image exposure
_nebands = 8  # Number of energy bands to process
_proj = "ZEA"  # projection from idl code that is used
_order = 10  # Healpix order parameter
_nside = 2**_order  # Healpix nside parameter

# Sco X-1 ra and dec
# _scox1_ra = 245.100
# _scox1_dec = -15.600
# _sco_coord = SkyCoord(_scox1_ra, _scox1_dec, frame="icrs", unit="deg")


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


def get_all_sky_wcs_grids(savedirectory=None):
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

    all_files = [
        i.as_posix() for i in dir.joinpath("data").glob("ra_c*_ZEA.img")
    ]  # get the size of the first one for us to allocate an array for

    all_files.sort()
    all_sky_zea_grids = {}
    for i, file in enumerate(all_files):
        hdr = fits.getheader(file)
        grid_l = hdr["CRVAL1"]
        grid_b = hdr["CRVAL2"]
        grid_name = f"sky_grid_{i+1:02d}_l_{grid_l:03.0f}_b_{grid_b:+03.0f}_ZEA.fits"
        all_sky_zea_grids[grid_name] = WCS(hdr)
    return all_sky_zea_grids


def convert_time_to_mettime(time, reverse=False):
    """Helper function to convert ISOT time to Swift mission time

    Args:
        time (str): ISOT time
        reverse (bool, optional): convert isot to met time or reverse. Defaults to False.
    """
    reftime = Time("2001-01-01")
    if not reverse:
        time = Time(time)
        return (time - reftime).to(u.second).value
    else:
        dt = time * u.second
        return reftime + dt


def divide_with_respect(a, b, fill_value=np.nan):
    """Helper function that divides two arrays but takes care of divide by zero errors

    Args:
        a (np.ndarray): The numerator array
        b (np.ndarray): The denominator array

    Returns:
        np.ndarray: The result of the division with divide by zero handled
    """
    result = np.divide(a, b, out=np.ones_like(b) * fill_value, where=b != 0)
    return result


def add_with_respect(a, b, fill_value=np.nan):
    """Helper function that adds two arrays taing care of nans

    Args:
        a (np.ndarray): The numerator array
        b (np.ndarray): The denominator array
        fill_value (float, optional): Value to fill. Defaults to np.nan.


    Returns:
        np.ndarray
    """
    empty_mask = np.isnan(a) & np.isnan(b)
    summed_array = np.nansum([a, b], axis=0)
    summed_array[empty_mask] = fill_value
    return summed_array


def get_ang_from_hpx_grid():
    """Helper function that creates the healpix grids that will be used for mosaicing.
    By defualt it uses _nside parameter to decide the resolution of the grid. The morale is to
    project the image onto a healpix map so that storage can be efficient
    """
    pix_ids = np.arange(hp.nside2npix(_nside)).astype(int)
    theta, phi = hp.pix2ang(_nside, pix_ids)
    ra = np.rad2deg(phi)
    dec = 90 - np.rad2deg(theta)
    return pix_ids, ra, dec


def get_common_healpix_grid(wcs):
    """By defualt the healpiz grid is all sky, but we dont want that. We want what is common
    to the image, so given a header file, make a WCS and get the common area.

    Args:
        wcs (astropy.wcs.WCS): WCS object created from the image header
    """
    pix_ids, ra_hp, dec_hp = get_ang_from_hpx_grid()
    coords = SkyCoord(ra_hp, dec_hp, unit="deg")
    good_pix_ind = wcs.footprint_contains(coord=coords)
    pixel_x, pixel_y = wcs.world_to_pixel(coords)

    # good_pix_ind = (
    #     (pixel_x >= -1)
    #     & (pixel_x < wcs.array_shape[1])
    #     & (pixel_y >= -1)
    #     & (pixel_y < wcs.array_shape[0])
    # )
    return (
        pixel_x[good_pix_ind],
        pixel_y[good_pix_ind],
        pix_ids[good_pix_ind],
    )


def convert_time_to_decimal_day(times, precision=3):
    """
    Converts an array of astropy Time objects to decimal days

    :param times: an array of astropy Time objects
    :param precision: an integer that denotes the number of decimal places to round the decimal day value to
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
        precision,
    ).astype(str)
    decimal_day = [str(int(float(i))) if float(i) % 1 == 0 else i for i in decimal_day]

    strfmt = f"0{precision+3}.{precision}f" if precision > 0 else f"02.0f"
    output_str = [
        f"{y}_{m.rjust(2, '0')}_{float(d):{strfmt}}"
        for y, m, d in zip(year, month, decimal_day)
    ]
    if nelements == 0:
        return output_str[0]
    else:
        return output_str


def project_maps(img, var, verbose=True):
    """Helper function to project the sky image and variance map on to a healpix grid.
    In principle this is supposed to project the sky image and variance map on to a common
    healpix grid that can be used for mosaicing later on. To make the additions easier later on,
    as the additions are noise weighted, the sky image is actually stored as sky/var^2
    and the variance map as 1/var^2. But even in this step, making sky images should be fine
    since, the projection on to a 2D images does the division and hence true sky images are stored.

    Args:
        img (astropy.io.fits.ImageHDU): The sky image HDU
        var (astropy.io.fits.ImageHDU): The variance map HDU
        verbose (bool, optional): Whether to print out information during processing. Defaults to True.
    """
    is_truncated = []
    with fits.open(img.as_posix()) as img_hdu, fits.open(var.as_posix()) as var_hdu:
        # read the partial coding map

        pcoding_image = img_hdu["BAT_PCODE_1"].data
        # save the header for use later
        pcoding_header = img_hdu[0].header
        pcoding_wcs = WCS(pcoding_header)

        pointing_exposure = pcoding_header["EXPOSURE"]
        pointing_tstart = pcoding_header["TSTART"]
        pointing_tstop = pcoding_header["TSTOP"]
        pointing_dateobs_start = pcoding_header["DATE-OBS"]
        pointing_dateobs_end = pcoding_header["DATE-END"]

        # get the image size and create array to hold the sky flux at each channel
        sz = pcoding_image.shape

        # Make the sky image
        sky_image = np.zeros(
            (sz[0], sz[1], _nebands + 1)
        )  # plus 1 for the total energy

        # Make the sky variance image
        var_image = np.zeros_like(sky_image)
        for k in range(_nebands):
            bin_trunc_mask = img_hdu[k].header.get("TRUNCATED", False)
            is_truncated.append(bin_trunc_mask)

            # If it is truncated, make sure image is 0, so is var
            if bin_trunc_mask:
                sky_image[:, :, k] = np.zeros_like(img_hdu[k].data)
                var_image[:, :, k] = np.zeros_like(var_hdu[k].data)
            else:
                sky_image[:, :, k] = img_hdu[k].data
                var_image[:, :, k] = var_hdu[k].data

    # Also read in corrections map
    corrections_map = read_correctionsmap()

    # correct for off axis effects
    sky_image[:, :, :-1] = divide_with_respect(
        a=sky_image[:, :, :-1], b=corrections_map, fill_value=np.nan
    )

    var_image[:, :, :-1] = divide_with_respect(
        a=var_image[:, :, :-1], b=corrections_map, fill_value=np.nan
    )

    # construct the total energy images for variance and flux, the zeros in last array dont affect
    # calculations of the total values
    # Fluxes add in a simple way
    sky_image[:, :, -1] = np.sum(sky_image, axis=2)

    # # Variances add in quadrature
    var_image[:, :, -1] = np.sqrt(np.sum(var_image**2, axis=2))

    # construct the quality map for each energy and for the total energy images
    energy_quality_mask = np.zeros_like(sky_image)
    good_idx = np.where(
        (
            np.repeat(
                pcoding_image[:, :, np.newaxis],
                sky_image.shape[-1],
                axis=2,
            )
            > _pcodethresh
        )
        & (var_image > 0)
        & np.isfinite(sky_image)
        & np.isfinite(var_image)
    )
    energy_quality_mask[good_idx] = 1

    # Now do the actual sky projection using reproject. This will project the sky image
    # onto a healpix grid. But nan handling isn't great in reproject so we will do this
    # manually here by filling zeros

    # The easiest (and efficient) way (for downstream purposes) is to use
    # healpix for the reprojection. We will use the reproject package

    # get the good values, in the idl file the shape of pointing_pimg is reversed, not sure if
    # this is correct here.
    # Multiplying the energy quality mask to the sky and variance images makes the masked regions
    # 0s which makes the interpolation tricky as it will average 0's and actual values together.
    # This will produce hotspots of islands with very low variance and hence false positives in source
    # detection, rather make them nan and handle them during interpolation
    # sky_image *= energy_quality_mask
    # var_image *= energy_quality_mask

    # sky_image[np.isnan(sky_image)] = 0
    # var_image[np.isnan(var_image)] = 0

    sky_image[energy_quality_mask == 0] = np.nan
    var_image[energy_quality_mask == 0] = np.nan

    if verbose:
        print("Getting common healpix grid\n")
    # get the common healpix grid

    if verbose:
        print("Calculating pixel coordinates for reprojection\n")

    pixel_x, pixel_y, ipix = get_common_healpix_grid(wcs=pcoding_wcs)

    interp_at_points = np.array([pixel_y, pixel_x])

    # need to interpolate the survey sky image onto the all sky image
    # need to verify that the eimg and pimg maps are energy independent, in idl code only does this
    # for te first energy iteration
    grid_x, grid_y = np.mgrid[0 : pcoding_image.shape[0], 0 : pcoding_image.shape[1]]
    points = np.array([grid_x.flatten(), grid_y.flatten()])

    # see if thie other method works,
    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    if verbose:
        print("Calculating weights and vertices for sky projection\n")
    vtx, wts = interp_weights(points.T, interp_at_points.T)

    # Initialize a table
    hp_pixel_inds = np.arange(hp.nside2npix(_nside)).astype(int)
    tab = Table([hp_pixel_inds], names=["hpix_ind"])

    for k in range(_nebands + 1):
        if k == _nebands:
            name_tag = "tot"
        else:
            name_tag = f"{k+1}"
        sky_val_hpx = np.zeros(len(hp_pixel_inds)) * np.nan
        var_val_hpx = np.zeros(len(hp_pixel_inds)) * np.nan

        values = sky_image[:, :, k]
        if verbose:
            print(f"Reprojecting sky map of energy band {k+1} of {_nebands}")
        sky_values = interpolate(values.flatten(), vtx, wts, fill_value=np.nan)

        # Do the same for variance maps
        values = var_image[:, :, k]
        if verbose:
            print(f"Reprojecting variance map of energy band {k+1} of {_nebands}")
        var_values = interpolate(values.flatten(), vtx, wts, fill_value=np.nan)

        # These values are at the values of indices ipix
        # so populate those
        sky_val_hpx[ipix] = divide_with_respect(
            sky_values, var_values**2, fill_value=np.nan
        )
        tab.add_column(sky_val_hpx, name=f"sky_e{name_tag}")

        var_val_hpx[ipix] = divide_with_respect(1, var_values**2, fill_value=np.nan)
        tab.add_column(var_val_hpx, name=f"var_e{name_tag}")

    # Do the same for the partial coding fractions
    if verbose:
        print(f"Reprojecting partial coding map")
    hpx_pcode_image = np.zeros(len(hp_pixel_inds)) * np.nan

    exposure_image = pcoding_image * energy_quality_mask[:, :, 0] * pointing_exposure
    values = exposure_image.flatten()
    # values[np.isnan(values)] = 0
    test = interpolate(values, vtx, wts, fill_value=np.nan)
    hpx_pcode_image[ipix] = test

    tab.add_column(hpx_pcode_image, name="exposure")

    # Add in metadata
    tab.meta["nside"] = _nside
    tab.meta["TSTART"] = pointing_tstart
    tab.meta["TSTOP"] = pointing_tstop
    tab.meta["DATE-OBS"] = pointing_dateobs_start
    tab.meta["DATE-END"] = pointing_dateobs_end
    tab.meta["EXPOSURE"] = pointing_exposure
    tab.meta["TELAPSE"] = pointing_tstop - pointing_tstart
    tab.meta["EBINS"] = _nebands
    tab.meta["TRUNCATED"] = is_truncated

    return tab


def do_sky_reprojection(obsdir, verbose=True):
    """Helper function to reproject the given sky image on to a global grid.
    This will be useful for creating all sky mosaics. By default, all one needs
    is a directory containing the sky images to be reprojected. This assumes that
    they are already processed using the batsurvey module in BatAnalysis as it will
    try to read the pickle file that is created when the batsurvey object is saved.

    Args:
        :param obsdir (str): The path to the directory containing the raw data
        :param verbose (bool, optional): Whether to print out information during processing. Defaults to True.
    """
    # First make the survey object
    # parse the obsid from the obsdir
    obs_dir = Path(obsdir)
    survey_obj = BatSurvey(obs_id=obs_dir.name, obs_dir=obs_dir.parent, recalc=False)

    # Get some basic parameters about pointing
    obsid = survey_obj.obs_id
    emin = survey_obj.emin
    emax = survey_obj.emax

    ncleaniter = survey_obj.batsurvey_result.params["ncleaniter"]

    # Using this read the stats file of interest
    stat_file = survey_obj.result_dir.joinpath("stats_point.fits")
    stats_data = fits.getdata(stat_file.as_posix(), 1)

    # Only proceed of there are good pointings
    if len(stats_data["NBATDETS"]) > 0:
        # pointing_ids = np.array(survey_obj.pointing_ids)
        pointing_ids = np.array(
            [i.replace("point_", "") for i in stats_data["IMAGE_ID"]]
        )
        processed_mask = np.zeros(len(pointing_ids)).astype(bool)
        try:
            truncated_mask = stats_data["TRUNCATED_MASK"]
        except:
            truncated_mask = np.zeros_like(stats_data["CHI2"]).astype(bool)

        # calculate the mask of which points we should use based on good image statistics
        chi_mask = compute_statistics_map(
            stats_data["CHI2"],
            stats_data["NBATDETS"],
            stats_data["RA_PNT"],
            stats_data["DEC_PNT"],
            stats_data["PA_PNT"],
            stats_data["TSTART"],
            truncated=truncated_mask,
            avoid_sco=False,
        )

        # test that we have good image statistics for every pointing
        for i in range(len(stats_data["NBATDETS"])):
            pointing_id = pointing_ids[i]
            bad_image_mask = (
                (chi_mask[i] == 0)
                | (stats_data["NBATDETS"][i] <= 0)
                | (stats_data["IMAGE_STATUS"][i] == False)
                | (stats_data["EXPOSURE"][i] <= 0)
                | (pointing_id not in survey_obj.pointing_ids)
            )
            if bad_image_mask:
                if verbose:
                    print(
                        f"Bad image Statistics. Skipping observation ID/Pointing: {obsid}/{pointing_id}\n"
                    )
                processed = False
            else:
                if verbose:
                    print(
                        f"Good image Statistics. Working on observation ID/Pointing: {obsid}/{pointing_id}\n"
                    )

                data_directory = survey_obj.result_dir.joinpath(f"point_{pointing_id}")
                # read the partial coding map, variance map, sky flux map for the pointing
                pointing_img_str = data_directory.joinpath(
                    f"point_{pointing_id}_{ncleaniter}.img"
                )
                pointing_var_str = data_directory.joinpath(
                    f"point_{pointing_id}_{ncleaniter}.var"
                )

                # Then read all the information about the pointing
                hdr = fits.getheader(pointing_img_str.as_posix(), ext=0)

                # get other header information
                pointing_exposure = hdr["EXPOSURE"]

                # Only proceed, if it exceeds a minimum exposure criterion
                if pointing_exposure >= _minexpo:
                    # Then read all the information about the pointing
                    tab = project_maps(
                        img=pointing_img_str, var=pointing_var_str, verbose=verbose
                    )
                    outfile = data_directory.joinpath(
                        f"point_{pointing_id}_{ncleaniter}_hybrid_healpix_projected.h5"
                    )
                    with h5py.File(outfile, "w") as f:
                        # Store data
                        if verbose:
                            print(f"Writing reprojected data to {outfile}")
                        f.create_dataset(
                            "mosaic", data=tab, compression="gzip", compression_opts=4
                        )

                        # Store header as attributes
                        for m in tab.meta.keys():
                            f["mosaic"].attrs[m] = tab.meta[m]
                        f["mosaic"].attrs["obsid"] = obsid
                        f["mosaic"].attrs["pointing_id"] = str(pointing_id)
                        f["mosaic"].attrs["emin"] = emin
                        f["mosaic"].attrs["emax"] = emax
                        f.close()
                    processed = True
                else:
                    processed = False
            processed_mask[i] = processed
        return obsid, pointing_ids[processed_mask]
    else:
        return obsid, np.array([])


def _add_hpx_mosaics(hpx_file_list, verbose=True, en_chans=8):
    """
    Helper function to add healpix mosaics together. But this only generates indermediate data products.
    The mosaic is calculated by doing a noise weighted average of the sky images and variance maps.
    However to use it on intermediate data products, the sky image is left as numerator and the variance map
    is left as denominator squared. This way when adding multiple images, the final division can be done
    at a later stage. The final division is always done in the final projection, at the convert_mosaic_to_sky_image
    function.

    :param hpx_file_list: list of Path objects that provide the healpix mosaic files to be added together.
    :param verbose: Boolean True by default. Tells the code to print progress/diagnostic information.
    :param altname: Boolean False by default. If True, use alternative naming convention for energy bands.
        By default they are names e0 to e8. with e8 being total energy, if alternate naming convention is used
        they will be names e1 to e8, with e_tot being total energy band.
    :param en_chans: Number of energy bins. In case it is missing from header, this will be used.
    :return: a h5py file
    """

    # Now read each file and add them
    # Instatiate an empty healpix grid
    # To do this read a sample h5py file

    if len(hpx_file_list) > 0:

        npix = hp.nside2npix(nside=_nside)

        # Get some meta data information
        tmin = 1e15
        tmax = 0

        metadata = {}
        metadata["EBINS"] = en_chans
        metadata["nside"] = _nside

        # Make empty sky images and variance maps
        sky_images = np.zeros((en_chans + 1, npix))  # +1 for total energy band
        var_images = np.zeros_like(sky_images)

        sky_exposure = np.zeros(npix)
        all_pointing_truncated_ebins = []
        emin = None
        emax = None
        for i in range(len(hpx_file_list)):
            if verbose:
                print(f"Adding pointing {i+1} of {len(hpx_file_list)}")
            ind_pointing_obj = h5py.File(hpx_file_list[i], "r")

            # Read in time inormation to update metadata
            tmin = np.min([tmin, ind_pointing_obj["mosaic"].attrs["TSTART"]])
            tmax = np.max([tmax, ind_pointing_obj["mosaic"].attrs["TSTOP"]])

            # Try to get the energy bins
            if emin is None:
                if "emin" in ind_pointing_obj["mosaic"].attrs:
                    emin = ind_pointing_obj["mosaic"].attrs["emin"]
                else:
                    emin = [14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0]
            if emax is None:
                if "emax" in ind_pointing_obj["mosaic"].attrs:
                    emax = ind_pointing_obj["mosaic"].attrs["emax"]
                else:
                    emax = [20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0]

            # Then try to get truncated information to update metadata
            # If these are individual pointing mosaics, they should have this information
            # from the TRUNCATED keyword, but if they are already mosaics of multiple pointings,
            # they will have this from the ALLTRUNC keyword, or ANYTRUNC keyword,
            # but if they are missing both, we will assume that there are no truncated
            # energy bins for that pointing
            if "TRUNCATED" in ind_pointing_obj["mosaic"].attrs:
                truncated_ebins = ind_pointing_obj["mosaic"].attrs["TRUNCATED"]
            elif "ANYTRUNC" in ind_pointing_obj["mosaic"].attrs:
                truncated_ebins = ind_pointing_obj["mosaic"].attrs["ANYTRUNC"]
            else:
                truncated_ebins = np.zeros(en_chans).astype(bool)
            all_pointing_truncated_ebins.append(truncated_ebins)

            ind_pointing_data = ind_pointing_obj["mosaic"][()]

            # Sky images are named sky_e1 and so on
            for ein in range(en_chans + 1):
                ename = ein + 1 if ein < en_chans else "tot"
                sky_images[ein, :] = add_with_respect(
                    sky_images[ein, :], ind_pointing_data[f"sky_e{ename}"]
                )
                var_images[ein, :] = add_with_respect(
                    var_images[ein, :], ind_pointing_data[f"var_e{ename}"]
                )

            # Next get the net exposure
            sky_exposure = add_with_respect(sky_exposure, ind_pointing_data["exposure"])

        if verbose:
            print("Mosaicking them")
        # Now convert them all to proper units
        # sky_images = divide_with_respect(sky_images, var_images)
        # var_images = np.sqrt(divide_with_respect(1, var_images))

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

        # Update metadata
        metadata["TSTART"] = tmin
        metadata["TSTOP"] = tmax
        metadata["TELAPSE"] = tmax - tmin
        metadata["DATE-OBS"] = convert_time_to_mettime(tmin, reverse=True).isot
        metadata["DATE-END"] = convert_time_to_mettime(tmax, reverse=True).isot
        metadata["EXPOSURE"] = np.nanmax(sky_exposure)
        metadata["EMIN"] = emin
        metadata["EMAX"] = emax
        metadata["ALLTRUNC"] = np.all(all_pointing_truncated_ebins, axis=0)
        metadata["ANYTRUNC"] = np.any(all_pointing_truncated_ebins, axis=0)
    else:
        healpix_mosaic_table = Table()
        metadata = {}

    return healpix_mosaic_table, metadata


def make_healpix_mosaics(
    outventory_file,
    start,
    end,
    dt=None,
    obsdir=None,
    recalc=False,
    make_image=False,
    wcs=None,
    use_intermediate=False,
    snr=5,
    plot_dest_dir=None,
    plot_all_sources=True,
    plot_individual_sources=False,
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
    :param snr: Signal to noise ratio threshold for source detection when making images from healpix mosaics.
    :param plot_dest_dir: Path object or string that provides the directory where the diagnostic plots
        will be saved when making images from healpix mosaics. If None is given, no plots are made.
    :param plot_all_sources: Boolean True by default. If making images from healpix mosaics, this tells the code
        to make diagnostic plots for all detected sources.
    :param plot_individual_sources: Boolean False by default. If making images from healpix mosaics, this tells the code
        to make diagnostic plots for individual sources.
    :param verbose: Boolean True by default. Tells the code to print progress/diagnostic information.
    """

    if verbose:
        print(f"Working on time bins from {start} to {end}.\n")

    # get the name of the file with binned outventory info and where its saved

    if dt < 1:
        precision = 1
    else:
        precision = 0

    start_time_str = convert_time_to_decimal_day(start, precision=precision)
    end_time_str = convert_time_to_decimal_day(end, precision=precision)

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

        metadata = {}
        metadata["nside"] = _nside
        metadata["MOSAIC_TSTART"] = start.isot
        metadata["MOSAIC_TEND"] = end.isot
        metadata["mosaic_dt"] = dt

        if not use_intermediate:
            # Build mosaic images from scratch, i.e using 300s BAT exposures

            # Get all the obsids and pointing IDs that fall within the time bin
            grouped_outventory_data = Table.read(output_file)

            # Only proceed if there are pointings in this time bin
            if len(grouped_outventory_data) > 0:

                obsids = np.unique(grouped_outventory_data["OBS_ID"]).astype(str)

                # So convert them to python strings
                obsids = [str(i) for i in obsids]
                pointing_ids = [
                    i.replace("point_", "") for i in grouped_outventory_data["IMAGE_ID"]
                ]

                # For each obsids, get the reprojected image
                # Which entails making a ReprojectSurvey object
                reproj_obj_list = [
                    ReprojectSurvey(obsid, obs_dir=obsdir, recalc=False)
                    for obsid in obsids
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
                            np.array(reproj_obj_list[i].reprojected_pointings)[
                                point_mask
                            ],
                        )
                    )
                    reproj_images = np.concatenate(
                        (
                            reproj_images,
                            np.array(reproj_obj_list[i].reprojected_images)[point_mask],
                        )
                    )
                metadata["obsids"] = obsids
                metadata["pointing_ids"] = reproj_pointings
                metadata["images"] = reproj_images
                metadata["DATEOBS_START"] = start.mjd
                metadata["DATEOBS_END"] = end.mjd

                if len(reproj_images) > 0:
                    if verbose:
                        print(
                            f"Found {len(reproj_pointings)} reprojected pointings, mosaicking them\n"
                        )
                    healpix_mosaic_table, meta = _add_hpx_mosaics(
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
                        # Add headers for metadata
                        for m in meta.keys():
                            f["mosaic"].attrs[m] = meta[m]
                        f.close()
                else:
                    raise ValueError(
                        "No reprojected images found for the given time bin, cannot make mosaic"
                    )
                # Now make a MosaicProjections object to capture this information
            else:
                # There are no observations in this time bin, so make an empty file
                if verbose:
                    print(
                        "No observations found for the given time bin, making empty mosaic file"
                    )
                outfile = img_dir.joinpath(
                    f"mosaic_{start_time_str}_{end_time_str}_hybrid_healpix_projected.h5"
                )
                with h5py.File(outfile, "w") as f:
                    # Store data
                    if verbose:
                        print(f"Writing reprojected data to {outfile}")
                    f.create_dataset(
                        "mosaic",
                        data=Table(),
                        compression="gzip",
                        compression_opts=4,
                    )

                    # Store header as attributes
                    # Add headers for metadata
                    for m in metadata.keys():
                        f["mosaic"].attrs[m] = metadata[m]
                    f.close()
        else:
            # Use existing intermediate mosaics to make the healpix mosaic
            if verbose:
                print("Using existing intermediate mosaics to make the healpix mosaic")

            # The initial plan was to find the GCF so that the number of images can be reduced
            # So for example, for 2 day mosaics, use 1 day mosaics, for 4 day mosaics use 2 day mosaics
            # but this is an issue when using sliding window approach since same observations will
            # Be included agaian and again. So instead, just use 1 days mosaics for now
            # But leave the code here to find the GCF for future use
            existing_mosiac_dir = [
                i
                for i in img_base_path.parent.glob(f"*day_mosaics")
                if f"{dt}_day_mosaics" not in i.name
            ]
            existing_mosiac_dts = np.array(
                [d.name.split("_")[0] for d in existing_mosiac_dir]
            )
            rem_mask = np.any(dt % existing_mosiac_dts.astype(float) == 0)

            if dt > 1:
                # Override and just use 1 day mosaics
                existing_mosiac_dir = img_base_path.parent.joinpath(f"1_day_mosaics")
                rem_mask = existing_mosiac_dir.exists()
            if rem_mask:

                # # Find the largest factor dt
                if dt > 1:
                    intermediate_dt = 1
                    intermediate_mosaic_dir = existing_mosiac_dir
                else:
                    full_rem_mask = dt % existing_mosiac_dts.astype(float) == 0
                    dt_max = np.max(existing_mosiac_dts.astype(float)[full_rem_mask])
                    dt_index = np.where(existing_mosiac_dts.astype(float) == dt_max)[0][
                        0
                    ]
                    intermediate_dt = existing_mosiac_dts[dt_index]
                    intermediate_mosaic_dir = existing_mosiac_dir[dt_index]

                if verbose:
                    print(
                        f"Using existing {intermediate_dt} day mosaics to make the {dt} day mosaic"
                    )
                # Select all the mosaics in this directory that fall within the time bin
                intermediate_mosaic_objs = [
                    MosaicProjections(mosaic_dir=i, recalc=False)
                    for i in intermediate_mosaic_dir.glob("mosaic_*")
                ]
                intermediate_mosaic_objs = [
                    m for m in intermediate_mosaic_objs if m.is_empty is False
                ]

                all_tstarts = []
                all_tstops = []
                for m in intermediate_mosaic_objs:
                    all_tstarts.append(m.DATEOBS_START)
                    all_tstops.append(m.DATEOBS_END)

                intermediate_mosaic_tstarts = Time(all_tstarts, format="mjd")
                intermediate_mosaic_tstops = Time(all_tstops, format="mjd")

                # Add some buffer to the time selection to avoid edge effects
                mask = (intermediate_mosaic_tstarts.mjd >= (start - 1 * u.hour).mjd) & (
                    intermediate_mosaic_tstops.mjd <= (end + 1 * u.hour).mjd
                )

                all_obsids = []
                all_pointing_ids = []
                all_reproj_images = []
                for m in np.array(intermediate_mosaic_objs)[mask]:
                    all_obsids = np.concatenate((all_obsids, m.obsids))
                    all_pointing_ids = np.concatenate(
                        (all_pointing_ids, m.pointing_ids)
                    )
                    all_reproj_images = np.concatenate(
                        (all_reproj_images, [m.mosaic_file])
                    )

                # Exclude duplicates
                all_obsids = np.unique(all_obsids).astype(str)
                all_pointing_ids = np.unique(all_pointing_ids).astype(str)
                all_reproj_images = np.unique(all_reproj_images).astype(str)

                # Now start adding these intermediate mosaics
                final_mosaic_table, meta = _add_hpx_mosaics(
                    all_reproj_images, verbose=verbose
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

                    # Add in metadata
                    for m in meta.keys():
                        f["mosaic"].attrs[m] = meta[m]

                    f.close()
                # Now add to metadata
                metadata["obsids"] = all_obsids
                metadata["pointing_ids"] = all_pointing_ids
                metadata["images"] = all_reproj_images
                metadata["DATEOBS_START"] = np.min(
                    intermediate_mosaic_tstarts.mjd[mask]
                )
                metadata["DATEOBS_END"] = np.max(intermediate_mosaic_tstops.mjd[mask])
            else:
                raise ValueError(
                    "No existing intermediate mosaics found that are integral factors of the requested time bin"
                )

        # Then decide whether to make the final sky image
        if make_image:
            if wcs is None:
                raise ValueError("WCS object must be provided if make_image is True")
            else:
                if outfile is not None:
                    convert_mosaic_to_sky_image(
                        hpx_file=outfile,
                        wcs=wcs,
                    )
        # Finally create the .batreproject file to indicate that reprojection is done
        img_dir.joinpath(".batreproject").touch()
        # Add make a MosaicProjections object to capture this information
        mosaic_survey_obj = MosaicProjections(
            mosaic_dir=img_dir,
            meta=metadata,
            snr=snr,
            plot_dest_dir=plot_dest_dir,
            plot_all_sources=plot_all_sources,
            plot_individual_sources=plot_individual_sources,
        )
    else:
        if verbose:
            print(f"Reprojected mosaic already exists in {img_dir}, not recalculating")
        # Load the MosaicProjections object
        mosaic_survey_obj = MosaicProjections(mosaic_dir=img_dir, recalc=False)
    return mosaic_survey_obj


def convert_mosaic_to_sky_image_grid(hpx_data, w, basepath, meta=None):
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
    emin = np.array(meta["EMIN"]).astype(float)
    emax = np.array(meta["EMAX"]).astype(float)
    any_obs_truncated = np.array(meta["ANYTRUNC"]).astype(bool)
    all_obs_truncated = np.array(meta["ALLTRUNC"]).astype(bool)
    mean_exp = np.nanmax(hpx_data["exposure"])

    for i in meta.keys():
        if i not in ["EBINS", "EMIN", "EMAX", "ANYTRUNC", "ALLTRUNC"]:
            hdr[i] = meta[i]
    hdr["EXPOSURE"] = (mean_exp, "Mean Exposure of the Healpix Mosaic [s]")

    sky_hdul = []
    var_hdul = []

    pri_hdu = fits.PrimaryHDU()
    sky_hdul.append(pri_hdu)
    var_hdul.append(pri_hdu)

    # Now get the weights and vertices for the interpolation
    # First get the image cordinates for the pixels, so that we can see what lies inside the
    # given WCS first get the common hpix coordinates (that lie in the WCS)

    print("Calculating pixel coordinates for reprojection\n")
    # Now get pixel indices
    # coords_hpx, ipix, _ = get_common_healpix_grid(wcs=w)
    pixel_x, pixel_y, ipix = get_common_healpix_grid(wcs=w)
    # pixel_x, pixel_y = w.world_to_pixel(coords_hpx)

    # need to interpolate this all sky image onto the given WCS
    # need to verify that the eimg and pimg maps are energy independent, in idl code only does this
    # for te first energy iteration

    grid_x, grid_y = np.mgrid[0 : w.pixel_shape[1], 0 : w.pixel_shape[0]]
    points = np.array([grid_x.flatten(), grid_y.flatten()])

    # before had: #np.array([chosen_pixel_x, chosen_pixel_y]) for the below line but the results
    # werent consistent with the idl code results. changing the x and y pixel coordinates here works
    interp_at_points = np.array([pixel_y, pixel_x])

    # see if thie other method works,
    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    print("Calculating weights and vertices for sky projection\n")
    vtx, wts = interp_weights(interp_at_points.T, points.T)

    # Now make the images
    # sky_images = np.zeros(np.concatenate(([ebins], w.pixel_shape)))
    # var_images = np.zeros_like(sky_images)
    # exp_image = np.zeros_like(sky_images[0])

    print(f"Found {ebins} energy bins, starting reprojection\n")
    for e in range(int(ebins)):
        # This is where we calculate the final mosaic sky image for each energy band
        # In the _add_hpx_mosaics function, we left the sky image as numerator and variance as denominator squared
        # So here we need to do the final division to get the actual sky image and variance
        # First do it for sky image
        sky_data = hpx_data[f"sky_e{e+1}"][ipix]
        var_data = hpx_data[f"var_e{e+1}"][ipix]

        hdr["E_MIN"] = (emin[e], "Minimum energy of the band [keV]")
        hdr["E_MAX"] = (emax[e], "Maximum energy of the band [keV]")
        hdr["ANYTRUNC"] = (
            any_obs_truncated[e],
            "Whether any of the observations contributing to this energy bin were truncated",
        )
        hdr["ALLTRUNC"] = (
            all_obs_truncated[e],
            "Whether all of the observations contributing to this energy bin were truncated",
        )

        # When we run batcelldetect, if everything is nan, it crashes, so if all the
        # observations contributing to this energy bin were truncated, we set the
        # sky image to be 0 everywhere, so that it can be used as a mask for source detection.
        # But we can keep the variance image as nan everywhere, as it will not get affected.
        # This is a bit hacky but it works.
        sky_data = divide_with_respect(sky_data, var_data)
        var_data = np.sqrt(divide_with_respect(1, var_data))
        test = interpolate(sky_data, vtx, wts, fill_value=np.nan)
        # This is a flattend array, so make it 2D again
        if all_obs_truncated[e]:
            sky_image = np.zeros_like(grid_x)
        else:
            sky_image = test.reshape(grid_x.shape)
        sky_image_hdu = fits.ImageHDU(data=sky_image, header=hdr, name=f"BATIMAGE{e+1}")
        sky_hdul.append(sky_image_hdu)

        # Do the same with variance image
        test = interpolate(var_data, vtx, wts, fill_value=np.nan)
        if all_obs_truncated[e]:
            var_image = np.zeros_like(grid_x)
        else:
            var_image = test.reshape(grid_x.shape)
        var_image_hdu = fits.ImageHDU(data=var_image, header=hdr, name=f"BATIMAGE{e+1}")
        var_hdul.append(var_image_hdu)

    # Make one for total energy
    # Make the same changes for total energy band
    # First for sky image
    sky_data = divide_with_respect(
        hpx_data[f"sky_etot"][ipix], hpx_data[f"var_etot"][ipix]
    )
    var_data = np.sqrt(divide_with_respect(1, hpx_data[f"var_etot"][ipix]))

    hdr["E_MIN"] = (
        np.min(emin[~all_obs_truncated]),
        "Minimum energy of the band [keV]",
    )
    hdr["E_MAX"] = (
        np.max(emax[~all_obs_truncated]),
        "Maximum energy of the band [keV]",
    )
    hdr["ALLTRUNC"] = False
    hdr["ANYTRUNC"] = False
    test = interpolate(sky_data, vtx, wts, fill_value=np.nan)
    sky_image = test.reshape(grid_x.shape)
    sky_image_hdu = fits.ImageHDU(data=sky_image, header=hdr, name=f"BATIMAGE_TOT")
    sky_hdul.append(sky_image_hdu)

    # Same for var image
    test = interpolate(var_data, vtx, wts, fill_value=np.nan)
    var_image = test.reshape(grid_x.shape)
    var_image_hdu = fits.ImageHDU(data=var_image, header=hdr, name=f"BATIMAGE_TOT")
    var_hdul.append(var_image_hdu)

    # Next make one for exposure, but normalize it so that it can be used as a mask for
    # cell detections
    test = interpolate(hpx_data[f"exposure"][ipix], vtx, wts, fill_value=np.nan)
    exp_image = test.reshape(grid_x.shape)
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


def convert_mosaic_to_sky_image(hpx_file, wcs):
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

    if len(data) == 0:
        print(f"No data found in healpix mosaic {hpx_file}, skipping sky projection")
    else:
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

        for sgrid in list(wcs.keys()):
            convert_mosaic_to_sky_image_grid(
                data,
                wcs[sgrid],
                filenames[sgrid],
                meta,
            )


class ReprojectSurvey(BatSurvey):
    """
    A general reproject object that holds all information about reprojecting survey images.

    Attributes
    ---------------
    obs_id : str
        observation ID
    obs_dir : str or None
        Directory that the observation ID folder resides within
    result_dir : str
        The directory that holds the output of the heasoft batsurvey calculations
    batsurvey_result : heasoftpy Result object
        The output of calling heasoftpy batsurvey
    pointing_ids : list of strings
        The pointing  ids for the successfully analyzed pointings associated with the analyzed obs_id
    pointing_info : dictionary of dictionaries
        The encompassed information including MET time, exposure time, etc for each pointing in a observation id
        Can be access as pointing_info[pointing_id]["key"]. This also includes poining IDs that failed to be analyzed
        and includes their reason for failure.
    channel : list
        List of the channel number for the survey data energy channels
    emin : list
        List of the energy lower limits for the survey data energy channels
    emax : list
        List of the energy upper limits for the survey data energy channels
    syserr : list
        List of the systematic errors associated with each energy channel

    Methods
    ---------------
    save():
        Saves a ReprojectSurvey object
    """

    def __init__(self, obsid, obs_dir=None, recalc=False, verbose=False):
        """
        Initializer method for the ReprojectSurvey object.

        :param obsid: The observational ID to be analyzed
        :param recalc: Boolean default False, which indicates that the method should try to load data from a file in
            the mosaic_dir directory. True means that the load file will be ignored and attributes will be re-obtained
            for the object.
        :param verbose: Boolean default False, which indicates where to print informative messages during the processing.
        """
        if obs_dir is None:
            obs_dir = datadir()
        super().__init__(obs_id=obsid, obs_dir=obs_dir, recalc=False)
        # Always set recalc to False for the parent class, this
        # is not meant to run the batsurvey again

        # Now use batsurvey class to call in parameters

        load_file = self.result_dir.joinpath("batsurvey.pickle")
        obsdir = self.obs_dir.expanduser().resolve()

        reprojection_obj_file = self.result_dir.joinpath("batreproject.pickle")
        batsurvey_res_file = self.result_dir.joinpath("stats_point.fits")

        # See if a loadfile exists, if not dont proceed
        if load_file.exists() and batsurvey_res_file.exists():
            self._nside = _nside

            # Then use this to run the reprojection
            # if there is already a file load it, if not run it
            if (not reprojection_obj_file.exists()) or recalc:
                if reprojection_obj_file.exists():
                    print(
                        f"""A previous reprojection {reprojection_obj_file} is found. But recalculating"""
                    )
                else:
                    print(
                        f"""No existing reprojection file {reprojection_obj_file} found.\
                        Running reprojection now."""
                    )

                _, reprojected_pointings = do_sky_reprojection(
                    obsdir=obsdir, verbose=verbose
                )
                self.batsurvey_obj = self.batsurvey_result

                # Store other parameters
                self.nside = _nside

                # Store path of the reprojected image
                self.reprojected_pointings = reprojected_pointings

                reprojected_images = [
                    f"{self.result_dir}/point_{pid}/point_{pid}_{self.batsurvey_result.params['ncleaniter']}_hybrid_healpix_projected.h5"
                    for pid in reprojected_pointings
                ]
                self.reprojected_images = reprojected_images

                # If it proceeded till here, it means that everthing was successful
                # reprojection_status_file.touch()

                # Save it to pickle file
                self._save(reprojection_obj_file)
            else:
                load_file = Path(reprojection_obj_file).expanduser().resolve()
                self._load(reprojection_obj_file)

        else:
            raise FileNotFoundError(
                f"Please run the batsurvey analysis first. There are no sky images to be reprojected for {obsid}"
            )

    def _load(self, f):
        """
        Loads a saved BatReproject object
        :param f: String of the file that contains the previously saved BatReproject object
        :return: None
        """
        print(f"Loading state from {f} into current instance...")
        with open(f, "rb") as f:
            content = pickle.load(f)

        # Update the current instance's internal dictionary
        self.__dict__.update(content)
        print("Load successful.")

    def _save(self, file):
        """
        Saves the current BatSurvey object
        :param f: String of the file to save the BatReproject object
        :return: None
        """

        try:
            with open(file, "wb") as f:
                pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Object successfully saved to {file}")
        except Exception as e:
            print(f"Error saving object: {e}")


def filter_duplicate_sources(tab, sep=5 * u.arcmin):
    coords = SkyCoord(
        l=tab["GLON_OBJ"], b=tab["GLAT_OBJ"], frame="galactic", unit="deg"
    )

    # Convert to 3D cartesian (unit sphere)
    xyz = np.vstack(
        [coords.cartesian.x.value, coords.cartesian.y.value, coords.cartesian.z.value]
    ).T

    # Run DBSCAN: eps in radians
    cluster = DBSCAN(eps=(sep).to(u.rad).value, min_samples=1, metric="euclidean").fit(
        xyz
    )

    labels = cluster.labels_
    vals, counts = np.unique(labels, return_counts=True)

    new_tab = Table()
    new_tab = vstack([new_tab, tab[np.isin(labels, vals[counts == 1])]])

    # Add a column for number of detections
    new_tab.add_column(Column(np.ones(len(new_tab), dtype=int), name="NDETECTIONS"))

    for l in vals[counts > 1]:
        subtab = tab[np.isin(labels, [l])]
        subtab.add_column(
            Column([len(subtab)] * len(subtab), name="NDETECTIONS"),
        )
        new_tab.add_row(subtab[np.argmax(subtab["SNR"])])
    return new_tab


def post_process_catalogs(files, outfile=None, mode="merge"):
    """Helper function to combine and sort two fits files

    Args:
        files (list): list of paths to the input files
        outfile (str): path to the output file
        mode (str): "merge" to merge the files, "concat" to concatenate and remove duplicates
    Returns:
        None
    """

    if mode == "merge":
        if outfile is None:
            outfile = (
                Path(files[0])
                .expanduser()
                .resolve()
                .parent.joinpath("merged_catalog.fits")
            )

        heatools.ftmerge(
            infile=",".join(files),
            outfile=str(outfile),
            clobber="YES",
        )
    if mode == "concat":
        for f in files:
            if outfile is None:
                outfile = f
            # get the coordinates from galactic to RA/DEC
            heatools.ftcoco(
                infile=str(outfile),
                outfile=str(outfile),
                incoord="G",
                outcoord="R",
                lon1="GLON_OBJ",
                lat1="GLAT_OBJ",
                lon2="RA_OBJ",
                lat2="DEC_OBJ",
                clobber="YES",
            )

            # sort based on catalog number and how far away the source is from the edge of the FOV of a facet
            # (want further away to minimize edge effects of interpolation)
            # and then denote the duplicates and get rid of them
            heatools.ftsort(
                infile=f"{outfile}[col *;FACET_DIST = ANGSEP(GLON_OBJ,GLAT_OBJ,CRVAL1,CRVAL2); DUP = F]",
                outfile=str(outfile),
                columns="CATNUM, FACET_DIST",
                clobber="YES",
            )


class MosaicProjections:
    """
    A general mosaic object that holds all information about mosaiced reprojected survey images.
    """

    def __init__(
        self,
        mosaic_dir,
        meta=None,
        recalc=False,
        persistent_source_sep=10 * u.arcmin,
        snr=5,
        plot_dest_dir=None,
        plot_all_sources=True,
        plot_individual_sources=False,
    ):
        """Initializer method for the MosaicReprojectedSurvey object.

        Args:
            mosaic_dir (str): Path to the mosaic directory
            meta (dict, optional): Any additional metadata to be stored in the object. Defaults to None.
            recalc (bool, optional): Boolean default False, which indicates that the method should try to load data from a file in
                the mosaic_dir directory. True means that the load file will be ignored and attributes will be re-obtained
                for the object. Defaults to False
            persistent_source_sep (astropy.units.Quantity, optional): Separation threshold to distinguish from persistent sources.
                Defaults to 10 * u.arcmin.
            snr (float, optional): SNR threshold for source detection. Defaults to 5.
            :param plot_dest_dir: String of the directory to save the new source plots to. None defaults to the directory
            that holds the batsurvey result directory.
            :param plot_all_sources: Boolean to plot all the new sources found in the survey data to the same plot.
                Default is True.
            :param plot_individual_sources: Boolean to plot each new source found in the survey data to its own plot.
                Default is False.
        """
        mosaic_obj_file = (
            Path(mosaic_dir).expanduser().resolve().joinpath("batmosaic.pickle")
        )
        det_obj_file = Path(mosaic_dir).expanduser().resolve().joinpath(".batdetect")
        self.sep = persistent_source_sep
        if not mosaic_obj_file.exists() or recalc:
            self.mosaic_dir = Path(mosaic_dir).expanduser().resolve()
            self.mosaic_file = [
                i.as_posix()
                for i in self.mosaic_dir.glob("*_hybrid_healpix_projected.h5")
            ][0]
            self.mosaic_sky_images = [
                i.as_posix() for i in self.mosaic_dir.glob("*_sky_image.fits")
            ]
            self.mosaic_sky_images.sort()
            self.mosaic_var_images = [
                i.replace("_sky_image", "_var_image") for i in self.mosaic_sky_images
            ]
            self.mosaic_exp_images = [
                i.replace("_sky_image", "_exp_image") for i in self.mosaic_sky_images
            ]

            # Add any metadata
            for key, value in meta.items():
                setattr(self, key, value)

            # Now save it
            self.save()

            # Then run source detection, if there are any files
            if len(self.mosaic_sky_images) > 0:
                self.load_default_params()
                self.detect_sources(snr=snr)
                self.load_and_filter_sources(snr_thresh=snr)
                self.plot_cutouts(
                    dest_dir=plot_dest_dir,
                    save_coll=plot_all_sources,
                    save_ind=plot_individual_sources,
                )
                self.is_empty = False
            else:
                self.is_empty = True

            # Then save again
            self.save()
            det_obj_file.touch()
        else:
            self.load(mosaic_obj_file)
            if not det_obj_file.exists():
                # Then run source detection
                if len(self.mosaic_sky_images) > 0:
                    self.load_default_params()
                    self.detect_sources(snr=snr)
                    self.load_and_filter_sources(snr_thresh=snr)
                    self.plot_cutouts(
                        dest_dir=plot_dest_dir,
                        save_coll=plot_all_sources,
                        save_ind=plot_individual_sources,
                    )
                    self.is_empty = False
                else:
                    self.is_empty = True

                # Then save again
                self.save()
                det_obj_file.touch()

    def save(self, name="batmosaic.pickle"):
        """
        Saves the current MosaicReprojectedSurvey object
        :param f: String of the file to save the MosaicReprojectedSurvey object
        :return: None
        """
        obj_file = self.mosaic_dir.joinpath(name)
        try:
            with open(obj_file, "wb") as f:
                pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Object successfully saved to {obj_file}")
        except Exception as e:
            print(f"Error saving object: {e}")

    def load(self, f):
        """
        Loads a saved MosaicReprojectedSurvey object
        :param f: String of the file that contains the previously saved MosaicReprojectedSurvey object
        :return: None
        """
        print(f"Loading state from {f} into current instance...")
        with open(f, "rb") as f:
            content = pickle.load(f)

        # Update the current instance's internal dictionary
        self.__dict__.update(content)
        print("Load successful.")

    def _call_batcelldetect(self, input_dict):
        """
        Call heasoftpy batcelldetect.

        :param input_dict: dictionary of inputs to pass to batcelldetet.
        :return: heasoft output object
        """
        # make the local pfile dir if it doesnt exist and set this value
        # Set the local pfile directory, so that heasoftpy utils run
        self._local_pfile_dir = self.mosaic_dir.joinpath(".local_pfile")

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        out = hsp.batcelldetect(**input_dict)

        # And then once this is run, unlink it so that it doesnt take up space
        shutil.rmtree(self._local_pfile_dir)
        return out

    def load_default_params(self):
        """Load default parameters for batcelldetect."""
        self.bat_source_catalog = Path(__file__).parent.joinpath("data/survey6b_2.cat")
        self.default_input_dict = dict(
            incatalog=str(self.bat_source_catalog.as_posix()),
            snrthresh=5,
            psfshape="GAUSSIAN",
            psffwhm=0.37413,
            posfitwindow=0.2,
            srcfit="YES",
            posfit="YES",
            pospeaks="YES",
            posfluxfit="NO",
            bkgwindowtype="SMOOTH_CIRCLE",
            srcdetect="YES",
            nadjpix=3,
            srcradius=8,
            bkgradius=50,
            bkgfit="YES",
            keepbits="ALL",
            hduclasses="NONE",
            chatter=3,
            clobber="YES",
            distfile="NONE",
            nullborder="NO",
            carryover="YES",
            vectorflux="YES",
            vectorposmeth="MAX_SNR",
            keepkeywords="FACET,CRVAL1,CRVAL2,*VER",
        )

    def detect_sources(self, snr=5):
        """Run batcelldetect on the mosaiced sky images to detect sources"""

        # Get a list of all BAT sources

        # For the list of images, run batcelldetect
        source_catalogs = []
        empty_source_catalogs = []
        batcelldetect_output = []
        for i in range(len(self.mosaic_sky_images)):
            default_input_dict = self.default_input_dict.copy()
            file = self.mosaic_sky_images[i]
            # varfile = self.var_images[i]
            expfile = self.mosaic_exp_images[i]

            # Get the imaging region where source estimation has to be performed
            exp_data = fits.getdata(expfile, 0)
            # Cut off when the exposure time is < 5% of the max exposure time
            exp_cut = np.min(
                [500, 0.05 * np.nanmax(exp_data) + 0.95 * np.nanmin(exp_data)]
            )

            # There seems to be something in batcelldetect that causes the SNRs of the total
            # image to be lower when analyzed combinedly with the individual energy bin
            # images. I did not yet figure out what this issue is, but for now analyze them
            # separately.

            catalog_file = file.replace("sky_image", "src_catalog")
            snr_map = file.replace("sky_image", "snr_map")

            # Now analyze them
            # First do independent source detection on all energy bins
            print(f"Running batcelldetect on {file} and all energy bins\n")
            default_input_dict["infile"] = file
            default_input_dict["snrthresh"] = snr
            default_input_dict["incatalog"] = ""
            default_input_dict["vectorflux"] = "NO"
            default_input_dict["carryover"] = "NO"
            default_input_dict["pcodefile"] = expfile
            default_input_dict["pcodethresh"] = exp_cut
            default_input_dict["outfile"] = catalog_file
            default_input_dict["signifmap"] = snr_map
            default_input_dict["clobber"] = "YES"
            default_input_dict["rows"] = "1-9"
            batcelldetect_output.append(self._call_batcelldetect(default_input_dict))

            source_hdu = fits.open(catalog_file, mode="update")
            source_data = Table(source_hdu[1].data)
            if len(source_data) > 0:
                # Now select unique sources
                unique_sources = filter_duplicate_sources(source_data)

                source_hdu[1] = fits.BinTableHDU(
                    unique_sources, header=source_hdu[1].header
                )
                source_hdu.flush()
                source_hdu.close()
                # Only run if there are any sources detected
                print(f"Running batcelldetect on {file} for flux estimation\n")
                default_input_dict["vectorflux"] = "YES"
                default_input_dict["carryover"] = "YES"
                default_input_dict["posfit"] = "NO"
                default_input_dict["pospeaks"] = "YES"
                default_input_dict["posfluxfit"] = "NO"
                default_input_dict["posfitwindow"] = 0.0
                default_input_dict["incatalog"] = catalog_file
                default_input_dict["outfile"] = catalog_file + ".tmp"
                default_input_dict["signifmap"] = ""
                default_input_dict["srcdetect"] = (
                    "NO"  # No need to detect sources again
                )

                # Now use this to run a second iteration of batcelldetect to get the fluxes
                batcelldetect_output.append(
                    self._call_batcelldetect(default_input_dict)
                )

                # For each of these catalogs, add in extra information
                # Add other key columns
                # Now select unique sources
                ########################################################################
                # This is a patch work to filter sources that sometimes are detected when
                # allowed for position fitting, but not detected when position fitting is off
                source_hdu = fits.open(default_input_dict["outfile"], mode="update")
                src_hdu_old = fits.open(catalog_file)

                src_data_old = Table(src_hdu_old[1].data)
                src_data = Table(source_hdu[1].data)

                old_coords = SkyCoord(
                    l=src_data_old["GLON_OBJ"],
                    b=src_data_old["GLAT_OBJ"],
                    frame="galactic",
                    unit="deg",
                )
                coords = SkyCoord(
                    l=src_data["GLON_OBJ"],
                    b=src_data["GLAT_OBJ"],
                    frame="galactic",
                    unit="deg",
                )

                idx, d2d, _ = coords.match_to_catalog_sky(old_coords)
                mask = d2d.arcmin < 20  # 20 arcmin matching

                glon_err = src_data_old[idx[mask]]["GLON_OBJ_ERR"]
                glat_err = src_data_old[idx[mask]]["GLAT_OBJ_ERR"]

                # Add these columns
                source_data = src_data[mask]
                source_data.replace_column(
                    "GLON_OBJ_ERR", Column(glon_err, name="GLON_OBJ_ERR")
                )
                source_data.replace_column(
                    "GLAT_OBJ_ERR", Column(glat_err, name="GLAT_OBJ_ERR")
                )
                source_data.add_column(
                    Column([file] * len(source_data), name="IMAGE_FILE")
                )
                # Get mosaic timescale
                ts = [i for i in file.split("/") if "_day_mosaic" in i][0].split("_")[0]
                source_data.add_column(
                    Column([f"mosaic ({ts}d)"] * len(source_data), name="CAT_TYPE")
                )
                source_hdu[1] = fits.BinTableHDU(
                    source_data, header=source_hdu[1].header
                )
                source_hdu.flush()
                source_hdu.close()
                src_hdu_old.close()
                ###########################################################################
                # Remove the older file and move this
                Path(catalog_file).unlink(missing_ok=True)
                shutil.move(
                    src=default_input_dict["outfile"],
                    dst=catalog_file,
                )
                source_catalogs.append(catalog_file)
            else:
                source_hdu.close()
                empty_source_catalogs.append(catalog_file)
                print(f"No sources detected in {file}, skipping further analysis\n")

            # Now for these detected sources, calculate effective SNRs
            # First add other information to the catalog
            post_process_catalogs([catalog_file], mode="concat")

        # Then merge all the catalogs
        comb_source_file = self.mosaic_file.replace(
            "_hybrid_healpix_projected.h5", "_comb_source_catalog.fits"
        )
        post_process_catalogs(source_catalogs, outfile=comb_source_file, mode="merge")

        self.batcelldetect_output = dict(zip(source_catalogs, batcelldetect_output))
        self.source_catalogs = np.concatenate((source_catalogs, empty_source_catalogs))
        self.comb_source_file = comb_source_file

    def load_and_filter_sources(self, snr_thresh=5):
        """Load the combined source catalog and filter based on SNR threshold.

        Args:
            snr_thresh (float, optional): The SNR threshold to filter sources. Defaults to 5.0.

        Returns:
            Table: Filtered source catalog.
        """
        # Load the combined source catalog
        source_data = Table.read(self.comb_source_file, format="fits")

        coords = SkyCoord(
            b=source_data["GLAT_OBJ"],
            l=source_data["GLON_OBJ"],
            unit="deg",
            frame="galactic",
        )

        coords_icrs = coords.transform_to("icrs")
        # Change names
        src_names = coords_icrs.to_string("hmsdms", fields=3, sep="", precision=0)
        src_names = ["J" + name.replace(" ", "") for name in src_names]
        source_data.replace_column("NAME", src_names)

        source_data.add_column(np.max(source_data["SNR"], axis=1), name="MAX_SNR")
        # Also add the index of the energy band where the max SNR occurs

        source_data.add_column(
            np.argmax(source_data["SNR"], axis=1), name="MAX_SNR_ENBIN"
        )
        # source_data.sort(["NAME", "FACET_DIST"], reverse=True)

        # Select and remove BAT persistent sources
        persistent_sources = Table.read(self.bat_source_catalog, format="fits")
        bat_source_coords = SkyCoord(
            ra=persistent_sources["RA_OBJ"],
            dec=persistent_sources["DEC_OBJ"],
            unit="deg",
            frame="icrs",
        )

        _, sep, _ = coords.match_to_catalog_sky(bat_source_coords)
        source_data.add_column(sep.arcmin, name="BAT_SRC_SEP")

        # Filter sources based on SNR threshold
        snr_mask = source_data["MAX_SNR"] >= snr_thresh
        sep_mask = sep.arcmin >= self.sep.to(u.arcmin).value  # PSF of BAT

        self.unknown_sources = source_data[(sep_mask & snr_mask)]
        self.known_sources = source_data[(~sep_mask) & snr_mask]

        # Now for the unknown sources, add a FAR calculation
        eff_snr = []
        nfp = []
        for i in range(len(self.unknown_sources)):
            nametag = f"*_l_{self.unknown_sources['CRVAL1'][i]:03.0f}_b_{self.unknown_sources['CRVAL2'][i]:+03.0f}_ZEA_sky_image.fits"
            image_file = [
                fil for fil in Path(self.comb_source_file).parent.glob(nametag)
            ][0]
            var_file = str(image_file).replace("sky_image", "var_image")
            ext = self.unknown_sources["MAX_SNR_ENBIN"][i] + 1
            img = fits.getdata(image_file, ext)
            var = fits.getdata(var_file, ext)

            esnr, nfp_val = calculate_effective_snr(
                img=img, var=var, snr=self.unknown_sources["MAX_SNR"][i]
            )

            eff_snr.append(esnr)
            nfp.append(nfp_val)
        self.unknown_sources.add_column(eff_snr, name="ESNR")
        self.unknown_sources.add_column(nfp, name="NFP")
        # Write it to a file
        self.unknown_source_file = self.comb_source_file.replace(
            "_comb_source_catalog.fits", "_unknown_source_catalog.fits"
        )
        self.known_source_file = self.comb_source_file.replace(
            "_comb_source_catalog.fits", "_known_source_catalog.fits"
        )

        self.unknown_sources.write(
            self.unknown_source_file,
            format="fits",
            overwrite=True,
        )
        self.known_sources.write(
            self.known_source_file,
            format="fits",
            overwrite=True,
        )

    def plot_cutouts(self, dest_dir=None, save_coll=True, save_ind=False):
        """Plot cutouts around the given sources from the mosaiced sky images.

        Args:
            sources (Table): Table of sources with RA and DEC columns.
            dest_dir (str, optional): Directory to save the plots. Defaults to None.
            save_coll (bool, optional): Whether to save the collective plot as a PDF. Defaults to True.
            save_ind (bool, optional): Whether to save individual plots as PNG files. Defaults to False.
        """
        sources = self.unknown_sources

        dest_dir = (
            Path(dest_dir) if dest_dir is not None else Path(self.mosaic_file).parent
        )
        self.plot_dest_dir = dest_dir

        if save_coll:
            filename = dest_dir.joinpath(
                Path(self.mosaic_file).name.replace(
                    "_hybrid_healpix_projected.h5", "_source_cutouts.pdf"
                )
            )
            self.plotfile = filename
            plotfile = PdfPages(filename.as_posix())
        if save_ind:
            # Save individual cutout plots
            cutout_dir = (
                dest_dir.joinpath("source_cutouts") if dest_dir is None else dest_dir
            )
            cutout_dir.mkdir(parents=True, exist_ok=True)

        # Get all coords
        coords = SkyCoord(
            l=sources["GLON_OBJ"], b=sources["GLAT_OBJ"], unit="deg", frame="galactic"
        )

        for i in range(len(sources)):
            c = coords[i]
            c_icrs = c.transform_to("icrs")
            src_name = "J" + c_icrs.to_string(
                "hmsdms", fields=3, sep="", precision=0
            ).replace(" ", "")
            file = [
                fil
                for fil in Path(self.mosaic_file).parent.glob(
                    f"*_l_{sources['CRVAL1'][i]:03.0f}_b_{sources['CRVAL2'][i]:+03.0f}_ZEA_sky_image.fits"
                )
            ][0]

            hdu = fits.open(file)[sources["MAX_SNR_ENBIN"][i] + 1]

            wcs = WCS(hdu.header)
            data = hdu.data
            cut = Cutout2D(
                data,
                c,
                wcs=wcs,
                size=8 * u.degree,
            )

            # Make figure
            plt.clf()
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection=cut.wcs)
            vmin, vmax = ZScaleInterval().get_limits(cut.data[~np.isnan(cut.data)])
            ax.imshow(cut.data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
            ax.grid("on", color="k", ls="--")

            # Add a circle at the center of the image
            patch = Circle(
                np.array(cut.wcs.world_to_pixel(c)),
                radius=22 / 2.8 / 2,
                edgecolor="red",
                facecolor="none",
                fill=False,
                lw=2,
            )
            ax.add_patch(patch)
            ax.tick_params(labelsize=10)

            # Format it
            ax.coords[0].set_axislabel("Galactic Longitude (deg)", fontsize=12)
            ax.coords[1].set_axislabel("Galactic Latitude (deg)", fontsize=12)
            ax.coords[0].set_ticks_position("b")
            ax.coords[1].set_ticks_position("l")
            ax.coords[0].set_axislabel_position("b")
            ax.coords[1].set_axislabel_position("l")

            # Now add galactic coordinates grid
            gal = ax.get_coords_overlay("icrs")

            # Position
            gal[0].set_ticks_position("t")
            gal[1].set_ticks_position("r")

            gal[0].set_axislabel_position("t")
            gal[1].set_axislabel_position("r")

            # Color
            gal_color = "red"
            gal[0].set_axislabel("RA", color=gal_color)
            gal[1].set_axislabel("Dec", color=gal_color)
            gal[0].set_ticklabel(color=gal_color)
            gal[1].set_ticklabel(color=gal_color)
            gal[0].set_ticks(color=gal_color)
            gal[1].set_ticks(color=gal_color)
            gal.grid("on", color=gal_color, ls="dotted")

            # Make the title
            title = f"Source at RA={c_icrs.ra.deg:.2f} Dec={c_icrs.dec.deg:.2f} SNR= {sources['MAX_SNR'][i]:.2f} ({sources['MAX_SNR_ENBIN'][i]})"
            title += f", eSNR={sources['ESNR'][i]:.2f}, FP={int(sources['NFP'][i])}"

            fig.suptitle(title, fontsize=12)
            plt.tight_layout()

            if save_coll:
                plotfile.savefig(fig)
            if save_ind:
                ind_filename = cutout_dir.joinpath(f"{src_name}_cutout.jpg").as_posix()
                plt.savefig(ind_filename, dpi=150)
            plt.close(fig)

        if save_coll:
            plotfile.close()
