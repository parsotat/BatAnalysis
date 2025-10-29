"""
This file is meant to hold the functions that allow users to reproject the survey images on to a healpix grid
"""

import h5py, sys
from pathlib import Path

import pickle
import numpy as np
import pkg_resources
import scipy.spatial.qhull as qhull
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy import units as u
from reproject import reproject_to_healpix
from astropy.table import Table
import healpy as hp
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from astropy.nddata import Cutout2D

from .bat_survey import BatSurvey
from .batlib import dirtest, met2utc, datadir

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
_cimgfile = "offaxiscorr_8bin_20061221.img"
_chilothresh = 0.50  # Minimum chi-square for any energy band
_chihithresh = 1.15  # Maximum chi-square for any energy band
_chiscobump = 0.35  # Additional bump of chi-square threshold around Sco X-1 (band 0)
_chiscotheta = 30  # Approximate angular scale of bump around Sco X-1 (deg)
_pcodethresh = 0.15  # Minimum image partial coding
_minexpo = 150  # Minimum image exposure
_nskyimg = 6  # Number of facets to sky image
_nebands = 8  # Number of energy bands to process
_proj = "ZEA"  # projection from idl code that is used
_order = 10  # Healpix order parameter
_nside = 2**_order  # Healpix nside parameter

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

    # If there are nans here, handle it
    f[np.isnan(ang_sep)] = np.inf

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

    # Found that for the observations that are not processed succesfully by batsurvey, due to
    # variety of flags, the stats_point.fits file can have -999 in ra and dec. This cases astropy
    # Coordinates to crash, write a small snippet to handle this
    bad_coords_mask = (ra_pnt == -999) | (dec_pnt == -999)
    ra_pnt[bad_coords_mask] = np.nan
    dec_pnt[bad_coords_mask] = np.nan

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


def do_sky_reprojection_hybrid(obsdir, verbose=True):
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

    ncleaniter = survey_obj.batsurvey_result.params["ncleaniter"]

    # Using this read the stats file of interest
    stat_file = survey_obj.result_dir.joinpath("stats_point.fits")
    stats_data = fits.getdata(stat_file.as_posix(), 1)

    # Also read in corrections map
    corrections_map = read_correctionsmap()

    # Only proceed of there are good pointings
    if len(stats_data["NBATDETS"]) > 0:
        # pointing_ids = np.array(survey_obj.pointing_ids)
        pointing_ids = np.array(
            [i.replace("point_", "") for i in stats_data["IMAGE_ID"]]
        )
        processed_mask = np.zeros(len(pointing_ids)).astype(bool)

        # calculate the mask of which points we should use based on good image statistics
        chi_mask = compute_statistics_map(
            stats_data["CHI2"],
            stats_data["NBATDETS"],
            stats_data["RA_PNT"],
            stats_data["DEC_PNT"],
            stats_data["PA_PNT"],
            stats_data["TSTART"],
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
                img_hdu = fits.open(pointing_img_str.as_posix())
                var_hdu = fits.open(pointing_var_str.as_posix())
                # read the partial coding map

                # get other header information
                survey_ver = img_hdu["BAT_PCODE_1"].header["BSURVER"]
                pointing_exposure = img_hdu["BAT_PCODE_1"].header["EXPOSURE"]
                pointing_tstart = img_hdu["BAT_PCODE_1"].header["TSTART"]
                pointing_tstop = img_hdu["BAT_PCODE_1"].header["TSTOP"]
                pointing_dateobs_start = img_hdu["BAT_PCODE_1"].header["DATE-OBS"]
                pointing_dateobs_end = img_hdu["BAT_PCODE_1"].header["DATE-END"]

                # Only proceed, if it exceeds a minimum exposure criterion
                if pointing_exposure >= _minexpo:

                    pcoding_image = img_hdu["BAT_PCODE_1"].data
                    # save the header for use later
                    pcoding_header = img_hdu[0].header
                    pcoding_wcs = WCS(pcoding_header)

                    # get the image size and create array to hold the sky flux at each channel
                    sz = pcoding_image.shape

                    # Make the sky image
                    sky_image = np.zeros(
                        (sz[0], sz[1], _nebands + 1)
                    )  # plus 1 for the total energy

                    # Make the sky variance image
                    var_image = np.zeros_like(sky_image)
                    for k in range(_nebands):
                        sky_image[:, :, k] = img_hdu[k].data
                        var_image[:, :, k] = var_hdu[k].data

                    # correct for off axis effects
                    sky_image[:, :, :-1] = np.divide(
                        sky_image[:, :, :-1],
                        corrections_map,
                        out=np.zeros_like(corrections_map),
                        where=corrections_map != 0,
                    )
                    var_image[:, :, :-1] = np.divide(
                        var_image[:, :, :-1],
                        corrections_map,
                        out=np.zeros_like(corrections_map),
                        where=corrections_map != 0,
                    )

                    # construct the total energy images for variance and flux, the zeros in last array dont affect
                    # calculations of the total values
                    # Fluxes add in a simple way
                    sky_image[:, :, -1] = sky_image.sum(axis=2)

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
                    sky_image *= energy_quality_mask
                    var_image *= energy_quality_mask

                    sky_image[np.isnan(sky_image)] = 0
                    var_image[np.isnan(var_image)] = 0

                    if verbose:
                        print("Getting common healpix grid\n")
                    # get the common healpix grid

                    coords_hpx, ipix, val_hpx = get_common_healpix_grid(pcoding_header)

                    if verbose:
                        print("Calculating pixel coordinates for reprojection\n")
                    # Now get pixel indices
                    pixel_x, pixel_y = pcoding_wcs.world_to_pixel(coords_hpx)

                    # need to interpolate the survey sky image onto the all sky image
                    # need to verify that the eimg and pimg maps are energy independent, in idl code only does this
                    # for te first energy iteration
                    grid_x, grid_y = np.mgrid[
                        0 : pcoding_image.shape[0], 0 : pcoding_image.shape[1]
                    ]
                    points = np.array([grid_x.flatten(), grid_y.flatten()])

                    # before had: #np.array([chosen_pixel_x, chosen_pixel_y]) for the below line but the results
                    # werent consistent with the idl code results. changing the x and y pixel coordinates here works
                    interp_at_points = np.array([pixel_y, pixel_x])

                    # see if thie other method works,
                    # https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
                    if verbose:
                        print("Calculating weights and vertices for sky projection\n")
                    vtx, wts = interp_weights(points.T, interp_at_points.T)

                    # Initialize a table
                    hp_pixel_inds = np.arange(hp.nside2npix(_nside)).astype(int)
                    tab = Table([hp_pixel_inds], names=["hpix_ind"])

                    for k in range(_nebands + 1):
                        sky_val_hpx = np.zeros_like(val_hpx)
                        var_val_hpx = np.zeros_like(val_hpx)

                        values = sky_image[:, :, k]
                        if verbose:
                            print(
                                f"Reprojecting sky map of energy band {k+1} of {_nebands}"
                            )
                        test = interpolate(values.flatten(), vtx, wts, fill_value=0)

                        # These values are at the values of indices ipix
                        # so populate those
                        sky_val_hpx[ipix] += test

                        tab.add_column(sky_val_hpx, name=f"sky_e{k}")

                        # Do the same for variance maps
                        values = var_image[:, :, k]
                        if verbose:
                            print(
                                f"Reprojecting variance map of energy band {k+1} of {_nebands}"
                            )
                        test = interpolate(values.flatten(), vtx, wts, fill_value=0)

                        var_val_hpx[ipix] += test
                        tab.add_column(var_val_hpx, name=f"var_e{k}")

                    # Do the same for the partial coding fractions
                    if verbose:
                        print(f"Reprojecting partial coding map")
                    hpx_pcode_image = np.zeros_like(val_hpx)
                    values = pcoding_image.flatten()
                    values[np.isnan(values)] = 0
                    test = interpolate(values, vtx, wts, fill_value=0)
                    hpx_pcode_image[ipix] += test

                    tab.add_column(hpx_pcode_image * pointing_exposure, name="exposure")

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
                        f["mosaic"].attrs["nside"] = _nside
                        f["mosaic"].attrs["TSTART"] = pointing_tstart
                        f["mosaic"].attrs["TSTOP"] = pointing_tstop
                        f["mosaic"].attrs["DATE-OBS"] = pointing_dateobs_start
                        f["mosaic"].attrs["DATE-END"] = pointing_dateobs_end
                        f["mosaic"].attrs["EXPOSURE"] = pointing_exposure
                        f["mosaic"].attrs["obsid"] = obsid
                        f["mosaic"].attrs["pointing_id"] = str(pointing_id)
                        f["mosaic"].attrs["BSURVER"] = survey_ver
                        f.close()
                    processed = True
                else:
                    processed = False

                img_hdu.close()
                var_hdu.close()

            processed_mask[i] = processed
        return obsid, pointing_ids[processed_mask]
    else:
        return obsid, np.array([])


def do_sky_reprojection_healpix(obsdir, verbose=True):
    """Helper function to reproject the given sky image on to a global grid.
    This will be useful for creating all sky mosaics. By default, all one needs
    is a directory containing the sky images to be reprojected. This assumes that
    they are already processed using the batsurvey module in BatAnalysis as it will
    try to read the pickle file that is created when the batsurvey object is saved.

    Args:
        obsdir (str): The path to the directory containing the raw data
        verbose (bool, optional): Whether to print out information during processing. Defaults to True.
    """
    # First make the survey object
    # parse the obsid from the obsdir
    obs_dir = Path(obsdir)
    survey_obj = BatSurvey(obs_id=obs_dir.name, obs_dir=obs_dir.parent, recalc=False)

    # Using this read the stats file of interest
    stat_file = survey_obj.result_dir.joinpath("stats_point.fits")
    stats_data = fits.getdata(stat_file.as_posix(), 1)

    # Also read in corrections map
    corrections_map = read_correctionsmap()

    # Only proceed of there are good pointings
    if len(stats_data["NBATDETS"]) > 0:
        # calculate the mask of which points we should use based on good image statistics
        chi_mask = compute_statistics_map(
            stats_data["CHI2"],
            stats_data["NBATDETS"],
            stats_data["RA_PNT"],
            stats_data["DEC_PNT"],
            stats_data["PA_PNT"],
            stats_data["TSTART"],
        )

        # test that we have good image statistics for every pointing
        for i in range(len(stats_data["NBATDETS"])):
            if (
                (chi_mask[i] == 0)
                or (stats_data["NBATDETS"][i] <= 0)
                or (stats_data["IMAGE_STATUS"][i] == False)
                or (stats_data["EXPOSURE"][i] <= 0)
            ):
                if verbose:
                    print(
                        f"Bad image Statistics. Skipping observation ID/Pointing: {survey_obj.obs_id}/{survey_obj.pointing_ids[i]}\n"
                    )
            else:
                if verbose:
                    print(
                        f"Good image Statistics. Working on observation ID/Pointing: {survey_obj.obs_id}/{survey_obj.pointing_ids[i]}\n"
                    )

                obsid = survey_obj.obs_id
                pointing_id = survey_obj.pointing_ids[i]
                ncleaniter = survey_obj.batsurvey_result.params["ncleaniter"]
                data_directory = survey_obj.result_dir.joinpath(f"point_{pointing_id}")
                # read the partial coding map, variance map, sky flux map for the pointing
                pointing_img_str = data_directory.joinpath(
                    f"point_{pointing_id}_{ncleaniter}.img"
                )
                pointing_var_str = data_directory.joinpath(
                    f"point_{pointing_id}_{ncleaniter}.var"
                )

                # Then read all the information about the pointing
                img_hdu = fits.open(pointing_img_str.as_posix())
                var_hdu = fits.open(pointing_var_str.as_posix())
                # read the partial coding map

                # get other header information
                survey_ver = img_hdu["BAT_PCODE_1"].header["BSURVER"]
                pointing_exposure = img_hdu["BAT_PCODE_1"].header["EXPOSURE"]
                pointing_tstart = img_hdu["BAT_PCODE_1"].header["TSTART"]
                pointing_tstop = img_hdu["BAT_PCODE_1"].header["TSTOP"]
                pointing_dateobs_start = img_hdu["BAT_PCODE_1"].header["DATE-OBS"]
                pointing_dateobs_end = img_hdu["BAT_PCODE_1"].header["DATE-END"]

                # Only proceed, if it exceeds a minimum exposure criterion
                if pointing_exposure >= _minexpo:

                    pcoding_image = img_hdu["BAT_PCODE_1"].data
                    # save the header for use later
                    pcoding_header = img_hdu[0].header

                    # get the image size and create array to hold the sky flux at each channel
                    sz = pcoding_image.shape

                    # Make the sky image
                    sky_image = np.zeros(
                        (sz[0], sz[1], _nebands)
                    )  # plus 1 for the total energy

                    # Make the sky variance image
                    var_image = np.zeros_like(sky_image)
                    for k in range(_nebands):
                        sky_image[:, :, k] = img_hdu[k].data
                        var_image[:, :, k] = var_hdu[k].data

                    # correct for off axis effects
                    sky_image = np.divide(
                        sky_image,
                        corrections_map,
                        out=np.zeros_like(sky_image),
                        where=corrections_map != 0,
                    )
                    var_image = np.divide(
                        var_image,
                        corrections_map,
                        out=np.zeros_like(var_image),
                        where=corrections_map != 0,
                    )

                    # construct the total energy images for variance and flux, the zeros in last array dont affect
                    # calculations of the total values
                    # Fluxes add in a simple way
                    # sky_image[:, :, -1] = sky_image.sum(axis=2)

                    # # Variances add in quadrature
                    # var_image[:, :, -1] = np.sqrt(np.sum(var_image**2, axis=2))

                    # construct the quality map for each energy and for the total energy images
                    energy_quality_mask = np.zeros_like(sky_image) * np.nan
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

                    # Now do the actial sky projection using reproject. This will project the sky image
                    # onto a healpix grid. But nan handling isn't great in reproject so we will do this
                    # manually here by filling zeros

                    # The easiest (and efficient) way (for downstream purposes) is to use
                    # healpix for the reprojection. We will use the reproject package

                    # get the good values, in the idl file the shape of pointing_pimg is reversed, not sure if
                    # this is correct here.
                    sky_image *= energy_quality_mask
                    var_image *= energy_quality_mask

                    # sky_image[np.isnan(sky_image)] = 0
                    # var_image[np.isnan(var_image)] = 0

                    # Initialize a table
                    hp_pixel_inds = np.arange(hp.nside2npix(_nside)).astype(int)
                    tab = Table([hp_pixel_inds], names=["hpix_ind"])

                    # As we go on, append to this table

                    # Once this is done, do the total energy band as well. Making a sky image and
                    # reprojecting for the total band is costly, so add flux from healpix indices

                    sky_tot_en = np.zeros(len(hp_pixel_inds))
                    var_tot_en = np.zeros(len(hp_pixel_inds))
                    for k in range(_nebands):

                        if verbose:
                            print(
                                f"Reprojecting sky map of energy band {k+1} of {_nebands}"
                            )
                        hpx_sky_image, hpx_footprint = reproject_to_healpix(
                            input_data=(sky_image[:, :, k], img_hdu[0].header),
                            coord_system_out="icrs",
                            nside=_nside,
                        )
                        # Beware of nan's
                        sky_tot_en = np.nansum([sky_tot_en, hpx_sky_image], axis=0)

                        # Add error in quadrature
                        if verbose:
                            print(
                                f"Reprojecting variance map of energy band {k+1} of {_nebands}"
                            )
                        hpx_var_image, _ = reproject_to_healpix(
                            input_data=(var_image[:, :, k], var_hdu[0].header),
                            coord_system_out="icrs",
                            nside=_nside,
                        )

                        # Beware of nan's
                        var_tot_en = np.nansum([var_tot_en + hpx_var_image**2], axis=0)
                        tab.add_columns(
                            [hpx_sky_image, hpx_var_image],
                            names=[f"sky_e{k+1}", f"var_e{k+1}"],
                        )

                    # Add a footprint column
                    tab.add_column(hpx_footprint, name="footprint", index=1)
                    tab.add_column(sky_tot_en, name="sky_etot")
                    tab.add_column(np.sqrt(var_tot_en), name="var_etot")

                    # Do the same for the partial coding fractions
                    # This is redundant since partical coding information is already folded in
                    # hpx_pcode_image, _ = reproject_to_healpix(
                    #     input_data=(pcoding_image, pcoding_header),
                    #     coord_system_out="icrs",
                    #     nside=_nside,
                    # )
                    # tab.add_column(hpx_pcode_image, name="pcode")

                    outfile = data_directory.joinpath(
                        f"point_{pointing_id}_{ncleaniter}_healpix_projected.h5"
                    )
                    with h5py.File(outfile, "w") as f:
                        # Store data
                        if verbose:
                            print(f"Writing reprojected data to {outfile}")
                        f.create_dataset(
                            "mosaic", data=tab, compression="gzip", compression_opts=4
                        )

                        # Store header as attributes
                        f["mosaic"].attrs["TSTART"] = pointing_tstart
                        f["mosaic"].attrs["TSTOP"] = pointing_tstop
                        f["mosaic"].attrs["DATE-OBS"] = pointing_dateobs_start
                        f["mosaic"].attrs["DATE-END"] = pointing_dateobs_end
                        f["mosaic"].attrs["EXPOSURE"] = pointing_exposure
                        f["mosaic"].attrs["obsid"] = obsid
                        f["mosaic"].attrs["pointing_id"] = str(pointing_id)
                        f["mosaic"].attrs["BSURVER"] = survey_ver
                img_hdu.close()
                var_hdu.close()


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
        super().__init__(obs_id=obsid, obs_dir=obs_dir, recalc=False)
        # Always set recalc to False for the parent class, this
        # is not meant to run the batsurvey again

        # Now use batsurvey class to call in parameters

        load_file = self.result_dir.joinpath("batsurvey.pickle")
        obsdir = self.obs_dir.expanduser().resolve()

        reprojection_status_file = self.result_dir.joinpath(".batreproject")
        reprojection_obj_file = self.result_dir.joinpath("batreproject.pickle")
        batsurvey_res_file = self.result_dir.joinpath("stats_point.fits")

        # See if a loadfile exists, if not dont proceed
        if load_file.exists() and batsurvey_res_file.exists():
            self._nside = _nside

            # Then use this to run the reprojection
            # if there is already a file load it, if not run it
            if (
                not reprojection_obj_file.exists()
                or not reprojection_status_file.exists()
                or recalc
            ):
                if reprojection_status_file.exists() and recalc:
                    print(
                        f"""A previous reproject status file {reprojection_obj_file} is found. But recalculating"""
                    )
                else:
                    print(
                        f"""No existing reproject status file {reprojection_obj_file} found.\
                        Running reprojection now."""
                    )

                _, reprojected_pointings = do_sky_reprojection_hybrid(
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
                reprojection_status_file.touch()

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


class MosaicSummary:
    """
    A general mosaic object that holds all information about mosaiced reprojected survey images.

    Attributes
    ---------------
    mosaic_dir : str
        observation ID
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
        Saves a MosaicReprojectedSurvey object
    """

    def __init__(self, obs_dir, snr_thresh=5):
        """
        Initializer method for the ReprojectSurvey object.

        :param obsdir: The path to the mosaic directory
        :param recalc: Boolean default False, which indicates that the method should try to load data from a file in
            the mosaic_dir directory. True means that the load file will be ignored and attributes will be re-obtained
            for the object.
        :param verbose: Boolean default False, which indicates where to print informative messages during the processing.
        """

        self.obs_dir = Path(obs_dir).expanduser().resolve()
        obj_file = self.obs_dir.joinpath("batmosaic.pickle")
        self.snr = snr_thresh

        # if there is already a file load it, if not run it
        if not obj_file.exists():
            # Get the summary of all the observations in the mosaic directory
            hpx_file = [i for i in self.obs_dir.glob("*_hybrid_healpix_projected.h5")]
            if len(hpx_file) == 0:
                raise FileNotFoundError(
                    f"No reprojected healpix files found in mosaic directory {self.obs_dir}"
                )
            self.mosaiced_hpx_file = hpx_file[0]

            # Then save all the images
            images = [i for i in self.obs_dir.glob("*_sky_image.fits")]
            images.sort()

            var_images = [
                Path(i.as_posix().replace("_sky_image", "_var_image")) for i in images
            ]
            exp_image = [
                Path(i.as_posix().replace("_sky_image", "_exp_image")) for i in images
            ]

            self.images = images
            self.var_images = var_images
            self.exp_image = exp_image
            self.nimages = len(images)

            # Run batcelldetect on these images
            self.detect_sources()

            # Save the object
            self._save(obj_file)
        else:
            self._load(obj_file)

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

    def _call_batcelldetect(self, input_dict):
        """
        Call heasoftpy batcelldetect.

        :param input_dict: dictionary of inputs to pass to batcelldetet.
        :return: heasoft output object
        """
        # make the local pfile dir if it doesnt exist and set this value
        out = hsp.batcelldetect(**input_dict)
        return out

    def detect_sources(self):
        """Run batcelldetect on the mosaiced sky images to detect sources"""

        # Get a list of all BAT sources
        bat_source_file = Path(__file__).parent.joinpath("data/survey6b_2.cat")

        default_input_dict = dict(
            incatalog=str(bat_source_file),
            snrthresh=self.snr,
            psfshape="GAUSSIAN",
            psffwhm=0.37413,
            srcfit="YES",
            posfit="NO",
            pospeaks="YES",
            posfluxfit="YES",
            posfitwindow=0.37413,
            bkgwindowtype="SMOOTH_CIRCLE",
            srcdetect="YES",
            nadjpix=3,
            srcradius=12,
            bkgradius=50,
            bkgfit="YES",
            keepbits="ALL",
            hduclasses="NONE",
            chatter=3,
            clobber="YES",
            distfile="NONE",
            nullborder="NO",
            vectorflux="YES",
            vectorposmeth="MAX_SNR",
            keepkeywords="FACET,CRVAL1,CRVAL2,*VER",
        )

        # For the list of images, run batcelldetect
        source_catalogs = []
        batcelldetect_output = []
        for i in range(self.nimages):
            file = self.images[i]
            varfile = self.var_images[i]
            expfile = self.exp_image[i]

            catalog_file = Path(file.as_posix().replace("sky_image", "src_catalog"))
            snr_map = Path(file.as_posix().replace("sky_image", "snr_map"))

            # Get the imaging region where source estimation has to be performed
            exp_data = fits.getdata(expfile.as_posix(), 0)
            # Cut off when the exposure time is < 5% of the max exposure time
            exp_cut = 0.1 * np.nanmax(exp_data)
            default_input_dict["infile"] = file.as_posix()
            default_input_dict["outfile"] = catalog_file.as_posix()
            default_input_dict["inbkgvarmap"] = varfile.as_posix()
            default_input_dict["pcodefile"] = expfile.as_posix()
            default_input_dict["pcodethresh"] = exp_cut
            default_input_dict["signifmap"] = snr_map.as_posix()

            print(f"Running batcelldetect on {file.name}\n")
            batcelldetect_output.append(self._call_batcelldetect(default_input_dict))
            source_catalogs.append(catalog_file.as_posix())
        self.batcelldetect_output = dict(zip(source_catalogs, batcelldetect_output))
        self.source_catalogs = source_catalogs

        # Then combine all the source catalogs into one
        comb_source_file = self.mosaiced_hpx_file.as_posix().replace(
            "_hybrid_healpix_projected.h5", "_comb_source_catalog.fits"
        )

        heatools.ftmerge(
            infile=",".join(self.source_catalogs),
            outfile=str(comb_source_file),
            clobber="YES",
        )

        # get the coordinates from galactic to RA/DEC
        heatools.ftcoco(
            infile=str(comb_source_file),
            outfile=str(comb_source_file),
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
            infile=f"{comb_source_file}[col *;FACET_DIST = ANGSEP(GLON_OBJ,GLAT_OBJ,CRVAL1,CRVAL2); DUP = F]",
            outfile=str(comb_source_file),
            columns="CATNUM, FACET_DIST",
            clobber="YES",
        )
        self.comb_source_file = comb_source_file

    def load_and_filter_sources(self, snr_thresh=3.0):
        """Load the combined source catalog and filter based on SNR threshold.

        Args:
            snr_thresh (float, optional): The SNR threshold to filter sources. Defaults to 8.0.

        Returns:
            Table: Filtered source catalog.
        """
        # Load the combined source catalog
        source_data = Table.read(self.comb_source_file)

        # Filter sources based on SNR threshold
        filtered_sources = source_data[np.max(source_data["SNR"], axis=1) >= snr_thresh]

        # Now select and remove BAT persistent sources
        # persistent_source_file = Path(__file__).parent.joinpath(
        #     "data/swift_bat_157_month_catalog.dat"
        # )
        # persistent_sources = Table.read(
        #     persistent_source_file.as_posix(), format="ascii"
        # )
        # bat_source_coords = SkyCoord(
        #     ra=persistent_sources["RA"],
        #     dec=persistent_sources["DEC"],
        #     unit="deg",
        #     frame="icrs",
        # )
        # coords = SkyCoord(
        #     b=filtered_sources["GLAT_OBJ"],
        #     l=filtered_sources["GLON_OBJ"],
        #     unit="deg",
        #     frame="galactic",
        # )
        # _, sep, _ = coords.match_to_catalog_sky(bat_source_coords)
        # sep_mask = sep.arcmin >= 22  # PSF of BAT
        # self.transient_sources = filtered_sources[sep_mask]
        names = np.array([i.strip() for i in filtered_sources["NAME"]])
        self.transient_sources = filtered_sources[names == "UNKNOWN"]

        # Add a column with max snr
        self.transient_sources.add_column(
            np.max(self.transient_sources["SNR"], axis=1), name="MAX_SNR"
        )
        self.transient_sources.sort(["MAX_SNR", "FACET_DIST"], reverse=True)

    def plot_cutouts(self, sources, filename=None, snr_limit=3):
        """Plot cutouts around the given sources from the mosaiced sky images.

        Args:
            sources (Table): Table of sources with RA and DEC columns.
        """
        if filename is None:
            filename = self.mosaiced_hpx_file.as_posix().replace(
                "_hybrid_healpix_projected.h5", "_source_cutouts.pdf"
            )
        plotfile = PdfPages(filename)

        sources = sources[sources["MAX_SNR"] >= snr_limit]
        # Get all coords
        coords = SkyCoord(
            sources["GLON_OBJ"], sources["GLAT_OBJ"], unit="deg", frame="galactic"
        )

        for i in range(len(sources)):
            c = coords[i]
            c_icrs = c.transform_to("icrs")
            file = [
                fil
                for fil in self.mosaiced_hpx_file.parent.glob(
                    f"*_l_{sources['CRVAL1'][i]:03.0f}_b_{sources['CRVAL2'][i]:+03.0f}_STG_sky_image.fits"
                )
            ][0]
            enbin = np.argmax(sources["SNR"][i])
            hdu = fits.open(file.as_posix())[enbin + 1]

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
                (cut.data.shape[1] / 2, cut.data.shape[0] / 2),
                radius=22 / 2.8 / 2,
                edgecolor="red",
                facecolor="none",
                fill=False,
                lw=2,
            )
            ax.add_patch(patch)

            # Make the title
            ax.set_title(
                f"Source at l={c_icrs.ra.deg:.3f}, b={c_icrs.dec.deg:.3f}, SNR={sources['MAX_SNR'][i]:.2f}",
                fontsize=15,
            )
            ax.tick_params(labelsize=10)
            ax.set_xlabel("Galactic Longitude (deg)", fontsize=12)
            ax.set_ylabel("Galactic Latitude (deg)", fontsize=12)
            plt.tight_layout()
            plotfile.savefig()
            plt.close()
        plotfile.close()
