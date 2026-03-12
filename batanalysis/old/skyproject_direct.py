"""
This file is meant to hold the functions that allow users to create mosaic-ed images for survey data
"""

import shutil
from pathlib import Path
import copy
import glob
import pickle
import numpy as np
from astropy import units as u
import scipy.spatial.qhull as qhull
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table

from .bat_survey import MosaicBatSurvey, BatSurvey
from .batlib import dirtest, met2utc, datadir
from .mosaic import (
    interp_weights,
    interpolate,
    read_correctionsmap,
    compute_statistics_map,
    convert_radec2xy,
)
from .skyproject import (
    add_with_respect,
    divide_with_respect,
    convert_time_to_decimal_day,
    convert_time_to_mettime,
)

# Off-axis flux correction file
# _cimgfile = "offaxiscorr_8bin_20061221.img"
# _chilothresh = 0.50  # Minimum chi-square for any energy band
# _chihithresh = 1.15  # Maximum chi-square for any energy band
# _chiscobump = 0.35  # Additional bump of chi-square threshold around Sco X-1 (band 0)
# _chiscotheta = 30  # Approximate angular scale of bump around Sco X-1 (deg)
_pcodethresh = 0.15  # Minimum image partial coding
_minexpo = 150  # Minimum image exposure
_nskyimg = 6  # Number of facets to sky image
_nebands = 8  # Number of energy bands to process
_proj = "ZEA"  # projection from idl code that is used

# Sco X-1 ra and dec
# _scox1_ra = 245.100
# _scox1_dec = -15.600
# _sco_coord = SkyCoord(_scox1_ra, _scox1_dec, frame="icrs", unit="deg")


def make_fits_hdul(
    img,
    header,
    emin=[14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 14.0],
    emax=[20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0, 195.0],
):
    """
    Write out the projected images to fits files.

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

    # read in the appropriate headers and create new headers and save files
    total_header = header + add_header

    all_hduls = []

    for i in range(_nskyimg):
        string = "c%d_%s" % (i, _proj)
        ra_string = "ra_" + string + ".img"

        ra_file = (
            Path(__file__).parent.joinpath("data").joinpath(ra_string)
        )  # os.path.join(direc, "data", ra_string)
        with fits.open(str(ra_file)) as file:
            skygrid_header = file[0].header

        total_header = header + add_header + skygrid_header
        total_header["BSKYPLAN"] = (string, "BAT mosaic ZEA sky plane ID (0-5)")

        if img.ndim == 3:
            # if this the pimg or eimg
            all_hduls.append(
                fits.HDUList(fits.PrimaryHDU(data=img[:, :, i], header=total_header))
            )
        else:
            # This means it is the sky image
            hdul = [fits.PrimaryHDU()]
            for j in range(len(emin)):
                # if this is the variance or sky flux image need ot add energy related header keys
                total_header["BENRGYBN"] = (
                    f"E_{int(emin[j]):03}_{int(emax[j]):03}",
                    "BAT mosaic energy bin (keV)",
                )
                total_header["E_MIN"] = (emin[j], " [keV] Lower energy bin edge")
                total_header["E_MAX"] = (emax[j], " [keV] Upper energy bin edge")

                total_header["HDUNAME"] = (
                    f"BAT_IMAGE{j+1}",
                    "BAT mosaic energy bin (keV)",
                )
                hdul.append(fits.ImageHDU(data=img[:, :, i, j], header=total_header))
            all_hduls.append(hdul)

    return all_hduls


def project_images(img, var, sky_grids, verbose=True):
    """Actual function that does the projection on to a global grid

    Args:
        img (str): Path to the image
        var (str): Path to the variance map
    """
    ra_skygrid, dec_skygrid = sky_grids
    corrections_map = read_correctionsmap()

    img_hdu = fits.open(img)
    var_hdu = fits.open(var)
    # read the partial coding map

    # get other header information
    pointing_exposure = img_hdu["BAT_PCODE_1"].header["EXPOSURE"]
    pointing_tstart = img_hdu["BAT_PCODE_1"].header["TSTART"]
    pointing_tstop = img_hdu["BAT_PCODE_1"].header["TSTOP"]
    pointing_dateobs_start = img_hdu["BAT_PCODE_1"].header["DATE-OBS"]
    pointing_dateobs_end = img_hdu["BAT_PCODE_1"].header["DATE-END"]

    pcoding_image = img_hdu["BAT_PCODE_1"].data
    # save the header for use later
    pcoding_header = img_hdu[0].header

    # get the image size and create array to hold the sky flux at each channel
    sz = pcoding_image.shape

    # Make the sky image
    sky_image = np.zeros((sz[0], sz[1], _nebands + 1))  # plus 1 for the total energy

    # Make the sky variance image
    var_image = np.zeros_like(sky_image)
    for k in range(_nebands):
        sky_image[:, :, k] = img_hdu[k].data
        var_image[:, :, k] = var_hdu[k].data

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
    exposure_img = energy_quality_mask[:, :, 0] * pcoding_image * pointing_exposure

    # Multiplying the energy quality mask to the sky and variance images makes the masked regions
    # 0s which makes the interpolation tricky as it will average 0's and actual values together.
    # This will produce hotspots of islands with very low variance and hence false positives in source
    # detection, rather make them nan and handle them during interpolation

    sky_image[np.isnan(energy_quality_mask)] = np.nan
    var_image[np.isnan(energy_quality_mask)] = np.nan

    if verbose:
        print("Getting the common sky grid\n")
    # get the common healpix grid

    if verbose:
        print("Calculating pixel coordinates for reprojection\n")
    # Now get pixel indices

    # need to compute the x/y position for each RA/DEC point in the sky map using the new
    # file for the pointing of interest
    pixel_x, pixel_y = convert_radec2xy(ra_skygrid, dec_skygrid, pcoding_header)

    # get the good values, in the idl file the shape of pointing_pimg is reversed, not sure if
    # this is correct here.
    pixel_idx = np.where(
        (pixel_y <= pcoding_image.shape[0])
        & (pixel_x <= pcoding_image.shape[1])
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
    grid_x, grid_y = np.mgrid[0 : pcoding_image.shape[0], 0 : pcoding_image.shape[1]]
    points = np.array([grid_x.flatten(), grid_y.flatten()])

    interp_at_points = np.array([chosen_pixel_y, chosen_pixel_x])
    vtx, wts = interp_weights(points.T, interp_at_points.T)

    # Now start projection
    # Start with the net exposure map
    # Now make a dummy image to fill values
    proj_exp_map = np.zeros_like(ra_skygrid) * np.nan
    values = exposure_img.flatten()
    interp_vals = interpolate(values, vtx, wts, fill_value=np.nan)
    proj_exp_map[pixel_idx] = interp_vals

    # Make images of dimensions (x, y, sky_tiles, energy_bands)
    nx, ny, nsky = ra_skygrid.shape
    proj_sky_image = np.zeros((nx, ny, nsky, _nebands + 1)) * np.nan
    proj_var_image = np.zeros_like(proj_sky_image) * np.nan

    for k in range(_nebands + 1):

        # Do it for sky image
        values = sky_image[:, :, k].flatten()
        interp_vals = interpolate(values, vtx, wts, fill_value=np.nan)
        proj_sky_image[:, :, :, k][pixel_idx] = interp_vals

        # Also do it for variance map
        values = var_image[:, :, k].flatten()
        interp_vals = interpolate(values, vtx, wts, fill_value=np.nan)
        proj_var_image[:, :, :, k][pixel_idx] = interp_vals  # new interpolate dir

    model_hdr = fits.Header()
    model_hdr["TSTART"] = (pointing_tstart, " start time of image")
    model_hdr["TSTOP"] = (pointing_tstop, " stop time of image")
    model_hdr["TELAPSE"] = (
        pointing_tstop - pointing_tstart,
        "  elapsed time of image (= TSTOP-TSTART)",
    )
    model_hdr["DATE-OBS"] = (
        pointing_dateobs_start,
        "  TSTART, expressed in UTC",
    )
    model_hdr["DATE-END"] = (
        pointing_dateobs_end,
        "  TSTOP, expressed in UTC",
    )

    model_hdr["EXPOSURE"] = (
        pointing_tstart - pointing_tstart,
        "[sec.] net exposure",
    )

    # Add info about the user specified TBIN that was used to create the mosaic
    # add/modify extra stuff for pcoding*exp image
    model_hdr["HDUCLAS2"] = (
        "VIGNETTING",
        " Contains partial coding map <== PCODE*EXP",
    )
    model_hdr["IMATYPE"] = ("EXPOSURE", " Contains partial coding map ")
    model_hdr["BUNIT"] = ("s ", " Exposure map")
    proj_exp_hdul = make_fits_hdul(proj_exp_map, model_hdr)

    # add/modify extra stuff for variance image
    model_hdr["HDUCLAS2"] = (
        "VAR",
        " The projected variation (std) map",
    )
    model_hdr["IMATYPE"] = ("VARIATION", "Variation map")
    model_hdr["BUNIT"] = (
        "counts/sec",
        " Physical units for std image",
    )
    proj_var_hdul = make_fits_hdul(proj_var_image, model_hdr)

    # add/modify extra stuff for sky flux image
    model_hdr["HDUCLAS2"] = (
        "SKY",
        "Projected sky image ",
    )
    model_hdr["IMATYPE"] = ("INTENSITY", " Contains projected sky flux map")
    model_hdr["BUNIT"] = (
        "(counts/sec)",
        " Physical units for flux image",
    )
    proj_sky_hdul = make_fits_hdul(proj_sky_image, model_hdr)

    return proj_sky_hdul, proj_var_hdul, proj_exp_hdul


def project_observation(obsdir, skygrids, verbose=True):
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

    proj_dir = survey_obj.result_dir.joinpath("skyproject/")
    if not proj_dir.exists:
        dirtest(proj_dir.as_posix(), clean_dir=False)

    # Get some basic parameters about pointing
    obsid = survey_obj.obs_id

    ncleaniter = survey_obj.batsurvey_result.params["ncleaniter"]

    # Using this read the stats file of interest
    stat_file = survey_obj.result_dir.joinpath("stats_point.fits")
    stats_data = fits.getdata(stat_file.as_posix(), 1)

    # Make a new directory for projected images

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
                pointing_info = fits.open(pointing_img_str)

                # get other header information
                pointing_exposure = pointing_info["BAT_PCODE_1"].header["EXPOSURE"]
                pointing_info.close()
                # Only proceed, if it exceeds a minimum exposure criterion
                if pointing_exposure >= _minexpo:

                    # Then create subdirectories inside the prjected directories
                    point_dir = proj_dir.joinpath(f"point_{pointing_id}")
                    dirtest(point_dir.as_posix(), clean_dir=False)

                    basepath = point_dir.joinpath(f"point_{pointing_id}_{ncleaniter}")
                    proj_sky_hdul, proj_var_hdul, proj_exp_hdul = project_images(
                        img=pointing_img_str,
                        var=pointing_var_str,
                        sky_grids=skygrids,
                        verbose=verbose,
                    )

                    # Now write them
                    for sk in range(_nskyimg):
                        proj_exp_hdul[sk].writeto(
                            f"{str(basepath)}_c{sk}_{_proj}_exp_image.fits",
                            overwrite=True,
                        )
                        fits.HDUList(proj_sky_hdul[sk]).writeto(
                            f"{str(basepath)}_c{sk}_{_proj}_sky_image.fits",
                            overwrite=True,
                        )
                        fits.HDUList(proj_var_hdul[sk]).writeto(
                            f"{str(basepath)}_c{sk}_{_proj}_var_image.fits",
                            overwrite=True,
                        )
                    processed = True
                else:
                    processed = False

            processed_mask[i] = processed
        return obsid, pointing_ids[processed_mask]
    else:
        return obsid, np.array([])


def add_images(image_list):
    """Main function that does the mosaicking. Takes in a list of sky images, it assumes that all sky facets
    are included in the image list.. It will try to build in the variance map names and exposure map from the
    image list

    Args:
        image_list (list): List of paths to the image files
    """
    # There will be _nskyimg facets, so analyze them one by one

    dummy_name = f"c0_ZEA"
    dummy_image_mask = [True if dummy_name in i else False for i in image_list]
    dummy_images = np.array(image_list)[dummy_image_mask]

    # First get the dimensions of the image
    dummy_data = fits.getdata(dummy_images[0], 1)
    nx, ny = dummy_data.shape
    nz = _nebands + 1

    intermediate_exp_data = np.zeros((nx, ny, _nskyimg)) * np.nan
    intermediate_sky_image = np.zeros((nx, ny, _nskyimg, nz)) * np.nan
    intermediate_var_image = np.zeros((nx, ny, _nskyimg, nz)) * np.nan

    # Now load in each file and store in the intermediate arrays
    tmin = 1e15
    tmax = 0
    for i in range(len(dummy_images)):
        for j in range(_nskyimg):
            print(
                f"Adding image {i+1} of {len(dummy_images)}, facet {j+1} of {_nskyimg}\n"
            )
            ind_facet_name = f"c{j}_ZEA"
            ind_fact_sky_images = [
                f.replace("c0_ZEA", ind_facet_name) for f in dummy_images
            ]
            ind_facet_var_images = [
                f.replace("sky_image", "var_image") for f in ind_fact_sky_images
            ]
            ind_facet_exp_images = [
                f.replace("sky_image", "exp_image") for f in ind_fact_sky_images
            ]

            # Load in exposure image and save data
            with fits.open(ind_facet_exp_images[i]) as exp_hdu:
                intermediate_exp_data[:, :, j] = add_with_respect(
                    intermediate_exp_data[:, :, j], exp_hdu[0].data
                )
                if j == 0:
                    model_exp_hdr = exp_hdu[0].header
                    tmin = np.min([tmin, model_exp_hdr["TSTART"]])
                    tmax = np.max([tmax, model_exp_hdr["TSTOP"]])

            # Next load in sky and var images
            with fits.open(ind_fact_sky_images[i]) as sky_hdu, fits.open(
                ind_facet_var_images[i]
            ) as var_hdu:
                for k in range(_nebands + 1):
                    intermediate_sky_image[:, :, j, k] = add_with_respect(
                        intermediate_sky_image[:, :, j, k],
                        divide_with_respect(
                            sky_hdu[k + 1].data, var_hdu[k + 1].data ** 2
                        ),
                    )
                    intermediate_var_image[:, :, j, k] = add_with_respect(
                        intermediate_var_image[:, :, j, k],
                        divide_with_respect(1, var_hdu[k + 1].data ** 2),
                    )
    # Now that data is read, do final calculation
    # Make combined sky and var images
    print("Finalizing combined images\n")
    combined_sky_image = divide_with_respect(
        intermediate_sky_image, intermediate_var_image
    )
    combined_var_image = divide_with_respect(1, np.sqrt(intermediate_var_image))

    # Add in other key words to the headers

    model_hdr = fits.Header()
    model_hdr["TSTART"] = (tmin, " start time of image")
    model_hdr["TSTOP"] = (tmax, " stop time of image")
    model_hdr["TELAPSE"] = (
        tmax - tmin,
        "  elapsed time of image (= TSTOP-TSTART)",
    )
    model_hdr["DATE-OBS"] = (
        convert_time_to_mettime(tmin, reverse=True).isot,
        "  TSTART, expressed in UTC",
    )
    model_hdr["DATE-END"] = (
        convert_time_to_mettime(tmax, reverse=True).isot,
        "  TSTOP, expressed in UTC",
    )

    model_hdr["EXPOSURE"] = (
        np.nanmax(intermediate_exp_data),
        "[sec.] net exposure",
    )

    # Add info about the user specified TBIN that was used to create the mosaic
    # add/modify extra stuff for pcoding*exp image
    model_hdr["HDUCLAS2"] = (
        "VIGNETTING",
        " Contains partial coding map <== PCODE*EXP",
    )
    model_hdr["IMATYPE"] = ("EXPOSURE", " Contains partial coding map ")
    model_hdr["BUNIT"] = ("s ", " Exposure map")
    combined_exp_hdul = make_fits_hdul(intermediate_exp_data, model_hdr)

    # add/modify extra stuff for variance image
    model_hdr["HDUCLAS2"] = (
        "VAR",
        " The projected variation (std) map",
    )
    model_hdr["IMATYPE"] = ("VARIATION", "Variation map")
    model_hdr["BUNIT"] = (
        "counts/sec",
        " Physical units for std image",
    )
    combined_var_hdul = make_fits_hdul(combined_var_image, model_hdr)

    # add/modify extra stuff for sky flux image
    model_hdr["HDUCLAS2"] = (
        "SKY",
        "Projected sky image ",
    )
    model_hdr["IMATYPE"] = ("INTENSITY", " Contains projected sky flux map")
    model_hdr["BUNIT"] = (
        "(counts/sec)",
        " Physical units for flux image",
    )
    combined_sky_hdul = make_fits_hdul(combined_sky_image, model_hdr)
    return combined_sky_hdul, combined_var_hdul, combined_exp_hdul


def make_mosaics(
    outventory_file,
    start,
    end,
    obsdir=None,
    dt=None,
    recalc=False,
    use_intermediate=False,
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

    if obsdir is None:
        obsdir = datadir()

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
            obsids = [Path(f"{str(obsdir)}/{i}_surveyresult") for i in obsids]
            pointing_ids = [
                i.replace("point_", "") for i in grouped_outventory_data["IMAGE_ID"]
            ]

            # net_hdr_exposure = tstop - tstart
            # For each obsids, get the reprojected image
            # Which entails making a ReprojectSurvey object
            reproj_obj_list = [ReprojectSurvey(obs_dir=obsid) for obsid in obsids]

            # Get the number of pointings
            # Only select those pointings that fall in the time bin
            reproj_pointings = []
            reproj_images = []
            for i in range(len(reproj_obj_list)):
                point_mask = np.isin(
                    reproj_obj_list[i].reprojected_pointings, pointing_ids
                )
                select_pointings = np.array(reproj_obj_list[i].reprojected_pointings)[
                    point_mask
                ]
                reproj_pointings = np.concatenate(
                    (
                        reproj_pointings,
                        select_pointings,
                    )
                )
                for sp in select_pointings:
                    reproj_images = np.concatenate(
                        (
                            reproj_images,
                            reproj_obj_list[i].reprojected_images[sp],
                        )
                    )

            if verbose:
                print(
                    f"Found {len(reproj_pointings)} reprojected pointings, mosaicking them\n"
                )
            if len(reproj_images) > 0:
                proj_sky_hdul, proj_var_hdul, proj_exp_hdul = add_images(
                    image_list=reproj_images
                )
                basepath = img_dir.joinpath(f"mosaic_{start_time_str}_{end_time_str}")
                # Now write them
                for sk in range(_nskyimg):
                    proj_exp_hdul[sk].writeto(
                        f"{str(basepath)}_c{sk}_{_proj}_exp_image.fits",
                        overwrite=True,
                    )
                    fits.HDUList(proj_sky_hdul[sk]).writeto(
                        f"{str(basepath)}_c{sk}_{_proj}_sky_image.fits",
                        overwrite=True,
                    )
                    fits.HDUList(proj_var_hdul[sk]).writeto(
                        f"{str(basepath)}_c{sk}_{_proj}_var_image.fits",
                        overwrite=True,
                    )
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
                    [h["DATE-END"] for h in all_intermediate_mosaic_hdrs], scale="tai"
                )
                # Now select the ones that fall within the time bin
                mask = (all_intermediate_mosaic_tstarts >= start) & (
                    all_intermediate_mosaic_tstops <= end
                )

                intermediate_mosaics_to_use = np.array(all_intermediate_mosaics)[mask]

                # Only process these mosaics
                proj_sky_hdul, proj_var_hdul, proj_exp_hdul = add_images(
                    image_list=intermediate_mosaics_to_use
                )
                basepath = img_dir.joinpath(f"mosaic_{start_time_str}_{end_time_str}")
                # Now write them
                for sk in range(_nskyimg):
                    proj_exp_hdul[sk].writeto(
                        f"{str(basepath)}_c{sk}_{_proj}_exp_image.fits",
                        overwrite=True,
                    )
                    fits.HDUList(proj_sky_hdul[sk]).writeto(
                        f"{str(basepath)}_c{sk}_{_proj}_sky_image.fits",
                        overwrite=True,
                    )
                    fits.HDUList(proj_var_hdul[sk]).writeto(
                        f"{str(basepath)}_c{sk}_{_proj}_var_image.fits",
                        overwrite=True,
                    )
            else:
                raise ValueError(
                    "No existing intermediate mosaics found that are integral factors of the requested time bin"
                )
        img_dir.joinpath(".batreproject").touch()
    else:
        if verbose:
            print(f"Reprojected mosaic already exists in {img_dir}, not recalculating")


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

    def __init__(
        self, obs_dir=None, recalc=False, skygrids=None, recache=False, verbose=False
    ):
        """
        Initializer method for the ReprojectSurvey object.

        :param obsid: The observational ID to be analyzed
        :param recalc: Boolean default False, which indicates that the method should try to load data from a file in
            the mosaic_dir directory. True means that the load file will be ignored and attributes will be re-obtained
            for the object.
        :param verbose: Boolean default False, which indicates where to print informative messages during the processing.
        """
        obs_name = obs_dir.name.replace("_surveyresult", "")
        super().__init__(obs_id=obs_name, obs_dir=obs_dir.parent, recalc=False)
        del self.batsurvey_result
        del self.survey_input
        # Always set recalc to False for the parent class, this
        # is not meant to run the batsurvey again

        # Now use batsurvey class to call in parameters

        load_file = self.result_dir.joinpath("batsurvey.pickle")
        obsdir = self.obs_dir.expanduser().resolve()

        if skygrids is None:
            ra_grid, dec_grid = read_skygrids()
            skygrids = [ra_grid, dec_grid]

        reprojection_status_file = self.result_dir.joinpath(".batreproject")
        reprojection_obj_file = self.result_dir.joinpath("batreproject.pickle")
        batsurvey_res_file = self.result_dir.joinpath("stats_point.fits")

        # See if a loadfile exists, if not dont proceed
        if load_file.exists() and batsurvey_res_file.exists():

            # Then use this to run the reprojection
            # if there is already a file load it, if not run it
            if not reprojection_status_file.exists() or recalc:
                if reprojection_status_file.exists() and recalc:
                    print(
                        f"""A previous reproject status file {reprojection_obj_file} is found. But recalculating"""
                    )
                else:
                    print(
                        f"""No existing reproject status file {reprojection_obj_file} found.\
                        Running reprojection now."""
                    )

                _, reprojected_pointings = project_observation(
                    obsdir=obsdir, skygrids=skygrids, verbose=verbose
                )
                # Store path of the reprojected image
                self.reprojected_pointings = reprojected_pointings

                reprojected_images = {}
                for pid in reprojected_pointings:
                    images = glob.glob(
                        f"{obs_dir}/skyproject/point_{pid}/{pid}*_sky_image.fits"
                    )
                    reprojected_images[pid] = images
                self.reprojected_images = reprojected_images

                # Save it to pickle file
                self._save(reprojection_obj_file)

                # If it proceeded till here, it means that everthing was successful
                reprojection_status_file.touch()
            elif recache:
                # The user asks to make the pickle file again, so do it
                if not reprojection_status_file.exists():
                    raise FileNotFoundError(
                        f"It seems like the reprojection has not been done before, so can not make the pickle file. Please run reprojection first"
                    )
                else:
                    reprojected_pointings = glob.glob(f"{obs_dir}/skyproject/point*")
                    reprojected_pointings.sort()
                    self.reprojected_pointing_paths = reprojected_pointings
                    self.reprojected_pointings = [
                        Path(i).name.replace("point_", "")
                        for i in self.reprojected_pointing_paths
                    ]

                    reprojected_images = {}
                    for i, pid in enumerate(reprojected_pointings):
                        images = glob.glob(f"{pid}/*_sky_image.fits")
                        reprojected_images[self.reprojected_pointings[i]] = images
                    self.reprojected_images = reprojected_images
                    self._save(reprojection_obj_file)
            else:
                load_file = Path(reprojection_obj_file).expanduser().resolve()
                self._load(reprojection_obj_file)

        else:
            raise FileNotFoundError(
                f"Please run the batsurvey analysis first. There are no sky images to be reprojected for {obs_name}"
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
