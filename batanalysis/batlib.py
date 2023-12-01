"""
This file holds various functions that users can call to interface with bat observation objects
"""
import astropy as ap
from astropy.io import fits
from astropy.time import Time
import numpy as np
import shutil
import matplotlib.pyplot as plt
import os
import warnings
from pathlib import Path
import requests
from astroquery.heasarc import Heasarc
import swifttools.swift_too as swtoo
import datetime
import dpath
from concurrent.futures import ThreadPoolExecutor
import functools
import swiftbat.swutil as sbu

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)

_orig_pdir = os.getenv("PFILES")


def dirtest(directory, clean_dir=True):
    """
    Tests if a directory exists and either creates the directory or removes it and then re-creates it

    :param directory: String of the directory that should be created or deleted and re-created
    :param clean_dir: Boolean to denote if the directory should be deleted and recreated
    :return: None
    """

    directory = Path(directory)

    # see if the directory exists
    if directory.exists():
        if clean_dir:
            # remove the directory and recreate it
            shutil.rmtree(directory)
            directory.mkdir(parents=True)
    else:
        # create it
        directory.mkdir(parents=True)


def curdir():
    """
    Get the current working directory. Is legacy, since moving to use the pathlib module.
    """
    cdir = os.getcwd() + "/"
    return cdir


def datadir(new=None, mkdir=False, makepersistent=False, tdrss=False) -> Path:
    """Return the data directory (optionally changing and creating it)

    Args:
        new (Path|str, optional): Use this as the data directory
        mkdir (bool, optional): Create the directory (and its parents) if necessary
        makepersistent (bool, optional): If set, stores the name in ~/.swift/swift_datadir_name and uses it as new
            default
        tdrss (bool, optional): subdirectory storing tdrss data types
    """
    global _datadir
    datadirnamefile = Path("~/.swift/swift_datadir_name").expanduser()

    if new is not None:
        new = Path(new).expanduser().resolve()
        if mkdir:
            new.mkdir(parents=True, exist_ok=True)
            new.joinpath("tdrss").mkdir(exist_ok=True)
            new.joinpath("trend").mkdir(exist_ok=True)
        if makepersistent:
            persistfile = datadirnamefile
            persistfile.parent.mkdir(exist_ok=True)  # make ~/.swift if necessary
            persistfile.open("wt").write(str(new))
        _datadir = new

    if not globals().get("_datadir", False):
        # Not previously initialized
        try:
            _datadir = Path(datadirnamefile.open().read())
            if not _datadir.exists():
                raise RuntimeError(
                    f'Persistent data directory "{_datadir}" does not exist'
                )
        except FileNotFoundError:
            # No persistent directory exists.  Use cwd
            _datadir = Path.cwd()
            warnings.warn(f"Saving data in current directory {_datadir}")

    assert isinstance(_datadir, Path)
    if tdrss:
        return _datadir.joinpath("tdrss")
    return _datadir


def create_custom_catalog(
    src_name_list,
    src_ra_list,
    src_dec_list,
    src_glon_list,
    src_glat_list,
    catalog_name="custom_catalog.cat",
    catalog_dir=None,
    catnum_init=32767,
):
    """
    This creates a catalog file for a number of sources that the user is interested in. Merges the created catalog with
    a past BAT survey catalog which includes typical bright sources observed by BAT. This allows the sources to be
    appropriately cleaned.

    :param src_name_list: List of the names of the sources that should be added to the catalog
    :param src_ra_list: List of the RA of the sources, in the same order as src_name_list
    :param src_dec_list: List of the Dec of the sources, in the same order as src_name_list
    :param src_glon_list: List of the galactic latitude of the sources, in the same order as src_name_list
    :param src_glat_list: List of the galactic longitude of the sources, in the same order as src_name_list
    :param catalog_name: String of the name of the resulting catalog that is produced
    :param catalog_dir: String (or None) of the directory where the catalog should be saved
    :param catnum_init: Int that denotes the initial catalog number to be assigned to the sources of interest, should
        not overlap with any cat_num values of other BAT survey sourced (this parameter should be ignored except for
        very few scenarios)
    :return: Path object pointing to the new catalog file
    """

    # Add check to make sure that input is not tuple
    if (
        type(src_name_list) is tuple
        or type(src_ra_list) is tuple
        or type(src_dec_list) is tuple
        or type(src_glon_list) is tuple
        or type(src_glat_list) is tuple
    ):
        raise ValueError(
            "The inputs cannot be tuples, either single values or lists are accepted."
        )

    # make the inputs lists if they are not
    if type(src_name_list) is not list:
        src_name_list = [src_name_list]
    if type(src_ra_list) is not list:
        src_ra_list = [src_ra_list]
    if type(src_dec_list) is not list:
        src_dec_list = [src_dec_list]
    if type(src_glon_list) is not list:
        src_glon_list = [src_glon_list]
    if type(src_glat_list) is not list:
        src_glat_list = [src_glat_list]

    # name sure that the source names are ascii strings
    src_name_list = [i.encode("ascii") for i in src_name_list]

    # set default for catalog name and location
    catalog_name = Path(catalog_name)
    if catalog_dir is None:
        catalog_dir = Path.cwd()
    else:
        catalog_dir = Path(catalog_dir)

    prev_name = catalog_name.stem
    cat = catalog_dir.joinpath(
        prev_name + "_prev.cat"
    )
    final_cat = catalog_dir.joinpath(
        catalog_name
    )

    # create the columns of file
    c1 = fits.Column(
        name="CATNUM",
        array=np.array(
            [i for i in range(catnum_init - len(src_name_list), catnum_init)]
        ),
        format="I",
    )  # 2 byte integer
    c2 = fits.Column(name="NAME", array=np.array(src_name_list), format="30A")
    c3 = fits.Column(
        name="RA_OBJ", array=np.array(src_ra_list), format="D", unit="deg", disp="F9.5"
    )
    c4 = fits.Column(
        name="DEC_OBJ",
        array=np.array(src_dec_list),
        format="D",
        unit="deg",
        disp="F9.5",
    )
    c5 = fits.Column(
        name="GLON_OBJ",
        array=np.array(src_glon_list),
        format="D",
        unit="deg",
        disp="F9.5",
    )
    c6 = fits.Column(
        name="GLAT_OBJ",
        array=np.array(src_glat_list),
        format="D",
        unit="deg",
        disp="F9.5",
    )
    c7 = fits.Column(
        name="ALWAYS_CLEAN", array=np.array([0] * len(src_name_list)), format="1L"
    )  # 1 byte logical

    cols = fits.ColDefs([c1, c2, c3, c4, c5, c6, c7])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(str(cat))

    # need to get the file name off to get the dir this file is located in
    dir = Path(__file__[::-1].partition("/")[-1][::-1])
    hsp.ftmerge(
        infile="%s %s"
        % (str(dir.joinpath("data").joinpath("survey6b_2.cat")), str(cat)),
        outfile=str(final_cat),
    )

    os.system("rm %s" % (str(cat)))

    return final_cat

def combine_survey_lc(survey_obsid_list, output_dir=None, clean_dir=True):
    """
    Concatenates a set of *.cat files to produce a fits file containing data over the duration of times specified in the
    BatSurvey objects. This runs for the catalog that was passed to the constructor methods of the BatSurvey objects

    :param survey_obsid_list: List of BatSurvey objects
    :param clean_dir: Boolean set to True by default. Denotes if the whole directory that holds all the compiled light curve
        data for the passed survey observations should be deleted and recreated if the directory exists.
    :return: Returns a string with the directory of the combined light curve files
    """

    if type(survey_obsid_list) is not list:
        survey_obsid_list = [survey_obsid_list]

    # get the main directory where we should create the total_lc directory
    if output_dir is None:
        output_dir = survey_obsid_list[0].result_dir.parent.joinpath(
            "total_lc"
        )  # os.path.join(main_dir, "total_lc")
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    # if the directory doesn't exist, create it otherwise overwrite it
    dirtest(output_dir, clean_dir=clean_dir)

    # make the local pfile dir if it doesn't exist and set this value
    _local_pfile_dir = output_dir.joinpath(".local_pfile")
    _local_pfile_dir.mkdir(parents=True, exist_ok=True)
    try:
        hsp.local_pfiles(pfiles_dir=str(_local_pfile_dir))
    except AttributeError:
        hsp.utils.local_pfiles(par_dir=str(_local_pfile_dir))

    ret = []
    for obs in survey_obsid_list:
        for i in obs.pointing_flux_files:
            dictionary = dict(
                keycolumn="NAME",
                infile=str(i),
                outfile=str(output_dir.joinpath("%s.cat")),
            )

            # there is a bug in the heasoftpy code so try to explicitly call it for now
            ret.append(hsp.batsurvey_catmux(**dictionary))

    return output_dir


def read_lc_data(filename, energy_band_index=None, T0=0):
    """
    Reads in a fits file that contains rate information, at different energy bands, at a number of METs

    :param filename: String of the name of the fits file
    :param energy_band_index: int or None to denote which energy band the user wants to choose
        The bands, in order of the index that they would be accessed are: 14-20 keV, 20-24 keV, 24-35 keV, 35-50 keV,
        50-75 keV, 75-100 keV, 100-150 keV, 150-195 keV
    :param T0: float that represents a critial time that observations should be measured in time with respect to
    :return: arrays of the time, time bin size, rate, rate_error, and the SNR of the measurement in time
    """

    # get fits file data
    time = []
    time_err = []
    rate = []
    rate_err = []
    snr = []

    filename = str(filename)
    lc_fits = fits.open(filename)
    lc_fits_data = lc_fits[1].data

    time_array = lc_fits_data.field("TIME")
    timestop_array = lc_fits_data.field("TIME_STOP")
    #exposure_array = lc_fits_data.field("EXPOSURE") this isn't needed
    rate_array = lc_fits_data.field("RATE")
    rate_err_array = lc_fits_data.field("RATE_ERR")
    bkg_var_array = lc_fits_data.field("BKG_VAR")
    snr_array = lc_fits_data.field("VECTSNR")

    for i in range(len(lc_fits_data)):
        time_start = time_array[i] - T0  # this is in MET
        time_stop = timestop_array[i] - T0
        time_mid = (time_start + time_stop) / 2.0 # we want to leave units as MET
        time_err_num = (time_stop - time_start) / 2.0 # we want to leave units as MET

        time.append(time_mid)
        time_err.append(time_err_num)

        if energy_band_index is not None:
            rate.append(rate_array[i][energy_band_index - 1])
            rate_err.append(rate_err_array[i][energy_band_index - 1])
            snr.append(snr_array[i][energy_band_index - 1])
        else:
            if len(rate_array[i]) > 8:
                rate.append(rate_array[i][-1])
                rate_err.append(rate_err_array[i][-1])
                snr.append(snr_array[i][-1])
            else:
                rate_tot = 0.0
                rate_err_2_tot = 0.0
                bkg_var_2_tot = 0.0
                for j in range(len(rate_array[i])):
                    rate_num = rate_array[i][j]
                    rate_err_2 = rate_err_array[i][j] * rate_err_array[i][j]
                    bkg_var_2 = bkg_var_array[i][j] * bkg_var_array[i][j]
                    rate_tot = rate_tot + rate_num
                    rate_err_2_tot = rate_err_2_tot + rate_err_2
                    bkg_var_2_tot = bkg_var_2_tot + bkg_var_2

                rate.append(rate_tot)
                rate_err_tot = np.sqrt(rate_err_2_tot)
                rate_err.append(rate_err_tot)
                snr_allband_num = rate_tot / np.sqrt(bkg_var_2_tot)
                snr.append(snr_allband_num)

    lc_fits.close()

    return time, time_err, rate, rate_err, snr


def calc_response(phafilename):
    """
        This function generates the response matrix for a given pha file by calling batdrmgen
        (this is a HEASOFT function).

        :param phafilename: String that denotes the location and name of the PHA file that the user would like to
            calculate the response matrix for.
        :return: Heasoftpy "Result" object obtained from calling heasoftpy batdrmgen. The "Result" object is the entire
            output, which helps to debug in case of an error.
    """

    if type(phafilename) is not list:
        phafilename = [phafilename]

    # when passing in tht whole filename, the paths mess up the connection between the response file and the pha file
    # since there seems to be some character limit to this header value. Therefore, we need to cd to the directory
    # that the PHA file lives in and create the .rsp file and then cd back to the original location.

    # make sure that all elements are paths
    phafilename = [Path(i) for i in phafilename]

    # we are passing in a whole filepath or
    # we are already located in the PHA directory and are mabe calculating the upperlimit bkg spectrum
    _local_pfile_dir = (
        phafilename[0].resolve().parents[1].joinpath(".local_pfile")
    )
    _local_pfile_dir.mkdir(parents=True, exist_ok=True)
    try:
        hsp.local_pfiles(pfiles_dir=str(_local_pfile_dir))
    except AttributeError:
        hsp.utils.local_pfiles(par_dir=str(_local_pfile_dir))

    # Check if the phafilename is a string and if it has an extension .pha. If NOT then exit
    for filename in phafilename:
        if ".pha" not in filename.name:
            raise ValueError(
                "The file name %s needs to be a string and must have an extension of .pha ."
                % (str(filename))
            )

        # get the cwd
        current_dir = Path.cwd()

        # get the directory that we have to cd to and the name of the file
        pha_dir = filename.parent
        pha_file = filename.name

        # cd to that dir
        if str(pha_dir) != str(current_dir):
            os.chdir(pha_dir)

        # Split the filename by extension, so as to remove the .pha and replace it with .rsp
        # this is necessary since sources can have '.' in name
        out = (
            filename.stem + ".rsp"
        )

        # create drm
        output = hsp.batdrmgen(
            infile=pha_file, outfile=out, chatter=2, clobber="YES", hkfile="NONE"
        )

        # cd back
        if str(pha_dir) != str(current_dir):
            os.chdir(current_dir)

    return output


def fit_spectrum(
    phafilename,
    surveyobservation,
    plotting=True,
    generic_model=None,
    setPars=None,
    use_cstat=True,
    fit_iterations=1000,
    verbose=True,
):
    """
    Fits a spectrum that is loaded in from a BAT pha file. The header of the PHA file must have the associated
    response information.

    User has to pass a phafilename and the BatSurvey object "surveyobservation" (mandatory).
    The user should have already run the "batsurvey" command and created the surveyobservation object.

    The user can specfiy their own spectral model that is XSPEC compatible.
    To learn about how to specify a spectral model in pyXspec the user can
    look at the following link: https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/index.html

    For e.g, to specify a model one has to do the following:
    model=xsp.Model(generic_model,setPars={1:45,2:"123,-1"}) Here -1 stands for "frozen".

    User has to specify cflux in their model. This is mandatory because we use this same function (and hence the user specified model)
    to test any detection in the BAT energy band.


    If no model is specfied by the user, then by default the specrum is fit with the following Xspec model:
    cflux*(powerlaw): with Cflux E_min=14 keV (Frozen), E_max=195 keV (Frozen), flux=-12 (initial value),
    powerlaw Gamma=2 Free, and norm=frozen. Powerlaw norm kept frozen.



    :param phafilename: String that denotes the location and name of the PHA file.
    :param surveyobservation: Object denoting the batsurvey observation object which contains all the
     necessary information related to this observation.
    :param plotting: Boolean statement, if the user wants to plot the spectrum.
    :param generic_model: String with XSPEC compatible model, which must include cflux.
    :param setPars: Boolean to set the parameter values of the model specified above.
    :param use_cstat: Boolean to use cstat in case of low counts (Poisson statistics), otherwise use chi squared stats.
    :param fit_iterations: Number of fit iterations to be carried out by XSPEC.
     Since BAT data has just 8 energy channels, a default of 100 is enough.
     But the user can specify any value that may be needed.
    :param verbose: Boolean to show every output during the fitting process.
     Set to True by default, that'll help the user to identify any issues with the fits.
    :return: None
    """

    try:
        import xspec as xsp
    except ModuleNotFoundError as err:
        # Error handling
        print(err)
        raise ModuleNotFoundError(
            "The pyXspec package needs to installed to fit spectra with this function."
        )

    # In the next few steps we will get into the directory where the PHA files and rsp files are located
    # Do the fitting and then get out to our current directory: current_dir
    # get the cwd.
    phafilename = Path(phafilename)
    current_dir = Path.cwd()

    # Check if the phafilename is a string and if it has an extension .pha. If NOT then exit
    if ".pha" not in phafilename.name:
        raise ValueError(
            "The file name %s needs to be a string and must have an extension of .pha ."
            % (str(phafilename))
        )

    # get the directory that we have to cd to and the name of the file
    pha_dir = phafilename.parent
    pha_file = phafilename.name

    # The old statement: pointing_id=pha_file.split(".")[0].split("_")[-1] didnt work if source_id has period in it
    pointing_id = phafilename.stem.split("_")[-1]

    if len(pha_file.split("_survey")) > 1:
        # we've got a pha for a normal survey catalog
        source_id = pha_file.split("_survey")[
            0
        ]  # This is the source name compatible with the catalog
    else:
        # we've got a mosaic survey result
        source_id = pha_file.split("_mosaic")[0]

    # cd to that dir
    if str(pha_dir) != str(current_dir):
        os.chdir(pha_dir)

    xsp.AllData -= "*"
    s = xsp.Spectrum(
        pha_file
    )

    # Define model
    if (
        generic_model is not None
    ):  # User provides a string of model, and a Dictionary for the initial values
        if type(generic_model) is str:
            if "cflux" in generic_model:
                # The user must provide the cflux, or else we will not be able to predict of there is a statistical
                # detection (in the next function).
                try:
                    model = xsp.Model(
                        generic_model, setPars=setPars
                    )  # Set the initial value for the fitting using the Model object attribute

                except Exception as e:
                    print(e)
                    raise ValueError("The model needs to be specified correctly")

            else:
                raise ValueError(
                    "The model needs cflux in order to calulate error on the flux in 14-195 keV"
                )

    else:
        # If User does not pass any model

        model = xsp.Model("cflux*po")
        p1 = model(1)  # cflux      Emin = 14 keV
        p2 = model(2)  # cflux      Emax = 195 keV
        p3 = model(3)  # cflux      lg10Flux
        p4 = model(4)  # Photon index Gamma
        p5 = model(5)  # Powerlaw norm

        # Setting the vlaues and freezing them.

        p1.values = 14  # already frozen
        p2.values = 195  # already frozen
        p4.values = 2
        p4.frozen = False
        p5.values = 0.001
        p5.frozen = True

        # model_components=model.componentNames  #This is a list of the model components
    # Check if the model is XSPEC compatible : Done Listing down the model parameters in a dictionary: parm1: Value,
    # param2: Value.... If no initial values given , default XSPEC values to be used. We will manipulate these param
    # values to "set a value" or "freeze/thaw" a value, set a range for these viable values. We can call the best fit
    # param values, after fit.

    # Fitting the data with this model

    if use_cstat:
        xsp.Fit.statMethod = "cstat"
    else:
        xsp.Fit.statMethod = "chi"

    # Stop fit at nIterations and do not query.
    xsp.Fit.query = "no"

    xsp.Fit.nIterations = fit_iterations
    xsp.Fit.renorm()

    # try to do the fitting if it doesn't work fill in np.nan values for things
    try:
        xsp.Fit.perform()
        if verbose:
            xsp.AllModels.show()
            xsp.Fit.show()

        # Get coordinates from XSPEC plot to use in matplotlib:
        xsp.Plot.device = "/null"
        xsp.Plot("data")
        chans = xsp.Plot.x()
        rates = xsp.Plot.y()
        xerr = xsp.Plot.xErr()
        yerr = xsp.Plot.yErr()
        folded = xsp.Plot.model()

        # Plot using Matplotlib:
        f, ax = plt.subplots()
        ax.errorbar(x=chans, xerr=xerr, y=rates, yerr=yerr, fmt="ro")
        ax.plot(chans, folded, "k-")
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("counts/cm^2/sec/keV")
        ax.set_xscale("log")
        ax.set_yscale("log")
        f.savefig(
            phafilename.parent.joinpath(phafilename.stem + ".pdf")
        )
        if plotting:
            plt.show()

        # Capturing the Flux and its error. saved to the model object, can be obtained by calling model(1).error,
        # model(2).error
        model_params = dict()
        for i in range(1, model.nParameters + 1):
            xsp.Fit.error("2.706 %d" % (i))

            # get the name of the parameter
            par_name = model(i).name
            model_params[par_name] = dict(
                val=model(i).values[0],
                lolim=model(i).error[0],
                hilim=model(i).error[1],
                errflag=model(i).error[-1],
            )
        surveyobservation.set_pointing_info(
            pointing_id, "model_params", model_params, source_id=source_id
        )

    except Exception as Error_with_Xspec_fitting:
        # this is probably that XSPEC cannot fit because of negative counts
        if verbose:
            print(Error_with_Xspec_fitting)

        # need to fill in nan values for all the model params and 'TTTTTTTTT' for the error flag
        model_params = dict()
        for i in range(1, model.nParameters + 1):
            # get the name of the parameter
            par_name = model(i).name
            model_params[par_name] = dict(
                val=np.nan, lolim=np.nan, hilim=np.nan, errflag="TTTTTTTTT"
            )
        surveyobservation.set_pointing_info(
            pointing_id, "model_params", model_params, source_id=source_id
        )

    # Incorporating the model names, parameters, errors into the BatSurvey object.
    xsp.Xset.save(phafilename.stem + ".xcm")
    xspec_savefile = phafilename.parent.joinpath(
        phafilename.stem + ".xcm"
    )
    surveyobservation.set_pointing_info(
        pointing_id, "xspec_model", xspec_savefile, source_id=source_id
    )

    # cd back
    if str(pha_dir) != str(current_dir):
        os.chdir(current_dir)

    return None


def calculate_detection(
    surveyobservation,
    source_id,
    pl_index=2,
    nsigma=3,
    bkg_nsigma=5,
    plot_fit=False,
    verbose=True,
):
    """
    This function uses the fitting function and statistically checks if there is any significant detection (at a specfied confidence).
    If there is no detection, then the function re-calculates the PHA with a bkg_nsigma times the background to calculate the
    upper limit on the flux, at a certain confidence level (given by the user specified bkg_nsigma).

    We deal with two cases:

     (1) Non-detection:  Checking if nsigma error on  The 14-195 keV flux is consistent with the equation (measured flux - nsigma*error)<=0,
     then return: upper limit=True
     and then recalculate the PHA +response again.... with count rate= bkg_nsigma*BKG_VAR

     (2) Detection: If (measured flux - nsigma*error)>=0 then return: "detection has been measured"

     This operates on the entire batsurvey object (corresponding to a batobservation id),
     and we want to see if there is a detection for 'any number of pointings for a given source' in that batobservation id.

     Note that it operates ONLY on one source.
     For different sources one can specify separate detection threshold ('sigma') for different sources.
     Thus we have kept this function to operate only ONE source at a time.

    :param surveyobservation: Object denoting the batsurvey observation object which contains all the necessary
        information related to this observation.
    :param source_id: String denoting the source name exactly as that in the phafilename.
    :param pl_index: Float (default 2) denoting the power law photon index that will be used to obtain a flux upper
        limit
    :param nsigma: Integer, denoting the number fo sigma the user needs to justify a detection
    :param bkg_nsigma: Integer, denoting the number of sigma the user needs to calculate flux upper limit in case
        of a non detection.
    :param plot_fit: Boolean to determine if the fit should be plotted or not
    :param verbose: Boolean to show every output during the fitting process. Set to True by default, that'll help the
        user to identify any issues with the fits.
    :return: In case of a non-detection a flux upper limit is returned.
    """

    try:
        import xspec as xsp
    except ModuleNotFoundError as err:
        # Error handling
        print(err)
        raise ModuleNotFoundError(
            "The pyXspec package needs to installed to determine if a source has been detected with this function."
        )

    current_dir = Path.cwd()

    # get the directory that we have to cd to and the name of the file
    pha_dir = surveyobservation.get_pha_filenames(id_list=[source_id])[0].parent

    pointing_ids = (
        surveyobservation.get_pointing_ids()
    )  # This is a list of pointing_ids in this bat survey observation

    # cd to that dir
    if str(pha_dir) != str(current_dir):
        os.chdir(pha_dir)

    flux_upperlim = []

    # By specifying the source_id, we now have the specific PHA filename list corresponding to the
    # pointing_id_list for this given bat survey observation.
    phafilename_list = surveyobservation.get_pha_filenames(
        id_list=[source_id], pointing_id_list=pointing_ids
    )

    for i in range(len(phafilename_list)):  # Loop over all phafilename_list,
        pha_dir = phafilename_list[i].parent
        pha_file = phafilename_list[i].name

        # The old statement: pointing_id=pha_file.split(".")[0].split("_")[-1] didnt work if source_id has period in it
        pointing_id = phafilename_list[i].stem.split("_")[-1]

        # Within the pointing dictionar we have the "key" called "Xspec_model" which has the parameters, values and
        # errors.
        error_issues = False  # preset this here
        try:
            pointing_dict = surveyobservation.get_pointing_info(
                pointing_id, source_id=source_id
            )
            model = pointing_dict["model_params"]["lg10Flux"]
            flux = model["val"]  # ".cflux.lg10Flux.values[0]              #Value
            fluxerr_lolim = model["lolim"]  # .cflux.lg10Flux.error[0]      #Error
            fluxerr_uplim = model["hilim"]  # .cflux.lg10Flux.error[1]

            avg_flux_err = 0.5 * (
                ((10**fluxerr_uplim) - (10**flux))
                + ((10**flux) - (10**fluxerr_lolim))
            )
            print(
                "The condition here is",
                10 ** (flux),
                [10**fluxerr_lolim, 10**fluxerr_uplim],
                nsigma,
                avg_flux_err,
                ((10**flux) - nsigma * avg_flux_err),
            )

            # check the errors for any issues:
            if "T" in model["errflag"]:
                error_issues = True

        except ValueError:
            # the fitting was not successful and the dictionary was not created but want to enter the upper limit if
            # statement
            fluxerr_lolim = 0
            flux = 1
            nsigma = 1
            avg_flux_err = 1

        if (
            fluxerr_lolim == 0
            or (((10**flux) - nsigma * avg_flux_err) <= 0)
            or np.isnan(flux)
            or error_issues
        ):
            print("No detection, just upperlimits for the spectrum:", pha_file)
            # Here redo the PHA calculation with 5*BKG_VAR
            surveyobservation.calculate_pha(
                calc_upper_lim=True,
                bkg_nsigma=bkg_nsigma,
                id_list=source_id,
                single_pointing=pointing_id,
            )

            # can also do surveyobservation.get_pha_filenames(id_list=source_id,pointing_id_list=pointing_id,
            # getupperlim=True) to get the created upperlimit file. Will do this because it is more robust
            # bkgnsigma_upper_limit_pha_file= pha_file.split(".")[0]+'_bkgnsigma_%d'%(bkg_nsigma) + '_upperlim.pha'
            bkgnsigma_upper_limit_pha_file = surveyobservation.get_pha_filenames(
                id_list=source_id, pointing_id_list=pointing_id, getupperlim=True
            )[0].name

            try:
                calc_response(bkgnsigma_upper_limit_pha_file)
            except:
                # This is a MosaicBatSurvey object which already has the default associated response file
                pass

            xsp.AllData -= "*"

            s = xsp.Spectrum(bkgnsigma_upper_limit_pha_file)

            xsp.Fit.statMethod = "cstat"

            model = xsp.Model("po")
            # p1 = m1(1)  # cflux      Emin = 15 keV
            # p2 = m1(2)  # cflux      Emax = 150 keV
            # p3 = m1(3)  # cflux      lg10Flux
            p4 = model(1)  # Photon index Gamma
            p5 = model(2)  # Powerlaw norm

            # p1.values = 15  # already frozen
            # p2.values = 150  # already frozen
            p4.frozen = True
            p4.values = pl_index
            p5.values = 0.001
            p5.frozen = False

            if verbose:
                print("******************************************************")
                print(
                    f"Fitting the {bkg_nsigma} times bkg of the spectrum {bkgnsigma_upper_limit_pha_file}"
                )

            xsp.Fit.nIterations = 100
            xsp.Fit.perform()
            if plot_fit:
                xsp.AllModels.show()
                xsp.Fit.show()
            xsp.AllModels.calcFlux("14.0 195.0")

            if verbose:
                print("******************************************************")
                print("******************************************************")
                print("******************************************************")

                print(s.flux)

            # Capturing the simple model. saved to the model object, can be obtained by calling model(1).error,
            # model(2).error
            model_params = dict()
            for j in range(1, model.nParameters + 1):
                # get the name of the parameter
                par_name = model(j).name
                model_params[par_name] = dict(
                    val=model(j).values[0],
                    lolim=model(j).error[0],
                    hilim=model(j).error[1],
                    errflag="TTTTTTTTT",
                )
            surveyobservation.set_pointing_info(
                pointing_id, "model_params", model_params, source_id=source_id
            )

            surveyobservation.set_pointing_info(
                pointing_id,
                "nsigma_lg10flux_upperlim",
                np.log10(s.flux[0]),
                source_id=source_id,
            )
        else:  # Detection
            if verbose:
                print("A detection has been measured at the %d sigma level" % (nsigma))

    # cd back
    if str(pha_dir) != str(current_dir):
        os.chdir(current_dir)

    return flux_upperlim  # This is a list for all the Valid non-detection pointings


def print_parameters(
    obs_list,
    source_id,
    values=["met_time", "utc_time", "exposure"],
    energy_range=[14, 195],
    latex_table=False,
    savetable=False,
    save_file="output.txt",
    overwrite=True,
    add_obs_id=True,
):
    """
    Convenience function to plot various survey data pieces of information in a formatted file/table

    :param obs_list: A list of BatSurvey objects
    :param source_id: A string with the name of the source of interest.
    :param values: A list of strings contaning information that the user would like to be printed out. The strings
        correspond to the keys in the pointing_info dictionaries of each BatSurvey object and the colmns will be put
        in this order.
    :param energy_range: a list or array of the minimum energy range that should be considered and the maximum energy
        range that should be considered. By default, this is 14-195 keV
    :param latex_table: Boolean to denote if the output should be formatted as a latex table
    :param savetable: Boolean to denote if the user wants to save the table to a file
    :param save_file: string that specified the location and name of the file that contains the saved table
    :param overwrite: Boolean that says to overwrite the output file if it already exists
    :param add_obs_id: Boolean to denote if the observation and pointing IDs should be added to the value list automatically
    :return: None
    """

    save_file = Path(save_file)

    if save_file.exists() and overwrite:
        save_file.unlink()

    if type(obs_list) is not list:
        obs_list = [obs_list]

    if add_obs_id:
        # make sure that the values list has obs_id and pointing_id in it
        if "pointing_id" not in values:
            values.insert(0, "pointing_id")

        if "obs_id" not in values:
            values.insert(0, "obs_id")

    # get all the data that we need
    all_data = concatenate_data(obs_list, source_id, values, energy_range=energy_range)[
        source_id
    ]

    if savetable and save_file is not None:
        # open the file to write the output to
        f = open(str(save_file), "w")

    outstr = " "  # Obs ID  \t Pointing ID\t"
    for i in values:
        outstr += f"{i: ^31}\t"  # "\t%s"%(i)

    if not savetable:
        print(outstr)
    else:
        f.writelines([str(outstr), "\n"])

    if latex_table:
        nchar = 29
    else:
        nchar = 30

    outstr = ""
    for i in range(len(all_data[list(all_data.keys())[0]])):
        for key in values:
            val = all_data[key]
            if "id" in key:
                # if we have just one observation ID then we still want to print the obs_id for the first entry in list
                # if we dont then we need to make sure that the printed value is not the same as the one prior
                if i == 0 or val[i] != val[i - 1]:
                    print_val = val[i]
                else:
                    print_val = ""
            else:
                print_val = val[i]

            # see if there are errrors associated with the key
            if key + "_lolim" in all_data.keys():
                # get the errors
                lolim = all_data[key + "_lolim"][i]
                hilim = all_data[key + "_hilim"][i]

                if not np.isnan(lolim) and not np.isnan(hilim):
                    middle_str = ""
                    if len(str(val[i]).split("e")) > 1:
                        base = int(str(val[i]).split("e")[-1])

                        if latex_table:
                            middle_str += "$"

                        middle_str += f"{val[i] / 10 ** base:-.3}^{{{hilim / 10 ** base:+.2}}}_{{{-1 * lolim / 10 ** base:+.2}}}"

                        if latex_table:
                            middle_str += f" \\times "
                        else:
                            middle_str += f" x "

                        middle_str += f"10^{{{base:+}}}"

                        if latex_table:
                            middle_str += "$"

                        print_val = middle_str

                    else:
                        print_val = ""
                        if latex_table:
                            print_val += "$"

                        print_val += (
                            f"{val[i]:-.3}" + f"^{{{hilim :+.2}}}_{{{-1 * lolim :+.2}}}"
                        )

                        if latex_table:
                            print_val += "$"

                    outstr += f"{print_val: ^{nchar}}" + "\t"
                else:
                    middle_str = ""
                    if len(str(val[i]).split("e")) > 1:
                        base = int(str(val[i]).split("e")[-1])

                        if latex_table:
                            middle_str += "$"

                        middle_str += f"{val[i] / 10 ** base:-.3}"

                        if latex_table:
                            middle_str += f" \\times "
                        else:
                            middle_str += f" x "

                        middle_str += f"10^{{{base:+}}}"

                        if latex_table:
                            middle_str += "$"

                        print_val = middle_str

                    outstr += f"{print_val: ^{nchar}}\t"
            else:
                outstr += f"{print_val: ^{nchar}}\t"

            if latex_table:
                outstr += " & "

        if latex_table:
            outstr = outstr[:-2]
            outstr += " \\\\"
        outstr += "\n"

    if savetable and save_file is not None:
        f.writelines([str(outstr), "\n"])
        f.close()
    else:
        print(outstr)

    if savetable and save_file is not None:
        f.close()


def download_swiftdata(
    observations,
    reload=False,
    fetch=True,
    jobs=10,
    bat=True,
    auxil=True,
    log=False,
    uvot=False,
    xrt=False,
    tdrss=True,
    save_dir=None,
    **kwargs,
) -> dict:
    """
    Download Swift data from HEASARC or quicklook sites to a local mirror directory.

    If the data already exists in the mirror, it is not reloaded unless it is from
    a quicklook site, or if reload is set.

    Data for observations can be selected by instrument or by filename match.

    Observations are specified as a list of OBSIDs, or a table with an 'OBSID' field.

    Match is a string or list of strings that match the filenames using unix globbing rules.
    e.g. `match=['*brtms*', '*sao.*']` will match both the BAT 64 ms rates and the
    instrument auxiliary orbit information file (if bat=True and auxil=True are set) for
    each observation.

    The result is returned in a dict indexed by OBSID.  The 'data' element of an OBSID's
    dict entry is a `swifttools.swift_too.Swift_Data` table including attributes for
    the  .url and .localpath of each file.


    :param observations: OBSIDs to download
    :param reload: load even if the data is already in the save_dir
    :param fetch: Download the data if it is not locally cached (defaults to True)
    :param jobs: number of simultaneous download jobs.  (Set to 1 to execute unthreaded.)
    :param bat: load the bat data
    :param auxil: load the auxil data
    :param log: load the log data   (mostly diagnostic, defaults to false)
    :param uvot: load the uvot data (high volume, defaults to false)
    :param xrt: load the xrt data (high volume, defaults to false)
    :param tdrss: load the tdrss data (necessary for triggered BAT event data, defaults to True)
    :param save_dir: The output directory where the observation ID directories will be saved
    (From swifttools.swift_too.Data )
    :param match: pattern (or list) to match (defaults to all)
    :param kwargs: passed to swifttools.swift_too.Data
    :return: dict{obsid: {obsoutdir:..., success:..., loaded:..., [, datafiles:swtoo.Data][, ]}
    """

    # for GRBs do eg. object_name='GRB220715B', mission="swiftmastr"
    # table = heasarc.query_object(object_name, mission=mission, sortvar="START_TIME")
    # The first entry in the table should be the TTE data observation ID, from when the GRB was triggered, want to
    # download the 0 segment. (Can download others for different survey analyses etc)
    # Can also query mission="swifttdrss" and get the GRB target ID and just download the obs_id=str(Target ID)+'000'

    if save_dir is None:
        save_dir = datadir()
    save_dir = Path(save_dir).resolve()
    if np.isscalar(observations) or isinstance(observations, ap.table.row.Row):
        observations = [observations]
    obsids = []
    for entry in observations:
        try:  # swiftmastr observation table
            entry = entry["OBSID"]
        except:
            pass
        try:  # swifttools.ObsQuery
            entry = entry.obsid  # f"{entry.targetid:08d}{entry.seg:03d}"
        except:
            pass
        if isinstance(entry, int):
            entry = f"{entry:011d}"
        if not isinstance(entry, str):
            raise RuntimeError(f"Can't convert {entry} to OBSID string")
        obsids.append(entry)
    # Remove duplicate obsids, but otherwise keep in order.
    obsids = list({o: None for o in obsids}.keys())
    nowts = datetime.datetime.now().timestamp()
    kwargs["fetch"] = fetch
    download_partialfunc = functools.partial(
        _download_single_observation,
        reload=reload,
        bat=bat,
        auxil=auxil,
        log=log,
        uvot=uvot,
        xrt=xrt,
        tdrss=tdrss,
        save_dir=save_dir,
        nowts=nowts,
        **kwargs,
    )
    if jobs == 1:
        results = {}
        for obsid in obsids:
            result = download_partialfunc(obsid)
            results[obsid] = result
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            results = {
                result["obsid"]: result
                for result in executor.map(download_partialfunc, obsids)
            }
    return results


def _download_single_observation(
    obsid, *, reload, bat, auxil, log, uvot, xrt, tdrss, save_dir, nowts, **kwargs
):
    """Helper function--not for general use

    Downloads files for a single OBSID, given parameters from download_swiftdata()
    after encapsulation as a partial function for threading.

    Args:
        obsid (str): Observation ID to download
        (remaining arguments are as in download_swiftdata())


    Raises:
        RuntimeError: If missing local directory.  Other exceptions are presented as warnings and
        by setting the 'success' flag to False.

    Returns:
        _type_: _description_
    """
    obsoutdir = save_dir.joinpath(obsid)
    quicklookfile = obsoutdir.joinpath(".quicklook")
    result = dict(obsid=obsid, success=True, obsoutdir=obsoutdir, quicklook=False)
    try:
        clobber = reload or quicklookfile.exists()
        data = swtoo.Swift_Data(
            obsid=obsid,
            clobber=clobber,
            bat=bat,
            log=log,
            auxil=auxil,
            uvot=uvot,
            xrt=xrt,
            tdrss=tdrss,
            outdir=str(save_dir),
            **kwargs,
        )
        result["data"] = data
        if data.status.status != "Accepted":
            raise RuntimeError(" ".join(data.status.warnings + data.status.errors))
        if data.quicklook:  # Mark the directory as quicklook
            quicklookfile.open("w").close()
            result["quicklook"] = True
        elif quicklookfile.exists():
            # This directory just transitioned from quicklook to archival version
            oldqlookdir = save_dir.joinpath("old_quicklook", obsid)
            oldqlookdir.mkdir(exist_ok=True, parents=True)
            for stalefile in obsoutdir.glob("**/*"):
                # Any file older than the time before the data was downloaded
                if (
                    stalefile.is_file()
                    and stalefile.stat().st_mtime < nowts
                    and not stalefile.name.startswith(".")
                ):
                    stalefile.replace(oldqlookdir.joinpath(stalefile.name))
            quicklookfile.unlink()
            result.update(
                datafiles=data,
                quicklook=data.quicklook,
                outdir=Path(data.outdir),
                success=True,
                downloaded=True,
            )
        if not Path(data.outdir).is_dir():
            raise RuntimeError(f"Data directory {data.outdir} missing")
    except Exception as e:
        warnings.warn(f"Did not download {obsid} {e}")
        result["success"] = False
    return result


def test_remote_URL(url):
    return requests.head(url).status_code < 400


def from_heasarc(object_name=None, tablename="swiftmastr", **kwargs):
    heasarc = Heasarc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ap.utils.exceptions.AstropyWarning)
        table = heasarc.query_object(
            object_name=object_name, mission=tablename, **kwargs
        )
    return table


def find_trigger_data():
    raise NotImplementedError


def met2mjd(met_time):
    """
    A convenience function that calculates the MJD time from a Swift MET time. Ths function either uses the swiftbat
    code base which is quicker or the heasoftpy swifttime function which is slower.

    :param met_time: a number that is the Swift MET time that will be converted
    :return: a MJD date that includes the Swift time clock correction
    """

    try:
        val = sbu.met2mjd(met_time, correct=True)
    except (ModuleNotFoundError, RuntimeError):
        # calculate times in UTC and MJD units as well
        inputs = dict(
            intime=str(met_time),
            insystem="MET",
            informat="s",
            outsystem="UTC",
            outformat="m",
        )  # output in MJD
        o = hsp.swifttime(**inputs)
        val = float(o.params["outtime"])

    atime = Time(val, format="mjd", scale="utc")
    return atime.value


def met2utc(met_time, mjd_time=None):
    """
    A convenience function that calculates the UTC time from a Swift MET time. Ths function first converts the time to
    MJD, which either uses the swiftbat code base which is quicker or the heasoftpy swifttime function which is slower,
    and then converts it to UTC. The user can also supply a MJD time to save on computational time.

    :param met_time: a number that is the Swift MET time that will be converted
    :param mjd_time: default to None, which means that the code will first calculate the MJD time and then convert it to
        UTC time. If the user already has the MJD time, they can specify it here and the function will directly
        convert it.
    :return: a numpy datetime64 object of the MET time with the Swift clock correction applied
    """
    if mjd_time is None:
        mjd_time = met2mjd(met_time)

    atime = Time(mjd_time, format="mjd", scale="utc")
    return atime.datetime64


def save_progress(obs_list):
    """
    Convience function to save progress for a list of BatSurvey observations

    :param obs_list: list of BatSurvey or MosaicBatSurvey objects
    :return: None
    """
    if type(obs_list) is not list:
        obs_list = [obs_list]

    for i in obs_list:
        i.save()


def set_pdir(pdir):
    """
    Sets the custom pfile directory for calling a heasoftpy function. This ensures that functions can be called in
    parallel. This is depreciated since heasoftpy v1.2.

    :param pdir: None, Path, or string to the custom pfiles directory. a value of None will force heasoftpy to create a
        custom pfiles directory in /tmp, as is specified in their documentation.
    :return:
    """

    # if it's not None, make sure that it's a string that can be passed to heasoftpy. None will
    if pdir is not None:
        pdir = str(pdir)

    try:
        hsp.local_pfiles(pfiles_dir=pdir)
    except AttributeError:
        hsp.utils.local_pfiles(par_dir=pdir)


def reset_pdir():
    """
    Resets the pfiles environment variable to what it originally was. This is depreciated since heasoftpy v1.2.

    :return:
    """
    os.environ["PFILES"] = _orig_pdir


def concatenate_data(
    bat_observation, source_ids, keys, energy_range=[14, 195], chronological_order=True
):
    """
    This convenience function collects the data that was requested by the user as passed into the keys variable. The
    data is returned in the form of a dictionary with the same keys and numpy arrays of all the concatenated data. if
    the user asks for parameters with errors associated with them these errors will be automatically included. For
    example if the user wants rates information then the function will automatically include a dicitonary key to
    hold the rates error information as well

    :param bat_observation: a list of BatObservation objects including BatSurvey and MosaicBatSurvey objects that the
        user wants to extract the relevant data from.
    :param source_ids: The sources that the user would like to collect data for
    :param keys: a string or list of strings
    :param energy_range: a list or array of the minimum energy range that should be considered and the maximum energy
        range that should be considered
    :param chronological_order: Boolean to denote if the outputs should be sorted chronologically or kept in the same
        order as the BATSurvey objects that were passed in
    :return: dict with the keys specified by the user and numpy lists as the concatenated values for each key
    """

    # make sure that the keys are a list
    if type(keys) is not list:
        # it is a single string:
        keys = [keys]

    # see if the user has the rates included in here
    if "rate" in keys:
        # see if the rates_err is already included. If not add it.
        if "rate_err" not in keys:
            keys.append("rate_err")

    if type(source_ids) is not list:
        # it is a single string:
        source_ids = [source_ids]

    # create a dict from the keys for soure and what the user is interested in
    concat_data = dict().fromkeys(source_ids)
    for i in concat_data.keys():
        concat_data[i] = dict().fromkeys(keys)
        for j in concat_data[i].keys():
            concat_data[i][j] = []

    # deterine the energy range that may be of interest. This can be none for total E range or one of the basic 8
    # channel energies or a range that spans more than one energy range of the 8 channels.
    if np.isclose([14, 195], energy_range).sum() == 2:
        e_range_idx = [-1]  # this is just the last index of the arrays for counts, etc
    else:
        # get the index
        obs_min_erange_idx = bat_observation[0].emin.index(np.min(energy_range))
        obs_max_erange_idx = bat_observation[0].emax.index(np.max(energy_range))
        e_range_idx = np.arange(obs_min_erange_idx, obs_max_erange_idx + 1)

    if chronological_order:
        # sort the obs ids by time of 1st pointing id
        all_met = [
            i.pointing_info[i.pointing_ids[0]]["met_time"] for i in bat_observation
        ]
        sorted_obs_idx = np.argsort(all_met)
    else:
        sorted_obs_idx = np.arange(len(bat_observation))

    # iterate over observation IDs
    for idx in sorted_obs_idx:
        obs = bat_observation[idx]
        try:
            # have obs id for normal survey object
            observation_id = obs.obs_id
        except AttributeError:
            # dont have obs_id for mosaic survey object
            observation_id = "mosaic"

        if chronological_order:
            # sort the pointing IDs too
            sorted_pointing_ids = np.sort(obs.pointing_ids)
        else:
            sorted_pointing_ids = obs.pointing_ids

        # iterate over pointings
        for pointings in sorted_pointing_ids:
            # iterate over sources
            for source in concat_data.keys():
                # see if the source exists in the observation
                if source in obs.get_pointing_info(pointings).keys():
                    # iterate over the keys of interest
                    for user_key in keys:
                        save_val = np.nan

                        # see if the user wants observation ID or pointing ID
                        if "obs" in user_key:
                            save_val = observation_id
                            concat_data[source][user_key].append(save_val)
                            save_val = (
                                np.inf
                            )  # set to a crazy number so we don't get errors with np.isnan for a string

                        if "pointing" in user_key:
                            save_val = pointings
                            concat_data[source][user_key].append(save_val)
                            save_val = (
                                np.inf
                            )  # set to a crazy number so we don't get errors with np.isnan for a string

                        # search in all
                        continue_search = True
                        for dictionary in [
                            obs.get_pointing_info(pointings),
                            obs.get_pointing_info(pointings, source_id=source),
                        ]:
                            if (
                                continue_search
                                and np.isnan(save_val)
                                and len(
                                    dpath.search(
                                        obs.get_pointing_info(
                                            pointings, source_id=source
                                        )["model_params"],
                                        user_key,
                                    )
                                )
                                == 0
                                and ("flux" not in user_key.lower())
                                and ("index" not in user_key.lower())
                            ):
                                try:
                                    # if this is a rate/rate_err/snr need to calcualate these quantities based on the
                                    # returned array
                                    if "rate" in user_key or "snr" in user_key:
                                        rate, rate_err, snr = obs.get_count_rate(
                                            e_range_idx, pointings, source
                                        )
                                        if "rate_err" in user_key:
                                            save_val = rate_err
                                        elif "rate" in user_key:
                                            save_val = rate
                                        elif "snr" in user_key:
                                            save_val = snr

                                    else:
                                        save_val = dpath.get(dictionary, user_key)

                                except KeyError:
                                    # if the key doest exist don't do anything but add np.nan
                                    save_val = np.nan
                                    # this key for rate, rate_err, SNR doesn't exist probably because the source wasn't
                                    # detected so don't enter the outer if statement again which will keep saving
                                    # np.nan
                                    if "rate" in user_key or "snr" in user_key:
                                        continue_search = False

                                # save the value to the appropriate list under the appropriate key
                                concat_data[source][user_key].append(save_val)

                        # see if the values are for the model fit
                        if (
                            continue_search
                            and np.sum(np.isnan(save_val)) > 0
                            and "model_params"
                            in obs.get_pointing_info(pointings, source_id=source).keys()
                        ):
                            # can have obs.get_pointing_info(pointings, source_id=source)["model_params"]
                            # but it can be None if the source isn't detected
                            # if obs.get_pointing_info(pointings, source_id=source)["model_params"] is not None:
                            # have to modify the name of the flux related quantity here
                            if "flux" in user_key.lower():
                                real_user_key = "lg10Flux"
                            else:
                                real_user_key = user_key

                            # try to access the dictionary key
                            try:
                                save_val = dpath.get(
                                    obs.get_pointing_info(pointings, source_id=source)[
                                        "model_params"
                                    ],
                                    real_user_key,
                                )
                            except KeyError:
                                # if the key doest exist don't do anything but add np.nan
                                save_val = np.nan
                                # if the value that we want is flux but we only have an upper limit then we have to get
                                # the nsigma_lg10flux_upperlim value
                                if real_user_key == "lg10Flux":
                                    real_user_key = "nsigma_lg10flux_upperlim"
                                    # see if there is a nsigma_lg10flux_upperlim
                                    try:
                                        save_val = dpath.get(
                                            obs.get_pointing_info(
                                                pointings, source_id=source
                                            ),
                                            real_user_key,
                                        )
                                    except KeyError:
                                        # if the key doest exist don't do anything but add np.nan
                                        save_val = np.nan

                            # need to calculate the error on the value
                            # first do the case of flux upper limit
                            if real_user_key == "nsigma_lg10flux_upperlim":
                                save_value = 10**save_val
                                # there is no upper/lower error since we have an upper limit
                                error = np.ones(2) * np.nan
                                is_upper_lim = True
                            else:
                                is_upper_lim = False
                                if real_user_key == "lg10Flux":
                                    save_value = 10 ** save_val["val"]
                                    error = np.array(
                                        [
                                            10 ** save_val["lolim"],
                                            10 ** save_val["hilim"],
                                        ]
                                    )
                                    error = np.abs(save_value - error)
                                else:
                                    try:
                                        save_value = save_val["val"]
                                        error = np.array(
                                            [save_val["lolim"], save_val["hilim"]]
                                        )

                                        if "T" in save_val["errflag"]:
                                            error = np.ones(2) * np.nan
                                        else:
                                            error = np.abs(save_value - error)

                                    except TypeError:
                                        # this is the last resort for catching any keys that aren't found in the dict
                                        # so we may have save_val be = np.nan and we will get TypeError trying to
                                        # call it as a dict
                                        save_value = np.nan
                                        error = np.ones(2) * np.nan

                            # save the value to the appropriate list under the appropriate key
                            concat_data[source][user_key].append(save_value)

                            # save the errors as well. We may need to create the dictionary key for the error/upperlimit
                            user_key_lolim = user_key + "_lolim"
                            user_key_hilim = user_key + "_hilim"
                            user_key_upperlim = user_key + "_upperlim"
                            try:
                                concat_data[source][user_key_lolim].append(error[0])
                                concat_data[source][user_key_hilim].append(error[1])
                                concat_data[source][user_key_upperlim].append(
                                    is_upper_lim
                                )
                            except KeyError:
                                concat_data[source][user_key_lolim] = []
                                concat_data[source][user_key_hilim] = []
                                concat_data[source][user_key_upperlim] = []

                                concat_data[source][user_key_lolim].append(error[0])
                                concat_data[source][user_key_hilim].append(error[1])
                                concat_data[source][user_key_upperlim].append(
                                    is_upper_lim
                                )

    # turn things into numpy array for easier handling
    for src_key in concat_data.keys():
        for key, val in concat_data[src_key].items():
            concat_data[src_key][key] = np.array(val)

    return concat_data


def make_fake_tdrss_message(
    obs_id, trig_time, trig_stop, ra_obj, dec_obj, obs_dir=None
):
    """
    This function creates a fake TDRSS message file that specifies a few important pieces of information which can be
    used in the BAT TTE data processing pipeline.

    :param obs_id:
    :param trig_time:
    :param trig_stop:
    :param ra_obj:
    :param dec_obj:
    :param obs_dir:
    :return: The path object to the location of the created tdrss message file
    """

    from .batobservation import BatObservation

    # see if the observation directory exists
    obs = BatObservation(obs_id, obs_dir=obs_dir)

    # see if the tdrss directory exists. If not, then create it
    # create the tdrss message filename
    tdrss_dir = obs.obs_dir.joinpath("tdrss")
    tdrss_dir.mkdir(parents=True, exist_ok=True)
    tdrss_file = tdrss_dir.joinpath(f"sw{obs.obs_dir.stem}msbce_test.fits.gz")

    # get the trigger id from the observation id
    trig_id = obs.obs_dir.stem[1:-3]

    hdr = fits.Header()
    hdr["CREATOR"] = ("BatAnalysis", " Program that created FITS file")
    hdr["OBS_ID"] = (obs_id, "Observation ID")
    hdr["TARG_ID"] = (trig_id, "Target ID")
    hdr["TRIGGER"] = (trig_id, "Trigger Number")
    hdr["TRIGTIME"] = (trig_time, "[s] MET TRIGger Time")
    hdr["DATETRIG"] = (f"{met2utc(trig_time):.23}", "Corrected UTC date of the trigger")
    hdr["BRA_OBJ"] = (ra_obj, "[deg] BAT RA location of GRB or Object")
    hdr["BDEC_OBJ"] = (dec_obj, "[deg] BAT DEC location of GRB or Object")
    hdr["TRIGSTOP"] = (trig_stop, "[s] Trigger MET STOP time")
    hdr["BACKSTRT"] = (0.0, "[s] BACKground STaRT time")
    hdr["BACKSTOP"] = (0.0, "[s] BACKground STOP time")
    hdr["IMAGETRG"] = ("T", "Image Trigger occured?")

    tdrss_header = fits.PrimaryHDU(header=hdr)

    tdrss_header.writeto(tdrss_file)

    return tdrss_file
