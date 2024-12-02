"""
This file holds convience functions for conveniently analyzing batches of observation IDs using the joblib module
"""
import os
import shutil
import warnings
from copy import deepcopy
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from astropy.time import Time
from joblib import Parallel, delayed

from .bat_skyview import BatSkyView
from .bat_survey import MosaicBatSurvey, BatSurvey
from .bat_tte import BatEvent
from .batlib import combine_survey_lc as serial_combine_survey_lc
from .batlib import (
    dirtest,
    datadir,
    calc_response,
    calculate_detection,
    fit_spectrum,
    download_swiftdata,
)
from .batproducts import Spectrum
from .mosaic import (
    _mosaic_loop,
    merge_mosaics,
    finalize_mosaic,
    read_correctionsmap,
    read_skygrids,
)

# for python>3.6
try:
    import heasoftpy.swift as hsp
    import heasoftpy.utils as hsp_util

except ModuleNotFoundError as err:
    # Error handling
    print(err)


def _remove_pfiles():
    """
    This function removes the pfiles located in ~/pfiles so there is no conflict with the pfiles when running things in
    parallel
    :return:
    """
    direc = Path("~/pfiles").expanduser().resolve()

    if len(sorted(direc.glob("*"))) > 0:
        os.system(f"rm {direc}/*")


def _create_BatSurvey(
        obs_id,
        obs_dir=None,
        input_dict=None,
        recalc=False,
        load_dir=None,
        patt_noise_dir=None,
        verbose=False,
):
    """
    The inner loop that attempts to run batsurvey on a survey observation ID. If ther eis a load file saved already, it
    will try to load the BATSurvey object otherwise it will call batsurvey. This will return a BATSurvey object if the
    batsurvey code succesfully completes, otherwise it will return None.

    :param obs_id: string of the survey observation ID
    :param obs_dir: None or a Path object of where the directory that the observation ID data is located. This will most
        likely be the datadir path. None defaults to using the datadir() output
    :param input_dict: The input dictionary that will be passed to the heasoft batsurvey call
    :param recalc: Boolean False by default. The default, will cause the function to try to load a save file to save on
        computational time. If set to True, do not try to load the results of prior calculations. Instead rerun
        batsurvey on the observation ID.
    :param load_dir: Default None or a Path object. The default uses the directory as pointed to by
        obs_dir/obs_id+'_surveyresult' to try to look for a .batsurvey file to load.
    :param patt_noise_dir: String of the directory that holds the pre-calculated pattern noise maps for BAT.
        None defaults to looking for the maps in a folder called: "noise_pattern_maps" located in the ba.datadir()
        directory. If this directory doesn't exist then pattern maps are not used.
    :param verbose: Boolean False by default. Tells the code to print progress/diagnostic information.
    :return: None or a BATSurvey object
    """

    print(f"Working on Obsid {obs_id}")
    try:
        obs = BatSurvey(
            obs_id,
            obs_dir=obs_dir,
            recalc=recalc,
            load_dir=load_dir,
            input_dict=input_dict,
            verbose=verbose,
            patt_noise_dir=patt_noise_dir,
        )
        # see if there is already a .pickle file, if there is not or if the user wants to recalc, then do the save
        if not obs.result_dir.joinpath("batsurvey.pickle").exists() or recalc:
            obs.save()
    except ValueError as ve:
        print(f"{ve}")
        obs = None

    print(f"Done with Obsid {obs_id}")

    return obs


def batsurvey_analysis(
        obs_id_list,
        input_dict=None,
        recalc=False,
        load_dir=None,
        patt_noise_dir=None,
        verbose=False,
        nprocs=1,
):
    """
    Calls batsurvey for a set of observation IDs. Can process the observations in parallel if nprocs does not equal one.

    :param obs_id_list: list of strings that denote the observation IDs to run batsurvey on
    :param input_dict: user defined dictionary of key/value pairs that will be passed to batsurvey
    :param recalc:  Boolean False by default. The default, will cause the function to try to load a save file to save
        on computational time. If set to True, do not try to load the results of prior calculations. Instead rerun
        batsurvey on the observation ID.
    :param load_dir: Default None or a Path object. The default uses the directory as pointed to by
        obs_dir/obs_id+'_surveyresult' to try to look for a .batsurvey file to load.
    :param patt_noise_dir: String of the directory that holds the pre-calculated pattern noise maps for BAT.
        None defaults to looking for the maps in a folder called: "noise_pattern_maps" located in the ba.datadir()
        directory. If this directory doesn't exist then pattern maps are not used.
    :param verbose: Boolean False by default. Tells the code to print progress/diagnostic information.
    :param nprocs: The number of processes that will be run simulaneously. This number should not be larger than the
        number of CPUs that a user has available to them.
    :return: a list of BATSurvey objects for all the observation IDs that completed successfully.
    """

    _remove_pfiles()

    obs = Parallel(n_jobs=nprocs)(
        delayed(_create_BatSurvey)(
            i,
            obs_dir=datadir(),
            recalc=recalc,
            load_dir=load_dir,
            input_dict=input_dict,
            patt_noise_dir=patt_noise_dir,
            verbose=verbose,
        )
        for i in obs_id_list
    )

    final_obs = [i for i in obs if i is not None]

    return final_obs


def _survey_spectrum_analysis(
        obs,
        source_name,
        recalc=False,
        generic_model=None,
        setPars=None,
        fit_iterations=1000,
        use_cstat=True,
        ul_pl_index=2,
        nsigma=3,
        bkg_nsigma=5,
):
    """
    Calculate and fit a spectrum for a source at a single pointing.

    :param obs: the BATSurvey object for the survey pointings that will have their spectra exttracted and fitted.
    :param source_name: String of the source name as it appears in the BAT Survey catalog
    :param recalc: Boolean False by default. The default, will cause the function to try to load a save file to save on
        computational time. If set to True, do not try to load the results of prior calculations. Instead rerun the
        fitting on the pointings of the observation ID.
    :param generic_model: Default None or a generic model that can be passed to pyXspec, see the pyXspec documentation
        or the fit_spectrum docustring for more information on how to define this. The default None uses the basic
        powerlaw function (see the fit_spectrum function)
    :param setPars: None or a dictionary to specify values for the pyXspec model parameters. The value of None defaults
        to using the default parameter values found in the fit_spectrum function. More inforamtion on how to define this
        can be found by looking at the fit_spectrum docustring or the pyXspec documentation.
    :param fit_iterations: Integer, default 100, that defines the maximum iterations that can occur to conduct the
        fitting
    :param use_cstat: boolean, default False, to determine if CSTAT statistics should be used. In very bright sources,
        with lots of counts this should be set to False. For sources with small counts where the errors are not expected
        to be gaussian, this should be set to True.
    :param ul_pl_index: Float (default 2) denoting the power law photon index that will be used to obtain a flux upper
        limit
    :param nsigma: Integer, denoting the number for sigma the user needs to justify a detection
    :param bkg_nsigma: Integer, denoting the number of sigma the user needs to calculate flux upper limit in case
        of a non detection.

    :return: The updated BATSurvey object with updated spectral information
    """

    if recalc:
        val = True
    else:
        try:
            # otherwise check if theere are no pha files or no model_params key for any of the pointings
            pointing_id_test = 0
            for i in obs.get_pointing_ids():
                pointing_id_test += "model_params" in obs.get_pointing_info(
                    i, source_name
                )
            val = (len(obs.get_pha_filenames()) <= 0) or (
                    len(obs.get_pointing_ids()) != pointing_id_test
            )

            # if this is true then we should set recalc=True to redo the calculations for this observation ID
            if val:
                recalc = True
        except ValueError:
            val = True

    if val:
        if isinstance(obs, MosaicBatSurvey):
            print(
                "Running calculations for mosaic",
                obs.get_pointing_info("mosaic")["utc_time"],
            )
        else:
            print("Running calculations for observation id", obs.obs_id)

        obs.merge_pointings()

        obs.load_source_information(source_name)

        try:
            obs.calculate_pha(id_list=source_name, clean_dir=recalc)
            pha_list = obs.get_pha_filenames(id_list=source_name)
            if len(pha_list) > 0:
                if not isinstance(obs, MosaicBatSurvey):
                    calc_response(pha_list)

                # delete the upper limit key if necessary
                for i in obs.get_pointing_ids():
                    try:
                        obs.get_pointing_info(i, source_name).pop(
                            "nsigma_lg10flux_upperlim", None
                        )
                    except ValueError:
                        # if the source doesnt exist, just continue
                        pass

                # Loop over individual PHA pointings
                for pha in pha_list:
                    fit_spectrum(
                        pha,
                        obs,
                        use_cstat=use_cstat,
                        plotting=False,
                        verbose=False,
                        generic_model=generic_model,
                        setPars=setPars,
                        fit_iterations=fit_iterations,
                    )

                calculate_detection(
                    obs,
                    source_name,
                    pl_index=ul_pl_index,
                    nsigma=nsigma,
                    bkg_nsigma=bkg_nsigma,
                    verbose=False,
                )
                obs.save()
            else:
                print(
                    f"The source {source_name} was not found in the image and thus does not have a PHA file to analyze."
                )
                for i in obs.get_pointing_ids():
                    obs.set_pointing_info(
                        i, "model_params", None, source_id=source_name
                    )
                obs.save()
        except FileNotFoundError as e:
            print(e)
            print(
                f"This means that the batsurvey script didnt deem there to be good enough statistics for {source_name} "
                f"in this observation ID."
            )

    return obs


def batspectrum_survey_analysis(
        batsurvey_obs_list,
        source_name,
        recalc=False,
        generic_model=None,
        setPars=None,
        fit_iterations=1000,
        use_cstat=True,
        ul_pl_index=2,
        nsigma=3,
        bkg_nsigma=5,
        nprocs=1,
):
    """
    Calculates and fits the spectra for a single source across many BAT Survey observations in parallel.

    :param batsurvey_obs_list: list of BATSurvey observation objects
    :param source_name:  String of the source name as it appears in the BAT Survey catalog
    :param recalc: Boolean False by default. The default, will cause the function to try to load a save file to save on
        computational time. If set to True, do not try to load the results of prior calculations. Instead rerun the
        fitting on the pointings of the observation ID.
    :param generic_model: Default None or a generic model that can be passed to pyXspec, see the pyXspec documentation
        or the fit_spectrum docustring for more information on how to define this. The default None uses the basic
        powerlaw function (see the fit_spectrum function)
    :param setPars: None or a dictionary to specify values for the pyXspec model parameters. The value of None defaults
        to using the default parameter values found in the fit_spectrum function. More inforamtion on how to define this
        can be found by looking at the fit_spectrum docustring or the pyXspec documentation.
    :param fit_iterations: Integer, default 100, that defines the maximum iterations that can occur to conduct the
        fitting
    :param use_cstat: boolean, default False, to determine if CSTAT statistics should be used. In very bright sources,
        with lots of counts this should be set to False. For sources with small counts where the errors are not expected
        to be gaussian, this should be set to True.
    :param ul_pl_index: Float (default 2) denoting the power law photon index that will be used to obtain a flux upper
        limit
    :param nsigma: Integer, denoting the number for sigma the user needs to justify a detection
    :param bkg_nsigma: Integer, denoting the number of sigma the user needs to calculate flux upper limit in case of
        a non detection.
    :param nprocs: The number of processes that will be run simulaneously. This number should not be larger than the
        number of CPUs that a user has available to them.
    :return: a list of BATSurvey objects for all the observation IDs with updated spectral information
    """

    _remove_pfiles()

    not_list = False
    if type(batsurvey_obs_list) is not list:
        not_list = True
        batsurvey_obs_list = [batsurvey_obs_list]

    obs = Parallel(n_jobs=nprocs)(
        delayed(_survey_spectrum_analysis)(
            i,
            source_name=source_name,
            recalc=recalc,
            use_cstat=use_cstat,
            generic_model=generic_model,
            setPars=setPars,
            fit_iterations=fit_iterations,
            ul_pl_index=ul_pl_index,
            nsigma=nsigma,
            bkg_nsigma=bkg_nsigma,
        )
        for i in batsurvey_obs_list
    )

    # if this wasnt a list, just return the single object otherwise return the list
    if not_list:
        return obs[0]
    else:
        return obs


def _TTE_spectrum_analysis(
        spectrum,
        recalc=False,
        generic_model=None,
        setPars=None,
        fit_iterations=1000,
        use_cstat=True,
        ul_pl_index=2,
        nsigma=3,
        bkg_nsigma=5,
):
    """
    Fit a single spectrum object and determine if a detection has been made. If a detection has been made, the Spectrum
    object is returned with it's spectral_model attribute updated. If no detection has been made, a new Spectrum object
    is returned where the spectrum correponds to the upper limit spectrum that is calculated and is used to place
    flux upper limits on the source that the spectrum corresponds to.

    :param spectrum: The Spectrum object or list of Spectrum objects which contains the spectrum/spectra that will be fit.
    :param recalc: Boolean False by default. The default, will cause the function to try to load a save file to save on
        computational time. If set to True, do not try to load the results of prior calculations. Instead rerun the
        fitting on the pointings of the observation ID.
    :param generic_model: Default None or a generic model that can be passed to pyXspec, see the pyXspec documentation
        or the fit_spectrum docustring for more information on how to define this. The default None uses the basic
        powerlaw function (see the fit_spectrum function)
    :param setPars: None or a dictionary to specify values for the pyXspec model parameters. The value of None defaults
        to using the default parameter values found in the fit_spectrum function. More inforamtion on how to define this
        can be found by looking at the fit_spectrum docustring or the pyXspec documentation.
    :param fit_iterations: Integer, default 100, that defines the maximum iterations that can occur to conduct the
        fitting
    :param use_cstat: boolean, default False, to determine if CSTAT statistics should be used. In very bright sources,
        with lots of counts this should be set to False. For sources with small counts where the errors are not expected
        to be gaussian, this should be set to True.
    :param ul_pl_index: Float (default 2) denoting the power law photon index that will be used to obtain a flux upper
        limit
    :param nsigma: Integer, denoting the number for sigma the user needs to justify a detection
    :param bkg_nsigma: Integer, denoting the number of sigma the user needs to calculate flux upper limit in case
        of a non detection.
    :return: a Spectrum object with updated spectral information. If there is a detection of a source, then the
        returned Spectrum object will be identical to what is passed in. If there is no detection, then the Spectrum
        that is returned will be that correspondent to the upper limit spectrum that gets fit.
    """

    # determine if we really need to do the fitting again. If recalc=True, do the refitting. If not then see which spectra
    # have been fit already and thus be skipped in fitting.
    if recalc:
        val = True
    else:
        try:
            if spectrum.spectral_model is None:
                val = True
            else:
                val = False
        except AttributeError as e:
            # if the attribute doesnt exist then it hasnt been set so need to do the fit
            val = True

    if val:
        fit_spectrum(
            spectrum,
            use_cstat=use_cstat,
            plotting=False,
            verbose=False,
            generic_model=generic_model,
            setPars=setPars,
            fit_iterations=fit_iterations,
        )

        return_spect = calculate_detection(
            spectrum,
            pl_index=ul_pl_index,
            nsigma=nsigma,
            bkg_nsigma=bkg_nsigma,
            verbose=False,
        )
    else:
        return_spect = spectrum

    return return_spect


def batspectrum_TTE_analysis(
        spectrum,
        recalc=False,
        generic_model=None,
        setPars=None,
        fit_iterations=1000,
        use_cstat=True,
        ul_pl_index=2,
        nsigma=3,
        bkg_nsigma=5,
        nprocs=1,
):
    """
    Calculates and fits the spectra that are passed in parallel. These should be BatAnalysis.Spectrum objects in a list.
    Note that this function returns spectrum objects that have the spectral_model parameter filled in, when there is a
    detection measured at the specified significance level. If there is not a detection measured, the returned Spectrum
    is correspondent to the upper limit spectrum that is calculated and fitted to obtain upper limit fluxes.

    :param spectrum: The Spectrum object or list of Spectrum objects which contains the spectrum/spectra that will be fit.
    :param recalc: Boolean False by default. The default, will cause the function to try to load a save file to save on
        computational time. If set to True, do not try to load the results of prior calculations. Instead rerun the
        fitting on the pointings of the observation ID.
    :param generic_model: Default None or a generic model that can be passed to pyXspec, see the pyXspec documentation
        or the fit_spectrum docustring for more information on how to define this. The default None uses the basic
        powerlaw function (see the fit_spectrum function)
    :param setPars: None or a dictionary to specify values for the pyXspec model parameters. The value of None defaults
        to using the default parameter values found in the fit_spectrum function. More inforamtion on how to define this
        can be found by looking at the fit_spectrum docustring or the pyXspec documentation.
    :param fit_iterations: Integer, default 100, that defines the maximum iterations that can occur to conduct the
        fitting
    :param use_cstat: boolean, default False, to determine if CSTAT statistics should be used. In very bright sources,
        with lots of counts this should be set to False. For sources with small counts where the errors are not expected
        to be gaussian, this should be set to True.
    :param ul_pl_index: Float (default 2) denoting the power law photon index that will be used to obtain a flux upper
        limit
    :param nsigma: Integer, denoting the number for sigma the user needs to justify a detection
    :param bkg_nsigma: Integer, denoting the number of sigma the user needs to calculate flux upper limit in case of
        a non detection.
    :param nprocs: The number of processes that will be run simulaneously. This number should not be larger than the
        number of CPUs that a user has available to them.
    :return: a list of Spectrum objects with updated spectral information. If there is a detection of a source, then the
        returned spectrum object will be identical to what is passed in. If there is no detection, then the Spectrum
        that is returned will be that correspondent to the upper limit spectrum that gets fit.
    """

    _remove_pfiles()

    not_list = False
    if type(spectrum) is not list:
        not_list = True
        spectrum = [spectrum]
    else:
        # verify that we have all elements be a Spectrum object
        if np.any([not isinstance(i, Spectrum) for i in spectrum]):
            raise ValueError("The input list must be made of all BatAnalysis Spectrum objects. "
                             "Please ensure that this condition is met.")

    out_spect = Parallel(n_jobs=nprocs)(
        delayed(_TTE_spectrum_analysis)(
            i,
            recalc=recalc,
            use_cstat=use_cstat,
            generic_model=generic_model,
            setPars=setPars,
            fit_iterations=fit_iterations,
            ul_pl_index=ul_pl_index,
            nsigma=nsigma,
            bkg_nsigma=bkg_nsigma,
        )
        for i in spectrum
    )

    # if this wasnt a list, just return the single object otherwise return the list
    if not_list:
        return out_spect[0]
    else:
        return out_spect


def batspectrum_analysis(*args, **kwargs):
    """
    This is a wrapper function that allows users to pass in arguments for fitting spectra produced from BAT survey data
    or TTE data. For fitting survey data spectra, see the documentation for batspectrum_survey_analysis for the values
    that need to be passed in/can be passed in. For TTE spectra, using the Spectrum object, refer to
    the batspectrum_TTE_analysis function for potential inputs to this function.

    :return: See the documentation for
    """

    # input can be a list of/single Spectrum objects or a list of/single BatSurvey objects, or a list of/single MosaicBatSurvey
    # object.

    if type(args[0]) is list:
        test_arg = args[0][0]
    else:
        test_arg = args[0]

    if isinstance(test_arg, Spectrum):
        # we have a spectrum object
        ret = batspectrum_TTE_analysis(*args, **kwargs)
    elif isinstance(test_arg, BatSurvey) or isinstance(test_arg, MosaicBatSurvey):
        # we are passing in a phafilename for
        ret = batspectrum_survey_analysis(*args, **kwargs)
    else:
        raise ValueError("The inputs cannot be parsed appropriately. Please consult the documentation for "
                         "batspectrum_TTE_analysis or batspectrum_survey_analysis for the values that should be passed in.")

    return ret


def batmosaic_analysis(
        batsurvey_obs_list,
        outventory_file,
        time_bins,
        catalog_file=None,
        compute_total_mosaic=True,
        total_mosaic_savedir=None,
        recalc=False,
        nprocs=1,
):
    """
    Calculates the mosaic images in parallel.

    :param batsurvey_obs_list: The list of BATSurvey objects that correpond to the observations listed in the
        outventory file parameter
    :param outventory_file: Path object of the outventory file that contains all the BAT survey observations that will
        be used to create the mosaiced images.
    :param time_bins: astropy Time array of the time bin edges that are created based on the user specification of the
        group_outventory function
    :param catalog_file: A Path object of the catalog file that should be used to identify sources in the mosaic images.
        This will default to using the catalog file that is included with the BatAnalysis package.
    :param compute_total_mosaic: Default True, set to False to skip the computation of the total mosaic and return
        a single object.
    :param total_mosaic_savedir: Default None or a Path object that denotes the directory that the total
        "time-integrated" images will be saved to. The default is to place the total mosaic image in a directory
        called "total_mosaic" located in the same directory as the outventory file.
    :param recalc: Boolean False by default. If this calculation was done previously, do not try to load the results of
        prior calculations. Instead recalculate the mosaiced images. The default, will cause the function to try to load
        a save file to save on computational time.
    :param nprocs: The number of processes that will be run simulaneously. This number should not be larger than the
        number of CPUs that a user has available to them.
    :return:
    """

    _remove_pfiles()

    # make sure its a path object
    outventory_file = Path(outventory_file)

    # get the corections map and the skygrids
    corrections_map = read_correctionsmap()
    ra_skygrid, dec_skygrid = read_skygrids()

    # determine format of the time_bins, ie an astropy Time array or a list of astropy Time arrays
    # no error checking here since it should be taken care of in group_outventory function
    time_bins_is_list = False
    if type(time_bins) is list:
        time_bins_is_list = True

    if not time_bins_is_list:
        # get the lower and upper time limits
        start_t = time_bins[:-1]
        end_t = time_bins[1:]
    else:
        start = []
        end = []
        for i in time_bins:
            start.append(i[0, 0])
            end.append(i[1, 0])

        start_t = Time(start)
        end_t = Time(end)

    if recalc:
        # make sure that the time bins are cleared
        for i in start_t:
            binned_savedir = outventory_file.parent.joinpath(
                f"mosaic_{i.datetime64.astype('datetime64[D]')}"
            )
            if not binned_savedir.exists():
                binned_savedir = outventory_file.parent.joinpath(
                    f"mosaic_{i.mjd}"
                )

            dirtest(binned_savedir)

    all_mosaic_survey = Parallel(n_jobs=nprocs)(
        delayed(_mosaic_loop)(
            outventory_file,
            start,
            end,
            corrections_map,
            ra_skygrid,
            dec_skygrid,
            batsurvey_obs_list,
            recalc=recalc,
            verbose=True,
        )
        for start, end in zip(start_t, end_t)
    )  # i in range(len(start_t)))

    final_mosaics = [i for i in all_mosaic_survey if i is not None]

    # if batcelldetect hasnt been run yet do so
    for i in final_mosaics:
        if not i.result_dir.joinpath("sources_tot.cat").exists():
            i.detect_sources(catalog_file=catalog_file)
            i.save()

    intermediate_mosaic_dir_list = [i.result_dir for i in final_mosaics]

    if compute_total_mosaic:
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
        else:
            total_mosaic = MosaicBatSurvey(total_mosaic_savedir)

        # if batcelldetect hasnt been run yet do so
        if not total_mosaic.result_dir.joinpath("sources_tot.cat").exists():
            total_mosaic.detect_sources(catalog_file=catalog_file)
            total_mosaic.save()

        return final_mosaics, total_mosaic
    else:
        return final_mosaics


"""
def download_swiftdata(table,  reload=False,
                        bat=True, auxil=True, log=False, uvot=False, xrt=False,
                        save_dir=None, nprocs=1, **kwargs):
    #sys.setrecursionlimit(10000)

    download_status=Parallel(n_jobs=nprocs, require='sharedmem')(
        delayed(download_swiftdata)(i, reload=reload, bat=bat, auxil=auxil, log=log, uvot=uvot, xrt=xrt,
                        save_dir=save_dir) for i in table)

    return download_status
"""


def download_swiftdata(
        table,
        reload=False,
        bat=True,
        auxil=True,
        log=False,
        uvot=False,
        xrt=False,
        save_dir=None,
        nprocs=1,
):
    # create temporary functions that will be called separately to download the data
    dl = lambda x: download_swiftdata(
        x,
        reload=reload,
        bat=bat,
        auxil=auxil,
        log=log,
        uvot=uvot,
        xrt=xrt,
        save_dir=save_dir,
    )

    # Run the function threaded. nprocs at a time.
    results = ThreadPool(nprocs).imap_unordered(dl, table)

    # combine the results into a dictionary that is typically output from download_swiftdata
    all_results = {}
    for i in results:
        all_results.update(i)

    return all_results


def combine_survey_lc(survey_obsid_list, output_dir=None, clean_dir=True, nprocs=1):
    if type(survey_obsid_list) is not list:
        survey_obsid_list = [survey_obsid_list]

    # create a list of subdirectories to hold parallelized catmux results
    sub_dirs = [
        survey_obsid_list[0].result_dir.parent.joinpath(f"total_lc_{i}")
        for i in range(nprocs)
    ]

    # setup the lc_dir
    if output_dir is None:
        lc_dir = survey_obsid_list[0].result_dir.parent.joinpath("total_lc")
    else:
        lc_dir = Path(output_dir)

    # reset/make the lc_dir if necessary
    dirtest(lc_dir, clean_dir=clean_dir)

    # create a list of sublists of the observations
    sublist = np.array_split(survey_obsid_list, nprocs)

    # combine the subsets of survey data
    all_catmux = Parallel(n_jobs=nprocs)(
        delayed(serial_combine_survey_lc)(
            list(surveys), output_dir=direc, clean_dir=clean_dir
        )
        for direc, surveys in zip(sub_dirs, sublist)
    )  # i in range(len(start_t)))

    # combine the files in the subdirectories
    source_names = []
    for i in sub_dirs:
        files = sorted(list(i.glob("*.cat")))
        for j in files:
            source_names.append(j.name)

    # get the unique file names
    uniq_source_names = np.unique(source_names)
    # data=dict().fromkeys(list(uniq_source_names)) maybe dont need this

    # concatenate the subdirectories
    for name in uniq_source_names:
        data = []
        for i in sub_dirs:
            if i.joinpath(f"{name}").exists():
                data.append(Table.read(i.joinpath(f"{name}")))
        all_data = vstack(data)
        all_data.write(lc_dir.joinpath(f"{name}"), format="fits")

    # remove the subdirectories
    for i in sub_dirs:
        shutil.rmtree(i)

    return lc_dir


def create_event_pha(batevent, nprocs=1, **kwargs):
    """
    This convenience functon allows pha files to be created in parallel for event data.

    :param batevent: the BatEvent object that the user would like to construct PHA files for
    :param nprocs: int, the number of procs that the user would like to use to create PHA files
    :param kwargs: parameters that will be passed to the BatEvent.create_pha method. See documentation for appropriate
        parameters and their values
    :return: list of created BatSpectrum objects which contain the created PHA files.
        These BatSpectrum objects are also appended to the batevent object's spectra property.
    """

    def _single_pha_calc(batevent, *args, **kwargs):
        with hsp_util.local_pfiles_context():
            pha = batevent.create_pha(*args, **kwargs)
        return pha

    if not isinstance(batevent, BatEvent):
        raise ValueError("A BatEvent object needs to be passed in. ")

    # go through the keys for timebins or tstart/tstop, use the same priority ordering as the create_pha method
    if "tstart" in kwargs or "tstop" in kwargs:
        # make sure that we have both
        if "tstart" not in kwargs or "tstop" not in kwargs:
            raise ValueError("Both tstart and tstop need to be passed in. One is currently missing")

        start_t = kwargs.pop("tstart")
        end_t = kwargs.pop("tstop")
    elif "timebins" in kwargs:
        timebins = kwargs.pop("timebins")
        start_t = timebins[:-1]
        end_t = timebins[1:]
    else:
        raise ValueError("No time information has been passed in to create pha files in parallel. ")

    all_pha = Parallel(n_jobs=nprocs)(
        delayed(_single_pha_calc)(
            batevent, tstart=start, tstop=end, **kwargs
        )
        for start, end in zip(start_t, end_t)
    )

    for i in all_pha:
        batevent.spectra = i

    return all_pha


def create_event_dpi(batevent, nprocs=1, **kwargs):
    """
    This convenience functon allows DPIs to be created in parallel for event data.

    :param batevent: the BatEvent object that the user would like to construct DPIs for
    :param nprocs: int, the number of procs that the user would like to use to create DPIs
    :param kwargs: parameters that will be passed to the BatEvent.create_dpi method. See documentation for appropriate
        parameters and their values
    :return: list of created BatDPI objects which contain the created PHA files.
        These BatDPI objects are also appended to the batevent object's dpis property.
    """

    def _single_dpi_calc(batevent, *args, **kwargs):
        with hsp_util.local_pfiles_context():
            dpi = batevent.create_dpi(*args, **kwargs)
        return dpi

    if not isinstance(batevent, BatEvent):
        raise ValueError("A BatEvent object needs to be passed in. ")

    # go through the keys for timebins or tstart/tstop, use the same priority ordering as the create_pha method
    if "tstart" in kwargs or "tstop" in kwargs:
        # make sure that we have both
        if "tstart" not in kwargs or "tstop" not in kwargs:
            raise ValueError("Both tstart and tstop need to be passed in. One is currently missing")

        start_t = kwargs.pop("tstart")
        end_t = kwargs.pop("tstop")
    elif "timebins" in kwargs:
        timebins = kwargs.pop("timebins")
        start_t = timebins[:-1]
        end_t = timebins[1:]
    else:
        raise ValueError("No time information has been passed in to create pha files in parallel. ")

    all_dpi = Parallel(n_jobs=nprocs)(
        delayed(_single_dpi_calc)(
            batevent, tstart=start, tstop=end, **kwargs
        )
        for start, end in zip(start_t, end_t)
    )

    for i in all_dpi:
        batevent.dpis = i

    return all_dpi


def create_event_skyview(batevent, nprocs=1, parse_images=True, **kwargs):
    """
    This convenience functon allows BatSkyViews to be created in parallel for event data.

    :param batevent: the BatEvent object that the user would like to construct BatSkyViews for
    :param nprocs: int, the number of procs that the user would like to use to create BatSkyViews
    :param parse_images: boolean to denote if the different images in the skyview should be loaded into each SkyView
        object or not. If this is set to False, it helps save memory and is especially useful if the user only
         cares about the mosaiced image and not the skyviews themselves.
    :param kwargs: parameters that will be passed to the BatEvent.create_skyview method. See documentation for appropriate
        parameters and their values
    :return: list of created BatSkyViews objects. These BatSkyViews objects are also appended to the batevent object's skyviews property.
    """

    def _single_skyview_calc(batevent, *args, **kwargs):
        with hsp_util.local_pfiles_context():
            skyview = batevent.create_skyview(*args, **kwargs)

            # set all images to None to save memory
            skyview.pcode_img = None
            skyview.sky_img = None
            skyview.snr_img = None
            skyview.bkg_stddev_img = None

        return skyview

    if not isinstance(batevent, BatEvent):
        raise ValueError("A BatEvent object needs to be passed in. ")

    # go through the keys for timebins or tstart/tstop, use the same priority ordering as the create_pha method
    # the user can pass in dpis here as well, so we need to see if this is the case and use these instead of assuming
    # that the user will need to also create DPIs
    dpis = None
    if "dpis" in kwargs:
        dpis = kwargs.pop("dpis")
    elif "tstart" in kwargs or "tstop" in kwargs:
        # make sure that we have both
        if "tstart" not in kwargs or "tstop" not in kwargs:
            raise ValueError("Both tstart and tstop need to be passed in. One is currently missing")

        start_t = kwargs.pop("tstart")
        end_t = kwargs.pop("tstop")
    elif "timebins" in kwargs:
        timebins = kwargs.pop("timebins")
        start_t = timebins[:-1]
        end_t = timebins[1:]
    else:
        raise ValueError("No time information has been passed in to create pha files in parallel. ")

    if dpis is None:
        all_skyviews = Parallel(n_jobs=nprocs)(
            delayed(_single_skyview_calc)(
                batevent, tstart=start, tstop=end, **kwargs
            )
            for start, end in zip(start_t, end_t)
        )
    else:
        all_skyviews = Parallel(n_jobs=nprocs)(
            delayed(_single_skyview_calc)(
                batevent, dpis=i, **kwargs
            )
            for i in dpis
        )

    # need to reload all images within skyview and set the skyviews batevent property,
    # this can take 11 minutes for 74 skyviews. This is better since if we did this in the Parallel portion, lots of
    # memory would be needed to copy everything from the threads
    if parse_images:
        for i in all_skyviews:
            i._parse_skyimages()
            batevent.skyviews = i

    return all_skyviews


def _add_skyviews(skyviews):
    """
    This function loops over a list of BatSkyViews and mosaics them together. It attempts to save memory by only
    loading in the sky images associated with each BatSkyView that is being added at the given point in the loop.

    :param skyviews: list of BatSkyViews
    :return: BatSkyView mosaic
    """
    for count, i in enumerate(skyviews):
        if count == 0:
            new_skyview = deepcopy(i)
        else:
            reset = False
            if not i.is_mosaic and i.sky_img is None:
                i._parse_skyimages()
                reset = True

            if not new_skyview.is_mosaic and new_skyview.sky_img is None:
                new_skyview._parse_skyimages()

            new_skyview += i

            if reset:
                # set all images to None to save memory
                i.pcode_img = None
                i.sky_img = None
                i.snr_img = None
                i.bkg_stddev_img = None

    return new_skyview


def _add_two_skyviews(skyview1, skyview2):
    """
    This will do inplace modification of skyview1 if both skyviews are mosaic skyviews otherwise, we will ID the
    one that is a mosaic skyview and do inplace modification. If none are mosaics, then we create a new mosaic skyview
    using the + operator. This function also loads in the sky images associated with the BatSkyViews if necessary.

    :param skyview1: BatSkyView object
    :param skyview2: BatskyView object
    :return: BatSkyView mosaic of the two BatSkyViews added together.
    """

    if None not in [skyview1, skyview2]:
        # to also try and save memory, if we have a
        if skyview1.is_mosaic and skyview2.is_mosaic:
            skyview1 += skyview2
            return skyview1
        else:
            reset = False
            reset1 = False
            reset2 = False
            if not skyview1.is_mosaic and not skyview2.is_mosaic:
                if skyview1.sky_img is None:
                    skyview1._parse_skyimages()
                    reset1 = True

                if skyview2.sky_img is None:
                    skyview2._parse_skyimages()
                    reset2 = True

                mosaic = skyview1 + skyview2

                if reset1:
                    # set all images to None to save memory
                    skyview1.pcode_img = None
                    skyview1.sky_img = None
                    skyview1.snr_img = None
                    skyview1.bkg_stddev_img = None

                if reset2:
                    # set all images to None to save memory
                    skyview2.pcode_img = None
                    skyview2.sky_img = None
                    skyview2.snr_img = None
                    skyview2.bkg_stddev_img = None

                return mosaic

            elif skyview1.is_mosaic:
                if skyview2.sky_img is None:
                    skyview2._parse_skyimages()
                    reset = True

                skyview1 += skyview2

                if reset:
                    # set all images to None to save memory
                    skyview2.pcode_img = None
                    skyview2.sky_img = None
                    skyview2.snr_img = None
                    skyview2.bkg_stddev_img = None

                return skyview1
            elif skyview2.is_mosaic:
                if skyview1.sky_img is None:
                    skyview1._parse_skyimages()
                    reset = True

                skyview2 += skyview1

                if reset:
                    # set all images to None to save memory
                    skyview1.pcode_img = None
                    skyview1.sky_img = None
                    skyview1.snr_img = None
                    skyview1.bkg_stddev_img = None

                return skyview2
            else:
                raise RuntimeError("The passed in skyviews cannot be determined to be a mosaic or not which is not "
                                   "typical.")
    else:
        ret = [i for i in [skyview1, skyview2] if i is not None]
        return ret[0]


def mosaic_skyview(skyview_list, healpix_nside=512, projection="healpix", healpix_coordsys="galactic", nprocs=1):
    """
    Mosaic skyviews together in parallel by projecting them into a healpix map with a resolution set by healpix_nside
    and the coordinate system defined in healpix_coordsys

    :param skyview_list: list of BatSkyViews that will be mosaiced
    :param healpix_nside: int representing the healpix map resolution
    :param projection: str representing the type of projection that is done to mosaic the images together. Currently
        only healpix is the only string that is allowed.
    :param healpix_coordsys: str representing the healpix map coordinate system. Can be "galactic" or "icrs"
    :param nprocs: int, the number of processes that the user would like to use to create the mosaic.
        THIS FUNCTION CAN BE MEMORY INTENSIVE. ENSURE THAT ~10-15 GB OF MEMORY PER PROCESS IS AVAILABLE.
    :return: BatSkyView mosaic
    """
    if type(skyview_list) is not list:
        raise ValueError("A list of BatSkyView objects need to be passed in.")
    else:
        if np.any([not isinstance(i, BatSkyView) for i in skyview_list]):
            raise ValueError(
                "Every element of the list that is passed in for skyview_list needs to be a BatSkyView object.")

    for i in skyview_list:
        i.healpix_nside = healpix_nside
        i.projection = projection
        i.healpix_coordsys = healpix_coordsys

    if nprocs != 1:
        warnings.warn("This parallel image addition can be memory intensive. Allocate ~10-15 GB of memory per process.",
                      stacklevel=2)

        with Pool(processes=nprocs) as pool:

            if len(skyview_list) % 2 != 0:
                skyview_list.append(None)
            output = pool.starmap(_add_two_skyviews, zip(skyview_list[::2], skyview_list[1::2]))

        mosaic_skyview = _add_skyviews(output)

    else:
        mosaic_skyview = _add_skyviews(skyview_list)

    # depending on the nprocs we can have a list or not, if we have a list get the 0th element and ake sure that it is the
    # only element
    if type(mosaic_skyview) is list:
        if len(mosaic_skyview) != 1:
            raise ValueError("There should only be 1 element in the returned calculation of the mosaic.")
        return mosaic_skyview[0]
    else:
        return mosaic_skyview
