"""
This file contains the survey class that is inherited from the observation class. this class contains additional
information about a given survey. it also reads in survey data and processes it

Tyler Parsotan April 5 2023
"""

import os
import pickle
import re
import shutil
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, Column
from astropy.coordinates import SkyCoord
from sklearn.cluster import DBSCAN
import astropy.units as u
from dataclasses import dataclass, field
from typing import List

from .batlib import datadir, dirtest, met2mjd, met2utc, calculate_effective_snr
from .batobservation import BatObservation
from .bat_survey_tools import BatTools
from .bat_truncated import patch_truncated_obsid, make_files_consistent


# for python>3.6
try:
    import heasoftpy.swift as hsp
    import heasoftpy.utils as hsp_util
    from heasoftpy import heatools
    import heasoftpy as hsp_core
except ModuleNotFoundError as err:
    # Error handling
    print(err)

@dataclass
class _PointStats:
    image_id: str
    status: bool
    descr: str
    tstart: float
    tstop: float
    raw_exposure: float
    exposure: float
    ra_pnt: float
    dec_pnt: float
    pa_pnt: float
    ndets: int
    date_obs: str = ""
    date_end: str = ""
    numband: int = 0
    chi2: List[float] = field(default_factory=list)
    bkg_counts: List[float] = field(default_factory=list)


def add_additional_info(file):
    """To the survey catalo files egnerated by batcelldetect, add additional information such as GLAT, GLON
        FACET CENTER distance

    Args:
        file (str): Path to the catalog file

    Returns:
    """

    heatools.ftcoco(
        infile=file,
        outfile=file,
        incoord="R",
        outcoord="G",
        lon1="RA_OBJ",
        lat1="DEC_OBJ",
        lon2="GLON_OBJ",
        lat2="GLAT_OBJ",
        clobber="YES",
    )

    heatools.ftsort(
        infile=f"{file}[col *;FACET_DIST = ANGSEP(RA_OBJ,DEC_OBJ,CRVAL1,CRVAL2); DUP = F]",
        outfile=file,
        columns="CATNUM, FACET_DIST",
        clobber="YES",
    )

    # Also copy over the position error information to a new column
    old_hdu = fits.open(file.replace(".tmp", ""))

    ra_err = old_hdu[1].data["RA_OBJ_ERR"]
    dec_err = old_hdu[1].data["DEC_OBJ_ERR"]

    old_hdu.close()
    # Also add other info
    hdu = fits.open(file, mode="update")

    tab = Table(hdu[1].data)
    # Add a few columns if they dont exist
    tab.replace_column("RA_OBJ_ERR", Column(ra_err, name="RA_OBJ_ERR"))
    tab.replace_column("DEC_OBJ_ERR", Column(dec_err, name="DEC_OBJ_ERR"))
    tab.add_column(Column(["individual"] * len(tab), name="CAT_TYPE"))
    tab.add_column(
        Column(
            [file.replace("_src_catalog.fits.tmp", ".img")] * len(tab),
            name="IMAGE_PATH",
        )
    )

    hdu[1] = fits.BinTableHDU(data=tab, header=hdu[1].header)
    hdu.flush()
    hdu.close()
    # Move this file to remove the .tmp, use shutil to be cross-platform
    # First remove any existing file without .tmp
    Path(file.replace(".tmp", "")).unlink(missing_ok=True)
    # except OSError:
    #     warnings.warn(
    #         f"Could not add additional information to the catalog file {file}."
    #     )
    shutil.move(src=file, dst=file.replace(".tmp", ""))


class BatSurvey(BatObservation):
    """
    A general Bat Survey object that holds all information necessary to analyze Bat survey data.

    Attributes
    ---------------
    obs_id : str
        observation ID
    obs_dir : str or None
        Directory that the observation ID folder resides within
    survey_input : dictionary
        Dictionary that holds the information that was passed to heasoft's batsurvey
    result_dir : str
        The directory that holds the output of the heasoft batsurvey calculations
    batsurvey_result : heasoftpy Result object
        The output of calling heasoftpy batsurvey
    pointing_flux_files : list of strings
        A list of the pointing files created by heasoftpy bat_survey for the specified obs_id
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
    load(f):
        Load a BatSurvey object
    save():
        Saves a BatSurvey object
    merge_pointings(input_dict=None, verbose=False):
        Merges the counts from multiple pointings found within an observation ID dataset
    calculate_pha(id_list=None, output_dir=None, calc_upper_lim=False, bkg_nsigma=None, verbose=True, clean_dir=False,
            single_pointing=None):
        Calculates the PHA file for each pointing found within an observation ID dataset
    load_source_information(sources):
        Loads the count rate, background variance, and snr from the .cat file produced by batsurvey for the sources of
        interest
    get_pointing_ids():
        Returns the pointing ids in the observation ID
    get_pointing_info(pointing_id, source_id=None)
        Gets the dictionary of information associated with the specified pointing id and source id if specified
    set_pointing_info(pointing_id, key, value, source_id=None)
        Sets the key/value pair for the dictionary of information associated with the specified pointing id and source,
        if the source_id is specified
    get_pha_filenames(id_list=None, pointing_id_list=None)
        Gets the pha filename list of the sources supplied in id_list and for the pointing ids supplied by
        pointing_id_list
    set_pha_filenames(file, reset=False)
        Sets the pha filenames attribute or resets it to be an empty list
    load_source_information(sources)
        Loads the rates information for a given source from the pointing_flux_files
    get_count_rate(energy_index, pointing_id, source)
        Returns the count rate information that is requested for a single (or multiple) energy range(s), a specified
            pointing ID, and a specified source.
    """

    def __init__(
        self,
        obs_id,
        obs_dir=None,
        input_dict=None,
        recalc=False,
        verbose=False,
        load_dir=None,
        patt_noise_dir=None,
        create_total_energy_image=True,
        use_independent_modules=False,
    ):
        """
        Constructs the BatSurvey object.

        Runs heasoft batsurvey on the observation ID folder. If this calculation was done previously and the results
        saved, the user can load the saved state.

        :param obs_id: String of the observation ID
        :param obs_dir: None or String of the location to the folder with the observation ID, defaults to datadir
            directory
        :param input_dict: Dictionary of values that will be passed to heasoftpy's batsurvey. The default values are:
                indir=obs_dir
                outdir=obs_dir + '_surveyresult'
                detthresh=10000
                detthresh2=10000
                incatalog=survey6b_2.cat (included with BatAnalysis code)
            Any parameters listed above that are excluded from a dictionary or set to None (not a string, but a python
            None object) will take on these values.
            A dictionary can take the form x=dict(incatalog="custom_catalog.cat", detthresh="10000"). Here, the
            remaining unspecified parameters will first take the values above and then the default values of
            heasoft's batsurvey.
        :param recalc: Boolean to either delete the existing batsurvey results and start over
        :param verbose: Boolean to print diagnostic information
        :param load_dir: String of the directory that holds the result directory of batsurvey for a given observation ID
        :param patt_noise_dir: String of the directory that holds the pre-calculated pattern noise maps for BAT.
            None defaults to looking for the maps in a folder called: "noise_pattern_maps" located in the ba.datadir()
            directory. If this directory doesn't exist then pattern maps are not used.
        :param add_total_energy_image: Boolean to add in the total energy band image and catalog to the batsurvey
            results. Default is True.
        :param use_independent_modules: Boolean to denote whether to use the independent modules of batsurvey instead of the
            full batsurvey command. Default is False, meaning that the full batsurvey command will be used. If True, then
            the individual modules of batsurvey will be called in sequence. This allows for more control and flexibility
            in the survey process, but also requires more time and computational resources.
        """

        # Set default energy ranges in keV and system errors
        self.channel = [1, 2, 3, 4, 5, 6, 7, 8]
        self.emin = [14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0]
        self.emax = [20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0]
        self.syserr = [0.6, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15, 0.6]

        # make sure that the observation ID is a string
        if type(obs_id) is not str:
            obs_id = f"{int(obs_id)}"

        # initialize super class
        super().__init__(obs_id, obs_dir)

        self.truncated = self._identify_truncated_files()
        self.use_independent_modules = use_independent_modules
        self._create_total_energy_image=create_total_energy_image

        # Check for pattern maps
        if patt_noise_dir is None:
            patt_noise_dir = datadir().joinpath("noise_pattern_maps")
        else:
            # make a Path object
            patt_noise_dir = Path(patt_noise_dir)

        # If it is truncated data, use independent modules since the full batsurvey command will not work with truncated data
        if self.truncated:
            # The same for pattern noise maps since these are not compatible with truncated data,
            # in _get_pattern_noise_maps we will return NONE
            patt_noise_dir = None

        #if we have truncated data and the user didnt specify use_independent_module set it to be true
        if self.truncated and not self.use_independent_modules:
            self.use_independent_modules=True
            warnings.warn(f"BatAnalysis detected truncated DPHs for obsid {obs_id}. Now setting use_independent_modules=True to process the data.")

        # Get the pattern maps
        self.patt_noise_dir = patt_noise_dir

        self.survey_input = input_dict

        # initalize the pha filename list attribute
        self.pha_file_names_list = []

        # See if a loadfile exists, if we dont want to recalculate everything, otherwise remove any load file and
        # .batsurveycomplete file (this is produced only if the batsurvey calculation was completely finished, and thus
        # know that we can safely load the batsurvey.pickle file)
        if not recalc and load_dir is None:
            load_dir = sorted(self.obs_dir.parent.glob(obs_id + "_survey*"))

            # see if there are any _surveyresult dir or anything otherwise just use obs_dir as a placeholder
            if len(load_dir) > 0:
                load_dir = load_dir[0]
            else:
                load_dir = self.obs_dir
        elif not recalc and load_dir is not None:
            load_dir_test = sorted(Path(load_dir).glob(obs_id + "_survey*"))
            # see if there are any _surveyresult dir or anything otherwise just use load_dir as a placeholder
            if len(load_dir_test) > 0:
                load_dir = load_dir_test[0]
            else:
                load_dir = Path(load_dir)
        else:
            # just give dummy values that will be written over later
            load_dir = self.obs_dir

        load_file = load_dir.joinpath("batsurvey.pickle")
        complete_file = load_dir.joinpath(".batsurvey_complete")
        self._local_pfile_dir = load_dir.joinpath(".local_pfile")

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        # Print the summary
        # print(
        #     f"Processing observation ID {self.obs_id} (truncated={self.truncated}, use_independent_modules={self.use_independent_modules}) with the following parameters:"
        # )
        # for key, value in self.survey_input.items():
        #     print(f"  {key}: {value}")

        # if load_file is None:
        # if the user wants to recalculate things or if there is no batsurvey.pickle file, or if there is no
        # .batsurvey_complete file (meaning that the __init__ method didn't complete)
        if recalc or not load_file.exists() or not complete_file.exists():
            # batsurvey relies on "bat" and "auxil" folders in the observation ID folder, therefore we need to check
            # for these https://heasarc.gsfc.nasa.gov/ftools/caldb/help/batsurvey.html
            if (
                not self.obs_dir.joinpath("bat").joinpath("survey").is_dir()
                or not self.obs_dir.joinpath("auxil").is_dir()
            ):
                raise ValueError(
                    "The observation ID folder needs to contain the bat/survey/ and auxil/ subdirectories in order to "
                    + "analyze BAT survey data. One or both of these folders are missing."
                )

            # can do hsp.batsurvey? in ipython to see what the default parameters are if the input directory is None,
            # the user wants the default parameters and also wants the below specified default observation directory
            # (indir), the directory that the results will be saved into (outdir), the imput catalog (incatalog),
            # the detector thresholds (detthresh/detthresh2)

            # need to determine if there is a pattern_map_directory. If this is None use the ba.datadir() and see if
            # the directory exists. If so, check that the appropriate pattern map exists for the day of observation,
            # if it doesnt then load the pattern map for the day that is closest. If there are no pattern map files
            # at all then dont pass anything into batsurvey for these parameters

            # Get the input parameters for batsurvey
            self._get_survey_parameters()

            # save result directory
            self.result_dir = Path(self.survey_input["outdir"])

            # if the user has already done this calculation and wants to redo it, can set clobber to True in input_dict
            if recalc:
                self.survey_input["clobber"] = "YES"
                if self.result_dir.exists():
                    shutil.rmtree(self.result_dir)

            # if the user wants to relaculate things or if recalc==False but the result directory specified doesnt exist
            # we need to recalculate things for further processing, IMPLEMENT LATER ON
            # call the heasoftpy command
            bs = self._call_batsurvey(
                self.survey_input, use_independent_modules=self.use_independent_modules
            )
            self.batsurvey_result = bs
            # can print output of batsurvey with ba.stdout.split("\n")

            # self._local_pfile_dir = self.result_dir.joinpath(".local_pfile")

            # make the local pfile dir if it doesnt exist and set this value
            # self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
            # try:
            #     hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
            # except AttributeError:
            #     hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

            complete_file = self.result_dir.joinpath(".batsurvey_complete")

            # identify the pointings that have been created
            # all these pointings may not have the object of interest, so need to double-check that with the
            # merge_pointings method
            self.pointing_flux_files = sorted(
                self.result_dir.glob(f"point*/*_{self.survey_input['ncleaniter']}.cat")
            )

            # need to extract the respective pointing IDs
            self.pointing_ids = []
            for pointing in self.pointing_flux_files:
                self.pointing_ids.append(pointing.parent.name.split("_")[-1])

            # create dict of pointings ids and their respective information of time, exposure, etc which is the same
            # for each pointing
            self.pointing_info = dict.fromkeys(self.pointing_ids)
            for pointing, ident in zip(self.pointing_flux_files, self.pointing_ids):
                # In cases there no input catalog was provided, and no sources were found,
                # the .cat file will be empty and thus there will be no data HDU to read.
                # Hence instead of cat files, read img files to get the time/exposure info
                img = str(pointing).replace(".cat", ".img")
                img_hdu = fits.open(img)

                tstart = img_hdu[0].header["TSTART"]  # MET start time
                exposure = img_hdu[0].header["EXPOSURE"]  # time in seconds
                img_hdu.close()

                # calculate times in UTC and MJD units as well
                mjdtime = met2mjd(tstart)
                utctime = met2utc(tstart, mjd_time=mjdtime)

                # open the status file to save the success code
                status_file = pointing.parent.joinpath(f"point_{ident}_status.txt")
                stat = self._batsurvey_error(status_file)[0]

                self.pointing_info[ident] = dict(
                    success=stat,
                    fail_code=None,
                    met_time=tstart,
                    exposure=exposure,
                    utc_time=utctime,
                    mjd_time=mjdtime,
                )

            # see if there were any pointings that failed
            status_file = self.result_dir.joinpath("stats_point.dat")

            # sometimes this file may not exist if batsurvey determines that there are no GTIs at all
            # in this case there will be no pointing flux files so we will execute the if statement a little later on
            if status_file.exists():
                with open(status_file, "r") as f:
                    batsurvey_output = f.readlines()
                if len(batsurvey_output) != len(self.pointing_ids):
                    # get the info for the other pointings that failed and save their fail codes
                    for line in batsurvey_output:
                        # see if the pointing ID exists in the
                        pointing = line.split(" ")[3]

                        if pointing.split("_")[-1] not in self.pointing_info.keys():
                            # open the status file to save the success code
                            status_file = self.result_dir.joinpath(
                                f"{pointing}"
                            ).joinpath(f"{pointing}_status.txt")
                            if status_file.exists():
                                stat, fail_str = self._batsurvey_error(status_file)
                            else:
                                stat = False
                                fail_str = "The dpi could not be analyzed."

                            self.pointing_info[pointing.split("_")[-1]] = dict(
                                success=stat, fail_code=fail_str
                            )
            else:
                batsurvey_output = [
                    f"There were no GTI intervals found for this observation ID {self.obs_id}"
                ]

            # if there are pointings found, merge them
            if verbose:
                if len(self.pointing_flux_files) > 0:
                    print(
                        "There were %d pointings found for the obsid %s."
                        % (len(self.pointing_flux_files), self.obs_id)
                    )
                else:
                    print("No pointings were found.")


            # Then check for truncated data and update headers accordingly
            if len(self.pointing_flux_files) > 0:
                # Once all of this is done, add in the total energy band images and catalogs
                if self._create_total_energy_image:
                    self._add_total_energy_image()

                # Retrieve any new sources found in the survey data
                #if get_new_sources:
                #    self._detect_sources()
                #    self._get_unknown_sources()


            # Save the pickle file of the current state
            self.save()

            # create the marker file that tells us that the __init__ method completed successfully
            complete_file.touch()

            # if there are no pointings throw an error and save the state so we know what it is
            if len(self.pointing_flux_files) == 0:
                # see if there is a status.txt file that was produced and read it
                # then print out the error

                self.save()
                raise ValueError(
                    f"The results for each pointing of observation ID {self.obs_id} is:\n {' '.join(batsurvey_output)}"
                )
            # Remove the local pfile directory since we dont need it anymore
            shutil.rmtree(self._local_pfile_dir)

        else:
            load_file = Path(load_file).expanduser().resolve()
            self.load(load_file)

            if len(self.pointing_flux_files) == 0:
                # see if there is a status.txt file that was produced and read it
                # then print out the error

                self.save()
                raise ValueError("No pointings were able to be loaded.")

    def load(self, f):
        """
        Loads a saved BatSurvey object
        :param f: String of the file that contains the previously saved BatSurvey object
        :return: None
        """
        with open(f, "rb") as pickle_file:
            content = pickle.load(pickle_file)
        self.__dict__.update(content)

    def save(self):
        """
        Saves the current BatSurvey object
        :return: None
        """
        file = self.result_dir.joinpath("batsurvey.pickle")
        with open(file, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
        print("A save file has been written to %s." % (str(file)))

    def _get_pattern_noise_maps(self):
        """Helper function to get the pattern noise map and mask for the observation
        ID based on the time of the observation and the files available in the pattern
        noise map directory. If the appropriate pattern noise map for the day of the
        observation is not available, then the function will search for the nearest
        pattern noise map based on the time of the observation. If there are no pattern
        noise maps available at all, then the function will return "NONE" for both the
        pattern noise map and mask.

        Returns:
            patt_map_name (str): The filename of the pattern noise map to be used for the
            observation ID. If no pattern noise maps are available, then this will be "NONE".
            patt_mask_name (str): The filename of the pattern noise mask to be used for the
            observation ID. If no pattern noise maps are available, then this will be "NONE".
        """

        #if we have truncated DPHs we found that the pipeline crashes for some reason so just ignore them when processing this type of data
        if self.truncated:
            patt_map_name = "NONE"
            patt_mask_name = "NONE"
            warnings.warn("pattern noise maps are being ignored for the processing of the truncated DPHs. ")
            return patt_map_name, patt_mask_name


        input_file = sorted(
            self.obs_dir.joinpath("bat").joinpath("survey").glob("*dph*")
        )
        if len(input_file) == 0:
            raise ValueError(
                f"The observation ID folder {self.obs_dir} does not contain any survey data files."
            )
        else:
            input_file = input_file[0]
        with fits.open(str(input_file)) as file:
            tstart = file[0].header["TSTART"]

        time = met2utc(tstart)

        # get the day of the year, need to add 1 since day 1 is the first day of the year
        obs_doy = str(
            np.timedelta64((time - np.datetime64(time.astype("M8[Y]"), "D")), "D")
            + np.timedelta64(1, "D")
        ).split(" ")[0]
        obs_year = str(time.astype("M8[Y]"))

        # see if the directory exists
        patt_noise_dir = self.patt_noise_dir
        if patt_noise_dir.is_dir():
            # if so then find the files with the year/doy combo that we need for this obs_id
            if len(sorted(patt_noise_dir.glob(f"*_{obs_year}{obs_doy}*"))) > 0:
                # these should be the files names
                patt_map_name = patt_noise_dir.joinpath(
                    f"pattern_noise_survey8a_{obs_year}{obs_doy}.dpi"
                )
                patt_mask_name = patt_noise_dir.joinpath(
                    f"pattern_noise_survey8a_{obs_year}{obs_doy}_inbands.detmask"
                )

                # make sure that the files exist
                if patt_map_name.is_file() and patt_mask_name.is_file():
                    patt_map_name = str(patt_map_name)
                    patt_mask_name = str(patt_mask_name)
                else:
                    # if the files dont exist then set these values to None
                    patt_map_name = "NONE"
                    patt_mask_name = "NONE"
            else:
                # if that file doesnt exist then search for file with nearest year/doy stamp
                # get allthe filenames and the years/days associated with them
                all_patt_map = sorted(patt_noise_dir.glob("*.dpi"))
                all_patt_mask = sorted(patt_noise_dir.glob("*_inbands.detmask"))

                years = [i.stem.split("_")[-1][:4] for i in all_patt_map]
                days = [i.stem.split("_")[-1][4:] for i in all_patt_map]

                # turn them into numpy dates
                patt_dates = np.array(
                    [
                        np.datetime64(datetime(int(i), 1, 1) + timedelta(int(j) - 1))
                        for i, j in zip(years, days)
                    ]
                )

                # find the date closest to the time of the observation start time
                idx = np.abs(time - patt_dates).argmin()

                # save the name
                patt_map_name = str(all_patt_map[idx])
                patt_mask_name = str(all_patt_mask[idx])

        else:
            # if the directory doesnt exist then set these values to None
            patt_map_name = "NONE"
            patt_mask_name = "NONE"
        return patt_map_name, patt_mask_name

    def _get_survey_parameters(self):
        """Helper function to get the survey parameters that will be passed to batsurvey
        based on the input dictionary provided by the user and the default values for
        these parameters. If the user does not provide a value for a parameter, then
        the default value will be used. If the user provides a value of None for a
        parameter, then the default value will be used. If the user provides a value
        for a parameter, then that value will be used.

        Raises:
            ValueError: _description_
        """
        batsurvey = hsp_core.HSPTask("batsurvey")
        # get the default names of the parameters for batsurvey including its name (which should never change)
        default_batsurvey_input_dict = batsurvey.default_params.copy()

        patt_map_name, patt_mask_name = self._get_pattern_noise_maps()
        input_dict = self.survey_input
        if input_dict is None:
            input_dict_copy = dict(
                indir=str(self.obs_dir),
                outdir=str(self.obs_dir.parent / f"{self.obs_dir.name}_surveyresult"),
            )

            input_dict_copy["incatalog"] = str(
                Path(__file__).parent.joinpath("data/survey6b_2.cat")
            )
            input_dict_copy["detthresh"] = "10000"
            input_dict_copy["detthresh2"] = "10000"

            input_dict_copy["global_pattern_map"] = patt_map_name
            input_dict_copy["global_pattern_mask"] = patt_mask_name
            input_dict_copy["cleansnr"] = 6
            input_dict_copy["cleanexpr"] = "ALWAYS_CLEAN==T"
        else:
            # need to create copy of input dict so we dont overwrite it
            input_dict_copy = input_dict.copy()
            # see if the user wanted the indir and outdir to be the defaults presented above, even though they
            # specify other preferences to the call to batsurvey
            if (
                "indir" not in input_dict_copy
                or str(input_dict_copy.get("indir", "NONE")).upper() != "NONE"
            ):
                input_dict_copy["indir"] = str(self.obs_dir)
            else:
                # make this a fully resolved path
                if not Path(input_dict_copy["indir"]).is_absolute():
                    input_dict_copy["indir"] = str(
                        Path.cwd().joinpath(input_dict_copy["indir"])
                    )

            if (
                "outdir" not in input_dict_copy
                or str(input_dict_copy.get("outdir", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["outdir"] = str(
                    self.obs_dir.parent / f"{self.obs_dir.name}_surveyresult"
                )
            else:
                # make this a fully resolved path
                if not Path(input_dict_copy["outdir"]).is_absolute():
                    input_dict_copy["outdir"] = str(
                        Path.cwd().joinpath(input_dict_copy["outdir"])
                    )

            # if detthresh/detthresh2 isnt defined need to set default detthresh to prevent gti identification
            # errors
            if (
                "detthresh" not in input_dict_copy
                or str(input_dict_copy.get("detthresh", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["detthresh"] = "10000"

            if (
                "detthresh2" not in input_dict_copy
                or str(input_dict_copy.get("detthresh2", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["detthresh2"] = "10000"

            if ("incatalog" not in input_dict_copy) or (input_dict_copy["incatalog"] is None):
                # If the user choses to provide no input catalog,
                # then use the default catalog
                input_dict_copy["incatalog"] = str(
                    Path(__file__).parent.joinpath("data/survey6b_2.cat")
                )

            if (
                "global_pattern_map" not in input_dict_copy
                or str(input_dict_copy.get("global_pattern_map", "NONE")).upper()
                == "NONE"
            ):
                input_dict_copy["global_pattern_map"] = str(patt_map_name)

            if (
                "global_pattern_mask" not in input_dict_copy
                or str(input_dict_copy.get("global_pattern_mask", "NONE")).upper()
                == "NONE"
            ):
                input_dict_copy["global_pattern_mask"] = str(patt_mask_name)

            if (
                "cleansnr" not in input_dict_copy
                or str(input_dict_copy.get("cleansnr", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["cleansnr"] = 6

            if "cleanexpr" not in input_dict_copy:
                # If the user choses to provide no clean expression,
                # then respect the choice and do not provide any clean expression to batsurvey
                if input_dict_copy["incatalog"] != "NONE":
                    input_dict_copy["cleanexpr"] = "ALWAYS_CLEAN==T"
                else:
                    input_dict_copy["cleanexpr"] = "NONE"
            else:
                if input_dict_copy["incatalog"] == "NONE":
                    # Without an input catalog, this makes no sense
                    input_dict_copy["cleanexpr"] = "NONE"

            # make sure that the output directory exists
            if not Path(input_dict_copy["outdir"]).parent.exists():
                raise ValueError(
                    "The directory %s needs to exist for batsurvey to save its results."
                    % (os.path.split(input_dict_copy["outdir"])[0])
                )

        #overwrite the defaults in the default_batsurvey_input_dict with what has been determined here
        for key, value in input_dict_copy.items():
            default_batsurvey_input_dict[key] = value

        self.survey_input = default_batsurvey_input_dict

    def _identify_truncated_files(self):
        """
        Whenever there is a data pile up issue, BAT records the data in reduced number
         of energy bins, leading to truncated files. This function identifies such files and patches them

        Returns:
            boolean: True if there are truncated DPH files otherwise false.
        """
        dph_files = self.obs_dir.joinpath("bat/survey").glob("*.dph*")
        mask = [True if "e20.dph.gz" in i.name else False for i in dph_files]
        has_truncated_files=np.any(mask)

        if has_truncated_files:
            make_files_consistent(self.obs_dir)

        return has_truncated_files

    def _call_batsurvey(self, input_dict, use_independent_modules=False):
        """
        Calls heasoftpy's batsurvey with an error wrapper
        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batsurvey
        :return: heasoftpy Result object from batsurvey
        """
        # directly calls batsurvey
        # See if the user wants to call the batsurvey module, or a custom
        # implementation of the batsurvey calculation using the independent modules.
        # The latter is useful for debugging and to bypass certain steps
        # of the batsurvey calculation if the user wants to. It is especially preferred
        # when working with truncated data
        if use_independent_modules:
            # Then import the BatTools module and run it using that
            try:
                batsurvey_return = BatTools(
                    indir=input_dict["indir"],
                    outdir=input_dict["outdir"],
                    params=input_dict,
                    truncated=self.truncated,
                )
                if batsurvey_return.success:
                    print("Successfully reduced the survey data.")

                # The backend will cause pickle to fail, so set it to none before starting
                batsurvey_return.backend = None
            except Exception as e:
                raise ValueError(
                    f"An error occurred while running the independent modules of batsurvey: {str(e)}"
                )
            # This will handle error warnings and will also patch the results if there is truncated data
        else:
            try:
                batsurvey_return=hsp.batsurvey(**input_dict)
            except Exception:
                # see if there were any pointings that failed
                status_file = self.result_dir.joinpath("stats_point.dat")
                if status_file.exists():
                    with open(status_file, "r") as f:
                        batsurvey_output = f.readlines()
                    raise ValueError(
                        f"The results for each pointing of observation ID {self.obs_id} is:\n {' '.join(batsurvey_output)}"
                    )
                else:
                    raise ValueError(f"Obsid {self.obs_id} has no survey data")

        # Then fix the stats file for any truncated data
        patch_truncated_obsid(obsid_dir=str(self.result_dir))

        return batsurvey_return

    def _add_total_energy_image(self):
        """
        batsurvey in general only makes images for non-overlapping energy bands, so when all 8 individual
        bands are made, a total energy band image is missing, we nned to estimate that. So take all the images
        for a given OBSID which can have multiple pointings and change them. This changes only the images,
        generated for the specified number of cleaning iterations.
        """
        all_cats = [i.as_posix() for i in self.pointing_flux_files]
        all_images = [i.replace(f"_{self.survey_input['ncleaniter']}.cat", f"_{self.survey_input['ncleaniter']}.img") for i in all_cats]
        all_vars = [i.replace(f"_{self.survey_input['ncleaniter']}.cat", f"_{self.survey_input['ncleaniter']}.var") for i in all_cats]

        # Now work on those images and variation maps
        for ind in range(len(all_images)):
            img_file = all_images[ind]
            var_file = all_vars[ind]

            # Open image and var file
            img_hdu = fits.open(img_file, mode="update")
            var_hdu = fits.open(var_file, mode="update")

            # Get data

            tot_en_image = np.zeros(img_hdu[0].data.shape)
            tot_en_var = np.zeros_like(tot_en_image)

            sel_hdus = [
                (
                    True
                    if ("PRIMARY" in i.name)
                    | (("BAT_IMAGE" in i.name) & ("TOT" not in i.name))
                    & (not i.header.get("TRUNCATED", False))
                    else False
                )
                for i in img_hdu
            ]
            all_hdus = [
                (
                    True
                    if ("PRIMARY" in i.name)
                    | (("BAT_IMAGE" in i.name) & ("TOT" not in i.name))
                    else False
                )
                for i in img_hdu
            ]

            sel_ebins = []
            emin = 250
            emax = 0
            for i in range(len(img_hdu)):
                if sel_hdus[i]:
                    tot_en_image += img_hdu[i].data
                    tot_en_var += var_hdu[i].data ** 2
                    sel_ebins.append(
                        f"{img_hdu[i].header['E_MIN']}-{img_hdu[i].header['E_MAX']}"
                    )
                    emin = min(emin, img_hdu[i].header["E_MIN"])
                    emax = max(emax, img_hdu[i].header["E_MAX"])

            sel_ebins = ",".join(sel_ebins) + " keV"
            tot_en_var = np.sqrt(tot_en_var)

            # Copy a dummy header from first extension and change the name
            hdr = img_hdu[0].header.copy()
            hdr["EXTNAME"] = "BAT_IMAGE_TOT"
            hdr["HDUNAME"] = "BAT_IMAGE_TOT"
            hdr["E_MIN"] = (14, "Minimum energy of the total energy band (keV)")
            hdr["E_MAX"] = (195, "Maximum energy of the total energy band (keV)")
            hdr["EBINS"] = (
                sel_ebins,
                "Energy bins summed to make this total energy image",
            )
            tot_en_img_hdu = fits.ImageHDU(data=tot_en_image, header=hdr)

            hdr = var_hdu[0].header.copy()
            hdr["EXTNAME"] = "BAT_VAR_TOT"
            hdr["HDUNAME"] = "BAT_VAR_TOT"
            hdr["E_MIN"] = (14, "Minimum energy of the total energy band (keV)")
            hdr["E_MAX"] = (195, "Maximum energy of the total energy band (keV)")
            hdr["EBINS"] = (
                sel_ebins,
                "Energy bins summed to make this total energy variance map",
            )
            tot_en_var_hdu = fits.ImageHDU(data=tot_en_var, header=hdr)

            # Now need to add these to the fits files
            hdu_ind = np.where(all_hdus)[0][-1] + 1
            img_hdu.insert(hdu_ind, tot_en_img_hdu)
            var_hdu.insert(hdu_ind, tot_en_var_hdu)

            # Now write back to the files
            img_hdu.flush()
            img_hdu.close()
            var_hdu.flush()
            var_hdu.close()

    def _filter_duplicate_sources(self, tab, sep=5 * u.arcmin):
        coords = SkyCoord(
            ra=tab["RA_OBJ"], dec=tab["DEC_OBJ"], frame="icrs", unit="deg"
        )

        # Convert to 3D cartesian (unit sphere)
        xyz = np.vstack(
            [
                coords.cartesian.x.value,
                coords.cartesian.y.value,
                coords.cartesian.z.value,
            ]
        ).T

        # Run DBSCAN: eps in radians
        cluster = DBSCAN(
            eps=(sep).to(u.rad).value, min_samples=1, metric="euclidean"
        ).fit(xyz)

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

    def _detect_sources(self, input_dict=None):
        """Helper function to call batcelldetect and detect sources in all energy bands

        Args:
            input_dict (dict): input parameters for batcelldetect
        """
        #TODO: the BatMosaic object also has a detect sources method that can be merged with this one or can call this one

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))


        # Then one needs to run batcelldetect on these new files to get the source information
        # Add in a dictionary of inputs

        all_cats = [i.as_posix() for i in self.pointing_flux_files]
        all_pointing_ids = self.pointing_ids

        all_images = [i.replace(f"_{self.survey_input['ncleaniter']}.cat", f"_{self.survey_input['ncleaniter']}.img") for i in all_cats]
        self.source_catalogs = {}
        self.bat_source_catalog = Path(__file__).parent.joinpath("data/survey6b_2.cat")
        batcelldetect_output = []

        input_dict_template = dict(
            # incatalog=str(self.bat_source_catalog.as_posix()),
            snrthresh=4.5,
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
            srcradius=4,
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

        if input_dict is not None:
            # Then overwrite whatever user gives
            for key in input_dict.keys():
                input_dict_template[key] = input_dict[key]

        # I observed that even when using posfit=NO, if I give it a posfitwindow value
        # it still tries to do the position fitting. This is because it overrides
        # the posfit value to YES if posfitwindow is given. Therefore, to
        # prevent this, one can take two approaches: Here we do the following

        # Only detect significant sources (snr>=5) using batcelldetect, once a
        # source is detected in the detection step, then use this catalog to do
        # flux fitting with position fixed to MAX_SNR location

        for ind in range(len(all_images)):
            img_file = all_images[ind]
            batcelldetect_input_dict = input_dict_template.copy()
            # # Then we need the partial coding file and threshold
            # # But we need to extract it from the image file
            pcode_file = f"{img_file}[BAT_PCODE_1]"
            # batcelldetect_input_dict["pcodethresh"] = 0.05
            # batcelldetect_input_dict["bkgpcodethresh"] = 0.025
            # batcelldetect_input_dict["outfile"] = img_file.replace(
            #     ".img", "_src_catalog.fits"
            # )
            # batcelldetect_input_dict["clobber"] = "YES"
            # batcelldetect_input_dict["rows"] = "1-9"

            # Now analyze them
            # First do independent source detection on all energy bins
            print(f"Running batcelldetect on {img_file} and all energy bins\n")
            batcelldetect_input_dict["infile"] = img_file
            batcelldetect_input_dict["incatalog"] = ""
            batcelldetect_input_dict["vectorflux"] = "NO"
            batcelldetect_input_dict["carryover"] = "NO"
            batcelldetect_input_dict["pcodefile"] = pcode_file
            batcelldetect_input_dict["pcodethresh"] = 0.05
            batcelldetect_input_dict["bkgvarmap"] = img_file.replace(".img", ".var")
            batcelldetect_input_dict["bkgpcodethresh"] = 0.05
            batcelldetect_input_dict["outfile"] = img_file.replace(
                ".img", "_src_catalog.fits"
            )
            # batcelldetect_input_dict["signifmap"] = snr_map
            batcelldetect_input_dict["clobber"] = "YES"

            # We need to decide which rows to run on.
            # batcelldetect_input_dict["rows"] = ",".join(good_rows)
            if self._add_total_energy_image:
                batcelldetect_input_dict["rows"] = "1-9"
            else:
                batcelldetect_input_dict["rows"] = "1-8"

            batcelldetect_output.append(
                self._call_batcelldetect(batcelldetect_input_dict)
            )

            source_hdu = fits.open(batcelldetect_input_dict["outfile"], mode="update")
            source_data = Table(source_hdu[1].data)
            if len(source_data) > 0:
                # Now select unique sources
                unique_sources = self._filter_duplicate_sources(source_data)

                source_hdu[1] = fits.BinTableHDU(
                    unique_sources,
                    header=source_hdu[1].header,
                )
                source_hdu.flush()
                source_hdu.close()

                # Only run if there are any sources detected
                print(f"Running batcelldetect on {img_file} for flux estimation\n")
                batcelldetect_input_dict["srcdetect"] = "NO"
                batcelldetect_input_dict["vectorflux"] = "YES"
                batcelldetect_input_dict["carryover"] = "YES"
                batcelldetect_input_dict["posfit"] = "NO"
                batcelldetect_input_dict["pospeaks"] = "YES"
                batcelldetect_input_dict["posfluxfit"] = "NO"
                batcelldetect_input_dict["posfitwindow"] = 0.0
                batcelldetect_input_dict["bkgvarmap"] = "NONE"
                batcelldetect_input_dict["incatalog"] = batcelldetect_input_dict[
                    "outfile"
                ]
                batcelldetect_input_dict["outfile"] = img_file.replace(
                    ".img", "_src_catalog.fits.tmp"
                )
                print(batcelldetect_input_dict)

                batcelldetect_output.append(
                    self._call_batcelldetect(batcelldetect_input_dict)
                )
                # Now use this to run a second iteration of batcelldetect to get the fluxes

                # Now add additional info to the catalog
                add_additional_info(batcelldetect_input_dict["outfile"])
                self.source_catalogs[all_pointing_ids[ind]] = batcelldetect_input_dict[
                    "outfile"
                ].replace(".tmp", "")

                ###########################################################################
            else:
                source_hdu.close()
                # empty_source_catalogs.append(catalog_file)
                self.source_catalogs[all_pointing_ids[ind]] = batcelldetect_input_dict[
                    "outfile"
                ]
                print(f"No sources detected in {img_file}, skipping further analysis\n")
        self.batcelldetect_output = batcelldetect_output

    def _get_unknown_sources(self, new_source_sep=10.0 * u.arcmin, snr_threshold=5, pcoding_threshold=0.05):
        """Function to cross match detected sources with known BAT persistent sources and identify unknown sources.

        Args:
            new_source_sep (Quantity): Minimum separation from known BAT persistent sources to consider a source as
                unknown.
        """
        persistent_sources = Table.read(self.bat_source_catalog, format="fits")
        bat_source_coords = SkyCoord(
            ra=persistent_sources["RA_OBJ"],
            dec=persistent_sources["DEC_OBJ"],
            unit="deg",
            frame="icrs",
        )

        self.unknown_sources_catalogs = []
        self.unknown_sources_ind = {}
        self.unknown_sources = Table()
        for pointing_id in self.source_catalogs:
            source_data = Table.read(self.source_catalogs[pointing_id], format="fits")
            if len(source_data) > 0:
                # First filter sources with SNR>=5 in any energy band
                filtered_sources = source_data[np.max(source_data["SNR"], axis=1) >= snr_threshold]

                # Now select and remove BAT persistent sources

                coords = SkyCoord(
                    ra=filtered_sources["RA_OBJ"],
                    dec=filtered_sources["DEC_OBJ"],
                    unit="deg",
                    frame="icrs",
                )
                _, sep, _ = coords.match_to_catalog_sky(bat_source_coords)
                sep_mask = sep.arcmin >= new_source_sep.to(u.arcmin).value  # PSF of BAT

                # select unknown sources
                unknown_sources = filtered_sources[sep_mask]

                # First of all rename the sources
                src_names = coords[sep_mask].to_string(
                    "hmsdms", fields=3, sep="", precision=0
                )
                src_names = ["J" + name.replace(" ", "") for name in src_names]
                unknown_sources.replace_column("NAME", src_names)

                unknown_sources.add_column(
                    np.max(unknown_sources["SNR"], axis=1), name="MAX_SNR"
                )
                # Also add the index of the energy band where the max SNR occurs
                unknown_sources.add_column(
                    np.argmax(unknown_sources["SNR"], axis=1), name="MAX_SNR_ENBIN"
                )
                unknown_sources.add_column(sep[sep_mask].arcmin, name="BAT_SRC_SEP")

                # Now we need to get the effective SNR for these sources
                # To do that first read SNR images
                eff_snr = []
                nfp = []

                for i in range(len(unknown_sources)):
                    imdata = fits.getdata(
                        self.source_catalogs[pointing_id].replace(
                            "_src_catalog.fits", ".img"
                        ),
                        unknown_sources["MAX_SNR_ENBIN"][i],
                    )
                    vardata = fits.getdata(
                        self.source_catalogs[pointing_id].replace(
                            "_src_catalog.fits", ".var"
                        ),
                        unknown_sources["MAX_SNR_ENBIN"][i],
                    )

                    # Also get the pcoding mask
                    # pcoding_mask = fits.getdata(
                    #     self.source_catalogs[pointing_id].replace(
                    #         "_src_catalog.fits", ".pcodemap"
                    #     )
                    # )
                    pcoding_mask = fits.getdata(
                        self.source_catalogs[pointing_id].replace(
                            "_src_catalog.fits", ".img"
                        ),
                        "BAT_PCODE_1",
                    )
                    mask = pcoding_mask > pcoding_threshold
                    esnr, nfp_val = calculate_effective_snr(
                        img=imdata,
                        var=vardata,
                        snr=unknown_sources["MAX_SNR"][i],
                        mask=mask,
                    )
                    if np.isnan(esnr):
                        warnings.warn(
                            f"Effective SNR could not be calculated for source {unknown_sources['NAME'][i]} in pointing {pointing_id}, obs_id {self.obs_id}."
                        )
                    eff_snr.append(esnr)
                    nfp.append(nfp_val)
                unknown_sources.add_column(eff_snr, name="ESNR")
                unknown_sources.add_column(nfp, name="NFP")

                # Also add info about obs_id and pointing_id
                unknown_sources.add_column(
                    [self.obs_id] * len(unknown_sources), name="OBS_ID"
                )
                unknown_sources.add_column(
                    [pointing_id] * len(unknown_sources), name="POINTING_ID"
                )

                # Write it to a file
                unknown_source_file = self.source_catalogs[pointing_id].replace(
                    "_src_catalog.fits", "_unknown_src_catalog.fits"
                )

                unknown_sources.write(
                    unknown_source_file,
                    format="fits",
                    overwrite=True,
                )

                self.unknown_sources_ind[unknown_source_file] = unknown_sources
                if len(unknown_sources) > 0:
                    self.unknown_sources = vstack(
                        [self.unknown_sources, unknown_sources], join_type="outer"
                    )
                self.unknown_sources_catalogs.append(unknown_source_file)

    def _call_batcelldetect(self, input_dict):
        """
        Call heasoftpy batcelldetect.

        :param input_dict: dictionary of inputs to pass to batcelldetet.
        :return: heasoft output object
        """
        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        out = hsp.batcelldetect(**input_dict)

        return out


    def _batsurvey_error(self, status_file):
        """
        Prints the error for a pointing ID's status file. the file needs to exist before calling this method.
        :param status_file: Path file that is the pointing_*_status.txt file that will be read in
        :return: Boolean denoting if the batsurvey call wasy successful for that pointing ID and a string of the
            associated reason for failure
        """

        with open(status_file, "r") as f:
            output = f.readlines()[0]
        stat = output.split(";")[0].split("status=")[-1]
        stat = stat.replace("'", "")
        stat = stat.replace('"', "")
        if "SUCCESS" in stat:
            stat = True
        else:
            stat = False

        # read in the failure reason
        fail_str = output.split(";")[2].split("reason=")[-1]
        fail_str = fail_str.replace("'", "")
        fail_str = fail_str.replace('"', "")

        return stat, fail_str

    def _call_batsurvey_catmux(self, input_dict):
        """
        Calls heasoftpy's batsurvey_catmux
        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batsurvey-catmux
        :return: None
        """
        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        # calls batsurvey_catmux to merge pointings, outputs to the survey result directory
        # there is a bug in the heasoftpy code so try to explicitly call it for now
        return hsp.batsurvey_catmux(**input_dict)

    def merge_pointings(self, input_dict=None, verbose=False):
        """
        Merges the data for each pointing within a given observation ID into a single file
        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batsurvey-catmux
        :param verbose: Boolean to print diagnostic information
        :return: None
        """

        # see if the directory is specified in the input values, if so then use default outdir
        if input_dict is None or "outfile" not in input_dict:
            output_dir = self.result_dir.joinpath("merged_pointings_lc")
        else:
            output_dir = (
                Path(input_dict["outfile"]).expanduser().resolve()
            )  # else use what was provided

        # see if directory exists, if no create of so then delete and recreate
        dirtest(output_dir)

        if verbose:
            if "keycolumn" in input_dict:
                print("Merging based on %s" % (input_dict["keycolumn"]))
            else:
                print("Merging based on NAME")

        for i in self.pointing_flux_files:
            if input_dict is None:
                dictionary = dict(
                    keycolumn="NAME",
                    infile=str(i),
                    outfile=str(output_dir.joinpath("%s.cat")),
                )
            else:
                dictionary = input_dict.copy()
                dictionary["infile"] = str(i)

            self._call_batsurvey_catmux(dictionary)

            self.merge_input = dictionary

    def calculate_pha(
        self,
        id_list,
        output_dir=None,
        calc_upper_lim=False,
        bkg_nsigma=None,
        verbose=True,
        clean_dir=False,
        single_pointing=None,
    ):
        """
        This function calculates the pha files for each object in the input catalog file by default. Can specify
        'keycolumn' value for specific objects. Based on make_spectrum.py by Taka and Amy

        :param id_list: A string or a List of Strings or None Denoting which sources the user wants the PHA files calculated for.
            None is reserved for when the user wants to calculate PHA files for all the sources in the catalog which includes those
            from the default bat survey catalog
        :param output_dir: None or a string where the output PHA file should be saved
        :param calc_upper_lim: Boolean to denote if the PHA file should be constructed to calculate upper limits for an
            object. This is done by using the bkg_var instead of the cent_rate for the count information in the
            produced PHA file
        :param bkg_nsigma: Float for the significance of the background scaling to obtain an upper limit at that limit
            (eg PHA count = bkg_nsigma*bkg_var)
        :param verbose: Boolean to print diagnostic information
        :param clean_dir: Boolean to denote if the resulting PHA_files/ directory should be removed and recreated
        :param single_pointing: None or a string with a pointing ID that corresponds to creating the PHA file at the
            specified pointing ID file
        :return: None
        """
        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        if calc_upper_lim and bkg_nsigma is None:
            raise ValueError(
                "A value for bkg_nsigma has not been passed to the function to calculate upper limits."
            )

        if output_dir is None:
            # set default directory to save files into
            output_dir = self.result_dir.joinpath("PHA_files")
        else:
            output_dir = Path(output_dir).expanduser().resolve()

        if not calc_upper_lim:
            # see if directory exists, if no create of so then delete and recreate
            # if we are calculating the upper limit, the directory already exists
            dirtest(output_dir, clean_dir=clean_dir)

        merge_output_path = Path(self.merge_input["outfile"]).parent

        # if something has been passed in make sure that its a list
        if id_list is not None:
            if type(id_list) is not list:
                # it is a single string:
                id_list = [id_list]
        else:
            # use the ids from the *.cat files produced, these are ones that have been identified in the survey obs_id
            x = sorted(merge_output_path.glob("*.cat"))
            id_list = [i.stem for i in x]

        # for each object/source of interest
        # reset the save the pha file names of all pha files if necessary
        if clean_dir:
            self.set_pha_filenames("", reset=True)
        for ident in id_list:
            if verbose:
                print("Creating PHA file for ", ident)

            # get the proper name for the source incase the user didnt get the name correct
            x = sorted(merge_output_path.glob("*.cat"))
            catalog_sources = [i.stem for i in x]
            test = self._compare_source_name(ident, catalog_sources)
            if np.sum(test) > 0:
                ident = np.array(catalog_sources)[
                    self._compare_source_name(ident, catalog_sources)
                ][0]
            else:
                ident = None

            # get info from the newly created cat file (from merge)
            catalog = merge_output_path.joinpath(f"{ident}.cat")
            try:
                cat_file = fits.open(str(catalog))
                tbdata = cat_file[1].data
                name_array = tbdata.field("NAME")
                raobj_array = tbdata.field("RA_OBJ")
                decobj_array = tbdata.field("DEC_OBJ")
                time_array = tbdata.field("TIME")
                tstart_sinceT0 = np.zeros_like(time_array)  # need to understand this
                # timestop_array = tbdata.field("TIME_STOP") #this isnt used
                exposure_array = tbdata.field("EXPOSURE")
                ffapp_array = tbdata.field("FFAPP")
                pcodeapp_array = tbdata.field("PCODEAPP")
                pcodefr_array = tbdata.field("PCODEFR")
                ngpixapp_array = tbdata.field("NGPIXAPP")
                ngoodpix_array = tbdata.field("NGOODPIX")
                pointing_array = tbdata.field("IMAGE_ID")

                if calc_upper_lim:
                    count_rate_array = tbdata.field("BKG_VAR")
                    count_rate_err_array = np.zeros_like(count_rate_array)
                    scale = bkg_nsigma
                else:
                    count_rate_array = tbdata.field("CENT_RATE")
                    count_rate_err_array = tbdata.field("BKG_VAR")
                    scale = 1

                # bkg = tbdata.field('BKG')
                # bkg_err = tbdata.field('BKG_ERR')
                # bkg_var = tbdata.field('BKG_VAR')
                theta_array = tbdata.field("THETA")
                phi_array = tbdata.field("PHI")
                cat_file.close()

                # if we want to calculate or recalculate pha for certain pointings we modify the arrays to just the
                # values of interest
                if single_pointing is not None:
                    # expect a pointing ID which we will match up to the pointing_array
                    idx = np.where("point_" + single_pointing == pointing_array)[0]
                    if np.size(idx) == 0:
                        raise ValueError(
                            "The pointing ID does not contain an observation of the source:",
                            ident,
                        )
                    name_array = name_array[idx]
                    raobj_array = raobj_array[idx]
                    decobj_array = decobj_array[idx]
                    time_array = time_array[idx]
                    tstart_sinceT0 = tstart_sinceT0[idx]  # need to understand this
                    # timestop_array = timestop_array[idx] #this isnt used
                    exposure_array = exposure_array[idx]
                    ffapp_array = ffapp_array[idx]
                    pcodeapp_array = pcodeapp_array[idx]
                    pcodefr_array = pcodefr_array[idx]
                    ngpixapp_array = ngpixapp_array[idx]
                    ngoodpix_array = ngoodpix_array[idx]
                    pointing_array = pointing_array[idx]
                    count_rate_array = count_rate_array[idx]  # [:]
                    count_rate_err_array = count_rate_err_array[idx]  # [:]
                    theta_array = theta_array[idx]
                    phi_array = phi_array[idx]

                # make pha file at the specified times
                # Looping over different pointings for a given observation.
                for i in range(len(time_array)):
                    # These are to ensure that we are starting fresh with our T_start and T_stop, and not
                    # appending them.
                    count_rate_band = []
                    count_rate_band_error = []
                    channel = []
                    gti_starttime = []
                    gti_stoptime = []

                    check = 0
                    # find the time in the light curve cat file
                    if (
                        (time_array[i] + tstart_sinceT0[i])
                        <= time_array[i]
                        < (time_array[i] + tstart_sinceT0[i] + exposure_array[i])
                    ):
                        check += 1
                        gti_starttime.append(time_array[i])
                        gti_stoptime.append(time_array[i] + exposure_array[i])
                        if verbose:
                            print("Time interval:", gti_starttime, gti_stoptime)
                        for i_band in range(len(count_rate_array[i])):
                            channel.append(i_band + 1)
                            # print(i_band, count_rate_array[i][i_band])
                            count_rate_band.append(scale * count_rate_array[i][i_band])
                            count_rate_band_error.append(
                                count_rate_err_array[i][i_band]
                            )

                        # get some info from the original cat file where all rows are the same value
                        org_catfile_name = self.result_dir.joinpath(
                            f"{pointing_array[i]}"
                        ).joinpath(
                            f"{pointing_array[i]}_{self.batsurvey_result.params['ncleaniter']}.cat"
                        )

                        org_cat_file = fits.open(str(org_catfile_name))
                        org_cat_data = org_cat_file[1].data
                        ra_pnt = org_cat_data.field("RA_PNT")[0]
                        dec_pnt = org_cat_data.field("DEC_PNT")[0]
                        pa_pnt = org_cat_data.field("PA_PNT")[0]
                        org_cat_file.close()

                        attfile = self.result_dir.joinpath(
                            f"{pointing_array[i]}"
                        ).joinpath(f"{pointing_array[i]}.att")
                        dpifile = self.result_dir.joinpath(
                            f"{pointing_array[i]}"
                        ).joinpath(f"{pointing_array[i]}_1.dpi")
                        detmask = self.result_dir.joinpath(
                            f"{pointing_array[i]}"
                        ).joinpath(f"{pointing_array[i]}.detmask")
                        output_srcmask = output_dir.joinpath("src.mask")

                        input_dict = dict(
                            outfile=str(output_srcmask),
                            attitude=str(attfile),
                            ra=str(raobj_array[i]),
                            dec=str(decobj_array[i]),
                            infile=str(dpifile),
                            detmask=str(detmask),
                            clobber="YES",
                        )

                        result = hsp.batmaskwtimg(**input_dict)

                        with fits.open(str(output_srcmask)) as file:
                            mskwtsqf = file[0].header["MSKWTSQF"]

                        # write count_rate in each band to a pha file
                        spec_col1 = fits.Column(
                            name="CHANNEL", format="J", array=channel
                        )
                        spec_col2 = fits.Column(
                            name="RATE",
                            format="E",
                            unit="count/s",
                            array=count_rate_band,
                        )
                        spec_col3 = fits.Column(
                            name="STAT_ERR",
                            format="E",
                            unit="count/s",
                            array=count_rate_band_error,
                        )
                        spec_col4 = fits.Column(
                            name="SYS_ERR", format="D", unit="", array=self.syserr
                        )

                        ebound_col1 = fits.Column(
                            name="CHANNEL", format="1I", unit="", array=channel
                        )
                        ebound_col2 = fits.Column(
                            name="E_MIN", format="1E", unit="keV", array=self.emin
                        )
                        ebound_col3 = fits.Column(
                            name="E_MAX", format="1E", unit="keV", array=self.emax
                        )

                        gti_col1 = fits.Column(
                            name="START", format="1D", unit="s", array=gti_starttime
                        )
                        gti_col2 = fits.Column(
                            name="STOP", format="1D", unit="s", array=gti_stoptime
                        )

                        spec_cols = fits.ColDefs(
                            [spec_col1, spec_col2, spec_col3, spec_col4]
                        )
                        ebound_cols = fits.ColDefs(
                            [ebound_col1, ebound_col2, ebound_col3]
                        )
                        gti_cols = fits.ColDefs([gti_col1, gti_col2])

                        spec_tbhdu = fits.BinTableHDU.from_columns(spec_cols)
                        ebound_tbhdu = fits.BinTableHDU.from_columns(ebound_cols)
                        gti_tbhdu = fits.BinTableHDU.from_columns(gti_cols)

                        spec_tbhdu.name = "SPECTRUM"
                        ebound_tbhdu.name = "EBOUNDS"
                        gti_tbhdu.name = "STDGTI"

                        pha_primary = fits.PrimaryHDU()

                        pha_thdulist = fits.HDUList(
                            [pha_primary, spec_tbhdu, ebound_tbhdu, gti_tbhdu]
                        )

                        if calc_upper_lim:
                            survey_pha_file = output_dir.joinpath(
                                f"{ident}_survey_{pointing_array[i]}_bkgnsigma_{int(bkg_nsigma)}_upperlim.pha"
                            )
                        else:
                            survey_pha_file = output_dir.joinpath(
                                f"{ident}_survey_{pointing_array[i]}.pha"
                            )
                        self.set_pha_filenames(survey_pha_file)
                        pha_thdulist.writeto(str(survey_pha_file))

                        pha_hdulist = fits.open(str(survey_pha_file), mode="update")

                        pha_prime_hdr = pha_hdulist[0].header
                        pha_spec_hdr = pha_hdulist[1].header
                        pha_ebound_hdr = pha_hdulist[2].header
                        # pha_gti_hdr = pha_hdulist[3].header this header is not currenlty used

                        pha_prime_hdr["TELESCOP"] = (
                            "SWIFT",
                            "Telescope (mission) name",
                        )
                        pha_prime_hdr["INSTRUME"] = ("BAT", "Instrument name")
                        pha_prime_hdr["TIMESYS"] = ("TT", " Time system")
                        pha_prime_hdr["MJDREFI"] = (
                            51910.0,
                            " Reference MJD Integer part",
                        )
                        pha_prime_hdr["MJDREFF"] = (
                            0.00074287037,
                            " Reference MJD fractional",
                        )
                        pha_prime_hdr["TIMEREF"] = (
                            "LOCAL",
                            " Time reference (barycenter/local)",
                        )
                        pha_prime_hdr["TASSIGN"] = (
                            "SATELLITE",
                            " Time assigned by clock",
                        )
                        pha_prime_hdr["TIMEUNIT"] = ("s", " Time unit")
                        pha_prime_hdr["TIERRELA"] = (
                            1.0e-8,
                            " [s/s] relative errors expressed as rate",
                        )
                        pha_prime_hdr["TIERABSO"] = (
                            1.0,
                            " [s] timing precision in seconds",
                        )
                        pha_prime_hdr["CLOCKAPP"] = (
                            "F",
                            "Is mission time corrected for clock drift?",
                        )

                        pha_prime_hdr["OBS_ID"] = (self.obs_id, "Observation ID")
                        pha_prime_hdr["OBJECT"] = (name_array[i], "Object name")

                        pha_prime_hdr["EQUINOX"] = (2000.0, " Equinox")
                        pha_prime_hdr["RADECSYS"] = ("FK5", " Coordinate system")

                        pha_prime_hdr["RA_OBJ"] = (raobj_array[i], "[deg] R.A. Object")
                        pha_prime_hdr["DEC_OBJ"] = (decobj_array[i], "[deg] Dec Object")
                        pha_prime_hdr["RA_PNT"] = (ra_pnt, "[deg] RA pointing")
                        pha_prime_hdr["DEC_PNT"] = (dec_pnt, "[deg] Dec pointing")
                        pha_prime_hdr["PA_PNT"] = (
                            pa_pnt,
                            "[deg] Position angle (roll)",
                        )
                        pha_prime_hdr["TSTART"] = (gti_starttime[0], "Start time")
                        pha_prime_hdr["TSTOP"] = (gti_stoptime[0], "Stop time")

                        utc_starttime = met2utc(gti_starttime[0]).astype(
                            "datetime64[s]"
                        )
                        utc_stoptime = met2utc(gti_stoptime[0]).astype("datetime64[s]")
                        pha_prime_hdr["DATE-OBS"] = (
                            f"{utc_starttime}",
                            "TSTART, expressed in UTC",
                        )
                        pha_prime_hdr["DATE-END"] = (
                            f"{utc_stoptime}",
                            "TSTOP, expressed in UTC",
                        )

                        pha_spec_hdr["TELESCOP"] = ("SWIFT", "Telescope (mission) name")
                        pha_spec_hdr["INSTRUME"] = ("BAT", "Instrument name")
                        pha_spec_hdr["TIMESYS"] = ("TT", " Time system")
                        pha_spec_hdr["MJDREFI"] = (
                            51910.0,
                            " Reference MJD Integer part",
                        )
                        pha_spec_hdr["MJDREFF"] = (
                            0.00074287037,
                            " Reference MJD fractional",
                        )
                        pha_spec_hdr["TIMEREF"] = (
                            "LOCAL",
                            " Time reference (barycenter/local)",
                        )
                        pha_spec_hdr["TASSIGN"] = (
                            "SATELLITE",
                            " Time assigned by clock",
                        )
                        pha_spec_hdr["TIMEUNIT"] = ("s", " Time unit")
                        pha_spec_hdr["TIERRELA"] = (
                            1.0e-8,
                            " [s/s] relative errors expressed as rate",
                        )
                        pha_spec_hdr["TIERABSO"] = (
                            1.0,
                            " [s] timing precision in seconds",
                        )
                        pha_spec_hdr["CLOCKAPP"] = (
                            "F",
                            "Is mission time corrected for clock drift?",
                        )
                        pha_spec_hdr["EQUINOX"] = (2000.0, " Equinox")
                        pha_spec_hdr["RADECSYS"] = ("FK5", " Coordinate system")
                        pha_spec_hdr["RA_OBJ"] = (raobj_array[i], "[deg] R.A. Object")
                        pha_spec_hdr["DEC_OBJ"] = (decobj_array[i], "[deg] Dec Object")
                        pha_spec_hdr["RA_PNT"] = (ra_pnt, "[deg] RA pointing")
                        pha_spec_hdr["DEC_PNT"] = (dec_pnt, "[deg] Dec pointing")
                        pha_spec_hdr["PA_PNT"] = (
                            pa_pnt,
                            "[deg] Position angle (roll)",
                        )

                        pha_spec_hdr["HDUCLASS"] = (
                            "OGIP",
                            "Conforms to OGIP/GSFC standards",
                        )
                        pha_spec_hdr["HDUCLAS1"] = ("SPECTRUM", "Contains spectrum")
                        pha_spec_hdr["GAINAPP"] = (
                            "T",
                            "Gain correction has been applied",
                        )
                        pha_spec_hdr["GAINMETH"] = (
                            "FIXEDDAC",
                            "Cubic ground gain/offset correction using DAC-b",
                        )
                        pha_spec_hdr["TELAPSE"] = (
                            exposure_array[i],
                            "[s] Total elapsed time from start to stop",
                        )
                        pha_spec_hdr["ONTIME"] = (
                            exposure_array[i],
                            "[s] Accumulated on-time",
                        )
                        pha_spec_hdr["LIVETIME"] = (
                            exposure_array[i],
                            "[s] ONTIME multiplied by DEADC",
                        )
                        pha_spec_hdr["EXPOSURE"] = (
                            exposure_array[i],
                            "[s] Accumulated exposure",
                        )
                        pha_spec_hdr["BAT_RA"] = (
                            raobj_array[i],
                            "[deg] Right ascension of source",
                        )
                        pha_spec_hdr["BAT_DEC"] = (
                            decobj_array[i],
                            "[deg] Declination of source",
                        )
                        pha_spec_hdr["TSTART"] = (gti_starttime[0], "Start time")
                        pha_spec_hdr["TSTOP"] = (gti_stoptime[0], "Stop time")
                        pha_spec_hdr["AREASCAL"] = (1.0, "Nominal effective area")
                        pha_spec_hdr["BACKSCAL"] = (1.0, "Background scale factor")
                        pha_spec_hdr["CORRSCAL"] = (0.0, "Correction scale factor")
                        pha_spec_hdr["BACKFILE"] = ("none", "Background FITS file")
                        pha_spec_hdr["CORRFILE"] = ("none", "Correction FITS file")
                        pha_spec_hdr["RESPFILE"] = (
                            "none",
                            "Redistribution Matrix file (RMF)",
                        )
                        pha_spec_hdr["ANCRFILE"] = ("none", "Effective Area file (ARF)")
                        pha_spec_hdr["QUALITY"] = (0, "Data quality flag")
                        pha_spec_hdr["GROUPING"] = (0, "Spectra are not grouped")
                        pha_spec_hdr["POISSERR"] = ("F", "Poisson errors do not apply")
                        pha_spec_hdr["SYS_ERR"] = (0.0, "Systematic error value")
                        pha_spec_hdr["DETCHANS"] = (
                            8,
                            "Total number of detector channels available",
                        )
                        pha_spec_hdr["CHANTYPE"] = ("PI", "Pulse height channel type")
                        pha_spec_hdr["HDUCLAS2"] = (
                            "NET",
                            "Spectrum is background subtracted",
                        )
                        pha_spec_hdr["HDUCLAS3"] = ("RATE", "Spectrum is count/s")
                        pha_spec_hdr["PHAVERSN"] = (
                            "1992a",
                            "Version of spectrum format",
                        )
                        pha_spec_hdr["HDUVERS"] = (
                            "1.2.0",
                            "Version of spectrum header",
                        )
                        pha_spec_hdr["FLUXMETH"] = (
                            "WEIGHTED",
                            "Flux extraction method",
                        )

                        phirad = np.deg2rad(phi_array[i])
                        thetarad = np.deg2rad(theta_array[i])
                        batx = np.cos(phirad) * np.sin(thetarad)
                        baty = -np.sin(phirad) * np.sin(thetarad)
                        batz = np.cos(thetarad)

                        pha_spec_hdr["BAT_XOBJ"] = (
                            batx,
                            "[cm] Position of source in BAT_X",
                        )
                        pha_spec_hdr["BAT_YOBJ"] = (
                            baty,
                            "[cm] Position of source in BAT_Y",
                        )
                        pha_spec_hdr["BAT_ZOBJ"] = (
                            batz,
                            "[cm] Position of source in BAT_Z",
                        )
                        pha_spec_hdr["COORTYPE"] = (
                            "sky",
                            "Type of coordinates specified for weighting",
                        )
                        pha_spec_hdr["FFAPP"] = (
                            ffapp_array[i],
                            "Projection correction applied?",
                        )
                        pha_spec_hdr["NFAPP"] = (
                            "F",
                            "Near-field correction applied? ~(COS+RSQ)",
                        )
                        pha_spec_hdr["PCODEAPP"] = (
                            pcodeapp_array[i],
                            "Partial coding correction applied?",
                        )
                        pha_spec_hdr["PCODEFR"] = (
                            pcodefr_array[i],
                            "Partial coding fraction of target",
                        )
                        pha_spec_hdr["NGPIXAPP"] = (
                            ngpixapp_array[i],
                            "Normalized by number of detectors?",
                        )
                        pha_spec_hdr["NGOODPIX"] = (
                            ngoodpix_array[i],
                            "Number of enabled detectors",
                        )
                        pha_spec_hdr["DATE-OBS"] = (
                            f"{utc_starttime}",
                            "TSTART, expressed in UTC",
                        )
                        pha_spec_hdr["DATE-END"] = (
                            f"{utc_stoptime}",
                            "TSTOP, expressed in UTC",
                        )

                        pha_ebound_hdr["HDUCLASS"] = (
                            "OGIP",
                            "Conforms to OGIP/GSFC standards",
                        )
                        pha_ebound_hdr["HDUCLAS1"] = ("RESPONSE", "Contains spectrum")
                        pha_ebound_hdr["GAINAPP"] = (
                            "T",
                            "Gain correction has been applied",
                        )
                        pha_ebound_hdr["GAINMETH"] = (
                            "FIXEDDAC",
                            "Cubic ground gain/offset correction using DAC-b",
                        )

                        pha_spec_hdr["MSKWTSQF"] = (
                            mskwtsqf,
                            "Half-variance of mask weight map",
                        )

                        pha_hdulist.flush()

                    if check > 1:
                        print("check = ", check)
                        print(
                            "Found more than one matched time, please double check the time interval."
                        )
                        print(
                            "This method does not add up the counts for more than one time intervals."
                        )
                        raise RuntimeError(
                            "Found more than one matched time, please double check the time interval.\n"
                            "This method does not add up the counts for more than one time intervals."
                        )
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError(
                    f"This means that the batsurvey script didnt deem there to be good enough statistics for "
                    + f"source {ident} in this observation ID."
                )

    def load_source_information(self, sources):
        """
        This function loads in all the source information for all the sources that have been specified. This can be done
        once batsurvey has been called. It reads in the pointing id cat file and searches for the source within it.

        :param sources: A string or a list of strings with source names that is in the catalog that was passed in to the BatSurvey object
        :return: None
        """

        if type(sources) is not list:
            # it is a single string:
            sources = [sources]

        # get the pointing flux files and the pointing ID, these arrays hsould be ordered with respect to one another
        # see for loop in init() to get the pointing IDs
        for pointing_file, point_id in zip(self.pointing_flux_files, self.pointing_ids):
            # make sure that the file exists
            if pointing_file.exists():
                # then read in the info and try to find where the object is within it if it exists there
                with fits.open(str(pointing_file)) as file:
                    # get the names of the sources
                    pointing_file_sources = file[1].data["NAME"]

                    # decode it
                    pointing_file_sources = [i for i in pointing_file_sources]

                    # iterate over each passed in source
                    for s in sources:
                        # get the index of the proper source name in the catalog
                        idx = np.arange(len(pointing_file_sources))[
                            self._compare_source_name(s, pointing_file_sources)
                        ]

                        if len(idx) != 0:
                            # then there is a row for the source of interest so we can read in the data and do the
                            # necessary calculations

                            # read in the cent rate, the error, etc and save it
                            rate_array = file[1].data[idx]["CENT_RATE"][0]
                            rate_err_array = file[1].data[idx]["RATE_ERR"][0]
                            bkg_var_array = file[1].data[idx]["BKG_VAR"][0]
                            snr_array = file[1].data[idx]["VECTSNR"][0]

                            self.set_pointing_info(
                                point_id, "rate", rate_array, source_id=s
                            )
                            self.set_pointing_info(
                                point_id, "rate_err", rate_err_array, source_id=s
                            )
                            self.set_pointing_info(
                                point_id, "bkg_var", bkg_var_array, source_id=s
                            )
                            self.set_pointing_info(
                                point_id, "snr", snr_array, source_id=s
                            )

                            # this does the calculation for the total energy range so set the if statement so the
                            # mosaic results dont attempt to calculate a wrong energy integrated count rate
                            if len(rate_array) == 8:
                                energy_idx = np.arange(len(rate_array))
                                (
                                    rate_tot,
                                    rate_err_2_tot,
                                    snr_allband_num,
                                ) = self.get_count_rate(energy_idx, point_id, s)

                                rate_array = np.concatenate(
                                    (
                                        self.pointing_info[point_id][s]["rate"],
                                        [rate_tot],
                                    )
                                )
                                rate_err_array = np.concatenate(
                                    (
                                        self.pointing_info[point_id][s]["rate_err"],
                                        [rate_err_2_tot],
                                    )
                                )
                                snr_array = np.concatenate(
                                    (
                                        self.pointing_info[point_id][s]["snr"],
                                        [snr_allband_num],
                                    )
                                )
                                # bkg_var_array = np.concatenate((bkg_var_array, [np.sqrt(bkg_var_2_tot)]))

                                self.set_pointing_info(
                                    point_id, "rate", rate_array, source_id=s
                                )
                                self.set_pointing_info(
                                    point_id, "rate_err", rate_err_array, source_id=s
                                )
                                self.set_pointing_info(
                                    point_id, "bkg_var", bkg_var_array, source_id=s
                                )
                                self.set_pointing_info(
                                    point_id, "snr", snr_array, source_id=s
                                )
                        else:
                            # a given pointing may not have the source in it so just raise a warning
                            try:
                                warn_str = (
                                    f"Observation ID: {self.obs_id} Pointing ID: {point_id} \n"
                                    f"There is no source {s} "
                                    f"found in the catalog file. Please double check the spelling.\nThis "
                                    f"source may also not be detected in this observation ID/pointing ID"
                                )
                                warnings.warn(warn_str)
                            except AttributeError:
                                warn_str = (
                                    f"Mosaic from {self.pointing_info['mosaic']['user_timebin']['utc_time']}-"
                                    f"{self.pointing_info['mosaic']['user_timebin']['utc_stop_time']}"
                                    f"\nThere is no source {s} found in the catalog file. Please double check "
                                    f"the spelling."
                                    f"\nThis source may also not be detected in this observation ID/pointing ID"
                                )
                                warnings.warn(warn_str)

    def get_pointing_ids(self):
        return self.pointing_ids

    def get_pointing_info(self, pointing_id, source_id=None):
        """
        gets the dictionary of information associated with the specified pointing id

        :param pointing_id: string of the pointing ID of interest
        :param source_id: None or string with the name of the source of interest. If value is None, the entire
            pointing_info[pointing_id] dictionary is returned
        :return: dict of the saved
        """

        if source_id is None:
            val = self.pointing_info[pointing_id]
        else:
            # the source_id dictionary within the pointing_id dictionary may not exist
            # also need to verify that the names are what we expect
            real_source_name = self.get_real_source_name(pointing_id, source_id)

            try:
                # if it does we are good
                val = self.pointing_info[pointing_id][real_source_name]
            except KeyError as ke:
                print(ke)
                raise ValueError(
                    "The dictionary for %s does not exist yet in the pointing id %s"
                    % (source_id, pointing_id)
                )

        return val

    def set_pointing_info(self, pointing_id, key, value, source_id=None):
        """
        Sets the key / value pair for the dictionary of information associated with the specified pointing id

        :param pointing_id: string of the pointing ID of interest
        :param source_id: None or string with the name of the source of interest. If value is None, the entire
            pointing_info[pointing_id] dictionary has the key/value pair appended or ammended to it. If the source is specified,
            then the source dictionary under the pointing ID has the key/value pair appended or ammended to it.
        :param key: string of the key that will be set in the pointing ID's dictionary
        :param value: the value that will be set in the dictionary for the associated key
        :return: None
        """
        if source_id is None:
            self.pointing_info[pointing_id][key] = value
        else:
            # the source_id dictionary within the pointing_id dictionary may not exist
            # also need to verify that the names are what we expect
            real_source_name = self.get_real_source_name(pointing_id, source_id)

            try:
                # if it does, we are good
                self.pointing_info[pointing_id][real_source_name][key] = value
            except KeyError:
                # otherwise create it and save the key value pair
                self.pointing_info[pointing_id][source_id] = dict()
                self.pointing_info[pointing_id][source_id][key] = value

    def get_pha_filenames(self, id_list=None, pointing_id_list=None, getupperlim=False):
        """
        Gets the pha filenames for the sources identified in id_list

        :param id_list: None, single string, or list of strings of catalog sources that the user wants to get the pha
            filenames of
        :param pointing_id_list: None, single string, or list of pointing IDs that the user wants to get the PHA
            filenames of
        :param getupperlim: Boolean to specify if the function should return just the upper limit PHA files. Default is
            False, meaning that both normal and upperlimit PHA files will be returned
        :return: returns a list of the pha filenames
        """

        # determine if we are dealing with survey or mosaic pha files
        for i in self.pha_file_names_list:
            if "survey" in i.name:
                split_str = "_survey"
            elif "mosaic" in i.name:
                split_str = "_mosaic"
            else:
                raise ValueError(
                    f"Could not determine if the pha file {i} belongs to a survey observation or a mosaiced image."
                )

        # make inputs into list if necessary
        if id_list is not None and type(id_list) is not list:
            # it is a single string:
            id_list = [id_list]
        if pointing_id_list is not None and type(pointing_id_list) is not list:
            # it is a single string:
            pointing_id_list = [pointing_id_list]

        if id_list is None:
            # get all the pha filenames
            val = self.pha_file_names_list
            if pointing_id_list is not None:
                # only get the pha filenames for the pointing ids specified
                val = [
                    i
                    for i in self.pha_file_names_list
                    if any(str(i) for j in pointing_id_list if j in str(i))
                ]
        else:
            # only get the pha filenames for the sources identified in id_list taking into account the real source name
            # that is specified in the BatSurvey dictionary which may be a different format than the pha file name
            val = [
                i
                for i in self.pha_file_names_list
                if any(
                    str(i)
                    for j in id_list
                    if self._compare_source_name(j, str(i.name).split(split_str)[0])
                )
            ]
            if pointing_id_list is not None:
                # only get the pha filenames for the pointing ids specified
                val = [
                    i
                    for i in val
                    if any(str(i) for j in pointing_id_list if j in str(i))
                ]

        if getupperlim:
            val = [i for i in val if "upperlim" in str(i)]

        return val

    def set_pha_filenames(self, file, reset=False):
        """
        Sets the list of pha filenames. Can reset the attibute so it is an empty list or append the attribute with additional
        filenames

        :param file: string of the pha filename that will be appended to the pha_file_names_list attribute
        :param reset: Boolean to determine if the attibute should be reset to an empty list
        :return: None
        """

        file = Path(file)

        if not reset:
            # not trying to reset the list of pha filenames
            self.pha_file_names_list.append(file)
        else:
            # reset the list
            self.pha_file_names_list = []

    def get_count_rate(self, energy_index, pointing_id, source):
        """
        This method returns the count rate information that was previously loaded by a call to the load_source_information
        method.

        :param energy_index: int or a list of ints that outline which energy channel(s) the user wants to get the counts for
        :param pointing_id: The pointing ID that the user would like to load the information from
        :param source: The name of the astrophysical source that the user would like the count information for
        :return: count rate, standard deviation of the count rate, the SNR of the detection
        """

        rate_array = self.get_pointing_info(pointing_id, source_id=source)["rate"]
        rate_err_array = self.get_pointing_info(pointing_id, source_id=source)[
            "rate_err"
        ]
        bkg_var_array = self.get_pointing_info(pointing_id, source_id=source)["bkg_var"]
        snr_array = self.get_pointing_info(pointing_id, source_id=source)["snr"]

        if type(energy_index) is not np.ndarray:
            if type(energy_index) is list:
                energy_index = np.array(energy_index)
            else:
                energy_index = np.array([energy_index])

        if len(energy_index) > 1:
            # vectorized with numpy
            rate_tot = np.sum(rate_array[energy_index])
            rate_err_2_tot = np.sum(rate_err_array[energy_index] ** 2)
            bkg_var_2_tot = np.sum(bkg_var_array[energy_index] ** 2)

            rate_err_tot = np.sqrt(rate_err_2_tot)
            snr_allband_num = rate_tot / np.sqrt(bkg_var_2_tot)

        else:
            rate_tot = rate_array[energy_index][0]
            rate_err_tot = rate_err_array[energy_index][0]
            snr_allband_num = snr_array[energy_index][0]

        return rate_tot, rate_err_tot, snr_allband_num

    def _compare_source_name(self, string_1, string_2):
        """
        This compares 2 strings that can be either the user supplied source ID or the source ID from a catalog and
        identifies if they are the same. This removes any non alphanumeric values(except dots) in each string and compares them.

        :param string_1: string
        :param string_2: string or array of strings
        :return: Boolean
        """

        reg = "[^0-9a-zA-Z.]"

        if type(string_1) is not str:
            raise ValueError("The first argument must be a single string")

        if type(string_2) is not list:
            string_2 = [string_2]

        if len(string_2) == 1:
            single_value = True
        else:
            single_value = False

        ret = [
            re.sub(reg, "", string_1).lower() == re.sub(reg, "", i).lower()
            for i in string_2
        ]

        if single_value:
            return ret[0]
        else:
            return ret

    def get_real_source_name(self, pointing_id, source):
        """
        This method deermines the real source name in the pointing ID's dictionary. This can be something that was
        passed in before when loading in calculated rate data or the name of a PHA file with the source name.
        This method matches these two formats so all the info related to a given source is saved appropriately.

        :param pointing_id: string of the pointing ID of interest
        :param source: string of the
        :return: string or None
        """
        # get the pointing info's keys
        key_list = list(self.get_pointing_info(pointing_id))

        # get the idx of the similar source name either from loading data or the pha filename
        idx = self._compare_source_name(source, key_list)

        # convert this to a np.array for indexing
        key_list = np.array(key_list)

        if np.sum(idx) == 1:
            real_source_name = key_list[idx][0]
        else:
            real_source_name = None

        return real_source_name


class MosaicBatSurvey(BatSurvey):
    """
    A general Bat Survey Mosaic object that holds all information necessary to analyze Bat survey moaic image that
    has already been created.

    Attributes
    ---------------
    result_dir : str
        The directory that holds the output of the user requested time bin mosaic calculation (ie the mosaic image
        and its asociated data products)
    pointing_flux_files : list of strings
        A list of the source catalog files created by heasoftpy batcelldetect for the specified source catalog and
        mosaic image
    pointing_ids : list of strings
        The pointing  ids for the successfully analyzed pointings associated with the analyzed user defined mosaic
        time bin. This is just set as 'mosaic'
    pointing_info : dictionary of dictionaries
        The encompassed information including MET time, exposure time, etc for each pointing in a mosaic image
        Can be access as pointing_info[pointing_id]["key"]. These values are not necessarily equal to the user defined
        parameters for the creation of the mosaic image.
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
    load(f):
        Load a MosaicBatSurvey object
    save():
        Saves a MosaicBatSurvey object
    merge_pointings(input_dict=None, verbose=False):
        Merges the counts from multiple pointings found within an observation ID dataset
    calculate_pha(id_list=None, output_dir=None, calc_upper_lim=False, bkg_nsigma=None, verbose=True, clean_dir=False,
            single_pointing=None):
        Calculates the PHA file for each source found in the mosaic image
    load_source_information(sources):
        Loads the count rate, background variance, and snr from the .cat file produced by batcelldetect for the sources
        of interest
    get_pointing_ids():
        Returns the pointing ids in the observation ID
    get_pointing_info(pointing_id, source_id=None)
        Gets the dictionary of information associated with the specified pointing id and source id if specified
    set_pointing_info(pointing_id, key, value, source_id=None)
        Sets the key/value pair for the dictionary of information associated with the specified pointing id and source,
        if the source_id is specified
    get_pha_filenames(id_list=None, pointing_id_list=None)
        Gets the pha filename list of the sources supplied in id_list and for the pointing ids supplied by
        pointing_id_list
    set_pha_filenames(file, reset=False)
        Sets the pha filenames attribute or resets it to be an empty list
    load_source_information(sources)
        Loads the rates information for a given source from the pointing_flux_files
    get_count_rate(energy_index, pointing_id, source)
        Returns the count rate information that is requested for a single (or multiple) energy range(s), a specified
            pointing ID, and a specified source.
     detect_sources(catalog_file=None, input_dict=None)
         Calls batcelldetect to detect sources in the mosaic image that is encompassed by a given MosaicBatSurvey object
    """

    def __init__(self, mosaic_dir, recalc=False):
        """
        Initializer method for the MosaicBatSurvey object.

        :param mosaic_dir: path object to the location of the mosaiced images that were calculated
        :param recalc: Boolean default False, which indicates that the method should try to load data from a file in
            the mosaic_dir directory. True means that the load file will be ignored and attributes will be re-obtained
            for the object.
        """

        # this isnt proper usage of super classes since the below lines are in the init of the BatSurvey class
        # just doing this for testing now, can polish and fix this later
        # Set default energy ranges in keV and system errors
        self.channel = [1, 2, 3, 4, 5, 6, 7, 8]
        self.emin = [14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0]
        self.emax = [20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0]
        self.syserr = [0.6, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15, 0.6]

        # initalize the pha filename list attribute
        self.pha_file_names_list = []

        self.result_dir = Path(mosaic_dir).expanduser().resolve()

        load_file = self.result_dir.joinpath("batsurvey.pickle")
        self._local_pfile_dir = self.result_dir.joinpath(".local_pfile")

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        # See if a loadfile exists, if we dont want to recalcualte everything, otherwise remove any load file and
        # .batsurveycomplete file (this is produced only if the batsurvey calculation was completely finished, and thus
        # know that we can safely load the batsurvey.pickle file)
        if not load_file.exists() or recalc:
            # can have a mosaic directory with no mosaic-ed images since there would be no survey observations in the
            # time bin. In this case throw an error
            if not self.result_dir.joinpath("swiftbat_exposure_c0.img").exists():
                raise ValueError("This mosaic time bin is invalid.")

            # get the number of mosaic facets
            # self.nfacets=len(glob.glob(os.path.join(os.path.split(__file__)[0], "data",'ra_*.img')))
            self.nfacets = len(
                sorted(Path(__file__).parent.joinpath("data").glob("ra_*.img"))
            )

            # need to set the mosaic pointing ID
            self.pointing_ids = ["mosaic"]

            # create dict of mosaic pointings ids and their respective information of time, exposure, etc which is
            # the same for each pointing
            self.pointing_info = dict.fromkeys(self.pointing_ids)
            for point_id in self.pointing_ids:
                file = fits.open(
                    str(self.result_dir.joinpath("swiftbat_exposure_c0.img"))
                )  # os.path.join(mosaic_dir, 'swiftbat_exposure_c0.img'))
                file_header = file[0].header

                time_array = file_header[
                    "TSTART"
                ]  # MET time of first survey dataset to be part of mosaic
                time_array_stop = file_header[
                    "TSTOP"
                ]  # MET time of last survey data to be part of mosaic
                exposure_array = file_header["EXPOSURE"]  # MET time in s
                time_elapse = file_header["TELAPSE"]  # MET time in s

                # calculate times in UTC and MJD units as well
                mjdtime = met2mjd(time_array)
                utctime = met2utc(time_array, mjd_time=mjdtime)

                mjdtime_stop = met2mjd(time_array_stop)
                utctime_stop = met2utc(time_array_stop, mjd_time=mjdtime_stop)

                tbin_start_met = file_header["S_TBIN"]
                tbin_end_met = file_header["E_TBIN"]

                file.close()

                tbin_start_mjdtime = met2mjd(tbin_start_met)
                tbin_start_utctime = met2utc(
                    tbin_start_met, mjd_time=tbin_start_mjdtime
                )

                tbin_end_mjdtime = met2mjd(tbin_end_met)
                tbin_end_utctime = met2utc(tbin_end_met, mjd_time=tbin_end_mjdtime)

                user_timebin = dict(
                    met_time=tbin_start_met,
                    met_stop_time=tbin_end_met,
                    utc_time=tbin_start_utctime,
                    mjd_time=tbin_start_mjdtime,
                    utc_stop_time=tbin_end_utctime,
                    mjd_stop_time=tbin_end_mjdtime,
                )

                self.pointing_info[point_id] = dict(
                    met_time=time_array,
                    exposure=exposure_array,
                    utc_time=utctime,
                    mjd_time=mjdtime,
                    elapse_time=time_elapse,
                    met_stop_time=time_array_stop,
                    utc_stop_time=utctime_stop,
                    mjd_stop_time=mjdtime_stop,
                    user_timebin=user_timebin,
                )
        else:
            self.load(load_file)


    def detect_sources(self, catalog_file=None, input_dict=None):
        """
        Detect sources in the skygrid facets. This currently does not detect new sources.

        :param catalog_file: None or a Path object to a catalog that the user has created. None defaults to using the
            default catalog file included with the BATAnalysis package.
        :param input_dict: A custom input dictionary with key/value pairs that will be passed to batcelldetect.
        :return: None
        """

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        # need to iterate through all the facets and detect the sources, then we merge the output catalogs
        if catalog_file is None:
            # catalog_file=os.path.join(os.path.split(__file__)[0], "data", 'survey6b_2.cat')
            catalog_file = (
                Path(__file__).parent.joinpath("data").joinpath("survey6b_2.cat")
            )
        else:
            catalog_file = Path(catalog_file).expanduser().resolve()

        resulting_files = ""
        for num in range(self.nfacets):
            file = self.result_dir.joinpath(f"swiftbat_flux_c{num}.img")
            pcodefile = self.result_dir.joinpath(f"swiftbat_exposure_c{num}.img")
            outfile = self.result_dir.joinpath(f"sources_c{num}.cat")

            batcelldetect_input_dict = dict(
                infile=f"{file}",
                # [col #HDUCLAS2 = "NET"; #FACET = {num}]', #This isnt needed since this info is in HDUCLAS2 and
                # BSKYPLAN already
                outfile=str(outfile),
                snrthresh=4.0,
                psfshape="GAUSSIAN",
                psffwhm="0.325",
                srcfit="YES",
                posfit="NO",
                posfitwindow=0.0,
                bkgwindowtype="SMOOTH_CIRCLE",
                srcdetect="NO",
                nadjpix=1,
                srcradius=15,
                bkgradius=100,
                bkgfit="no",
                keepbits="ALL",
                hduclasses="NONE",
                chatter=2,
                clobber="YES",
                distfile="NONE",
                pcodefile=str(pcodefile),
                pcodethresh=0.1,
                nullborder="NO",
                incatalog=str(catalog_file),
                vectorflux="YES",
                vectorposmeth="MAX_SNR",
                regionfile=str(self.result_dir.joinpath("test.reg")),
                keepkeywords="FACET,CRVAL1,CRVAL2,*VER",
            )

            if input_dict is None:
                passed_input_dict = batcelldetect_input_dict.copy()
            else:
                passed_input_dict = input_dict.copy()

            self.batcelldetect_output = self._call_batcelldetect(passed_input_dict)

            resulting_files = resulting_files + str(outfile) + " "

        # need to merge them and sort them
        all_src_file = self.result_dir.joinpath(
            "sources_tot.cat"
        )  # os.path.join(self.result_dir,'sources_tot.cat')
        tmp_all_src_file = self.result_dir.joinpath(
            "tmp_sources_tot.cat"
        )  # os.path.join(self.result_dir, 'tmp_sources_tot.cat')
        heatools.ftmerge(infile=str(resulting_files), outfile=str(tmp_all_src_file))

        # get the coordinates from galactic to RA/DEC
        heatools.ftcoco(
            infile=str(tmp_all_src_file),
            outfile=str(all_src_file),
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
            infile=f"{all_src_file}[col *;FACET_DIST = ANGSEP(GLON_OBJ,GLAT_OBJ,CRVAL1,CRVAL2); DUP = F]",
            outfile=str(tmp_all_src_file),
            columns="CATNUM, FACET_DIST",
            clobber="YES",
        )
        heatools.ftcopy(
            infile=f"{tmp_all_src_file}[col *; DUP=(CATNUM == CATNUM{{-1}})?T:F;]",
            outfile=str(all_src_file),
            clobber="YES",
        )
        heatools.ftselect(
            infile=f"{all_src_file}[col CATNUM;NAME;RA_OBJ;DEC_OBJ;BAT_NAME=CATNUM;RATE=CENT_RATE;RATE_ERR=BKG_VAR;VECTSNR=CENT_SNR;CENT_RATE;BKG_VAR;TIME;TIME_STOP;EXPOSURE=PCODEFR;DUP]",
            outfile=str(tmp_all_src_file),
            expression="(DUP==F || isnull(DUP))",
            clobber="YES",
        )

        # os.system(f"mv {tmp_all_src_file} {all_src_file}")
        tmp_all_src_file.rename(all_src_file)
        self.pointing_flux_files = [all_src_file]

    def calculate_pha(
        self,
        id_list,
        output_dir=None,
        calc_upper_lim=False,
        bkg_nsigma=None,
        verbose=True,
        clean_dir=False,
        single_pointing=None,
    ):
        """
        This function calculates the pha files for each object in the input catalog file by default. Can specify
        'keycolumn' value for specific objects. Based on make_spectrum.py by Taka and Amy

        :param id_list: A string or a List of Strings or None Denoting which sources the user wants the PHA files calculated for.
            None is reserved for when the user wants to calculate PHA files for all the sources in the catalog which includes those
            from the default bat survey catalog
        :param output_dir: None or a string where the output PHA file should be saved
        :param calc_upper_lim: Boolean to denote if the PHA file should be constructed to calculate upper limits for an
            object. This is done by using the bkg_var instead of the cent_rate for the count information in the
            produced PHA file
        :param bkg_nsigma: Float for the significance of the background scaling to obtain an upper limit at that limit
            (eg PHA count = bkg_nsigma*bkg_var)
        :param verbose: Boolean to print diagnostic information
        :param clean_dir: Boolean to denote if the resulting PHA_files/ directory should be removed and recreated
        :param single_pointing: Not used for the MosaicBatSurvey since there is only one pointing ID, just kept kere for
            compatibility with calls to the BatSurvey.calculate_pha calls in various functions
        :return: None
        """

        if calc_upper_lim and bkg_nsigma is None:
            raise ValueError(
                "A value for bkg_nsigma has not been passed to the function to calculate upper limits."
            )

        if output_dir is None:
            # set default directory to save files into
            output_dir = self.result_dir.joinpath("PHA_files")
        else:
            output_dir = Path(output_dir).expanduser().resolve()

        if not calc_upper_lim:
            # see if directory exists, if no create of so then delete and recreate
            # if we are calculating the upper limit, the directory already exists
            dirtest(output_dir, clean_dir=clean_dir)

        merge_output_path = Path(self.merge_input["outfile"]).parent

        # if something has been passed in make sure that its a list
        if id_list is not None:
            if type(id_list) is not list:
                # it is a single string:
                id_list = [id_list]
        else:
            # use the ids from the *.cat files produced, these are ones that have been identified in the survey obs_id
            x = sorted(merge_output_path.glob("*.cat"))
            id_list = [i.stem for i in x]

        # for each object/source of interest
        # reset the save the pha file names of all pha files if necessary
        if clean_dir:
            self.set_pha_filenames("", reset=True)

        # need to put the repsonse file in the directory since the filename with the full path can be way to long to
        # fit in the pha file and be read in by xspec, or can try doing a symbolic link responsefile=os.path.join(
        responsefile = (
            Path(__file__)
            .parent.joinpath("data")
            .joinpath("swiftbat_survey_full_157m.rsp")
        )
        copied_responsefile = output_dir.joinpath(responsefile.name)
        # if the file doesnt exist in the directory create a sym link to the file
        if not copied_responsefile.exists():
            copied_responsefile.symlink_to(responsefile)

        for ident in id_list:
            if verbose:
                print("Creating PHA file for ", ident)

            # get the proper name for the source incase the user didnt get the name correct
            x = sorted(merge_output_path.glob("*.cat"))
            catalog_sources = [i.stem for i in x]
            test = self._compare_source_name(ident, catalog_sources)
            if np.sum(test) > 0:
                ident = np.array(catalog_sources)[
                    self._compare_source_name(ident, catalog_sources)
                ][0]
            else:
                ident = None

            # get info from the newly created cat file (from merge)
            try:
                catalog = merge_output_path.joinpath(f"{ident}.cat")
                cat_file = fits.open(catalog)
                tbdata = cat_file[1].data
                name_array = tbdata.field("NAME")
                raobj_array = tbdata.field("RA_OBJ")
                decobj_array = tbdata.field("DEC_OBJ")
                time_array = tbdata.field("TIME")
                tstart_sinceT0 = np.zeros_like(time_array)  # need to understand this
                timestop_array = tbdata.field("TIME_STOP")
                exposure_array = tbdata.field("EXPOSURE")

                if calc_upper_lim:
                    count_rate_array = tbdata.field("BKG_VAR")
                    count_rate_err_array = np.zeros_like(count_rate_array)
                    scale = bkg_nsigma
                else:
                    count_rate_array = tbdata.field("CENT_RATE")
                    count_rate_err_array = tbdata.field("BKG_VAR")
                    scale = 1

                cat_file.close()

                # make pha file
                # write count_rate in each band to an pha file, exclude the 14-195 count
                spec_col1 = fits.Column(name="CHANNEL", format="I", array=self.channel)
                spec_col2 = fits.Column(
                    name="RATE",
                    format="E",
                    unit="count/s/pixel",
                    array=scale * count_rate_array[0][:-1],
                )
                spec_col3 = fits.Column(
                    name="STAT_ERR",
                    format="E",
                    unit="count/s/pixel",
                    array=count_rate_err_array[0][:-1],
                )

                ebound_col1 = fits.Column(
                    name="CHANNEL", format="1I", unit="", array=self.channel
                )
                ebound_col2 = fits.Column(
                    name="E_MIN", format="1E", unit="keV", array=self.emin
                )
                ebound_col3 = fits.Column(
                    name="E_MAX", format="1E", unit="keV", array=self.emax
                )

                spec_cols = fits.ColDefs([spec_col1, spec_col2, spec_col3])
                ebound_cols = fits.ColDefs([ebound_col1, ebound_col2, ebound_col3])

                spec_tbhdu = fits.BinTableHDU.from_columns(spec_cols)
                ebound_tbhdu = fits.BinTableHDU.from_columns(ebound_cols)

                spec_tbhdu.name = "SPECTRUM"
                ebound_tbhdu.name = "EBOUNDS"

                pha_primary = fits.PrimaryHDU()

                pha_thdulist = fits.HDUList([pha_primary, spec_tbhdu, ebound_tbhdu])

                if calc_upper_lim:
                    survey_pha_file = output_dir.joinpath(
                        f"{ident}_mosaic_bkgnsigma_{int(bkg_nsigma)}_upperlim.pha"
                    )

                else:
                    survey_pha_file = output_dir.joinpath(f"{ident}_mosaic.pha")
                self.set_pha_filenames(survey_pha_file)
                pha_thdulist.writeto(str(survey_pha_file))

                pha_hdulist = fits.open(str(survey_pha_file), mode="update")

                pha_prime_hdr = pha_hdulist[0].header
                pha_spec_hdr = pha_hdulist[1].header
                pha_ebound_hdr = pha_hdulist[2].header

                pha_prime_hdr["TELESCOP"] = ("SWIFT", "Telescope (mission) name")
                pha_prime_hdr["INSTRUME"] = ("BAT", "Instrument name")
                pha_prime_hdr["OBJECT"] = (name_array[0], "Object name")
                pha_prime_hdr["RA_OBJ"] = (raobj_array[0], "[deg] R.A. Object")
                pha_prime_hdr["DEC_OBJ"] = (decobj_array[0], "[deg] Dec Object")
                pha_prime_hdr["TSTART"] = (time_array[0], "Start time")
                pha_prime_hdr["TSTOP"] = (timestop_array[0], "Stop time")

                pha_spec_hdr["TELESCOP"] = ("SWIFT", "Telescope (mission) name")
                pha_spec_hdr["INSTRUME"] = ("BAT", "Instrument name")
                pha_spec_hdr["HDUCLASS"] = ("OGIP", "Conforms to OGIP/GSFC standards")
                pha_spec_hdr["HDUCLAS1"] = ("SPECTRUM", "Contains spectrum")
                pha_spec_hdr["GAINAPP"] = ("T", "Gain correction has been applied")
                pha_spec_hdr["GAINMETH"] = (
                    "FIXEDDAC",
                    "Cubic ground gain/offset correction using DAC-b",
                )
                pha_spec_hdr["EXPOSURE"] = (
                    exposure_array[0],
                    "[s] on-axis equivalent exposure (s)",
                )
                pha_spec_hdr["TSTART"] = (time_array[0], "Start time")
                pha_spec_hdr["TSTOP"] = (timestop_array[0], "Stop time")
                pha_spec_hdr["AREASCAL"] = (1.0, "Nominal effective area")
                pha_spec_hdr["BACKSCAL"] = (1.0, "Background scale factor")
                pha_spec_hdr["CORRSCAL"] = (1.0, "Correction scale factor")
                pha_spec_hdr["BACKFILE"] = ("none", "Background FITS file")
                pha_spec_hdr["CORRFILE"] = ("none", "Correction FITS file")
                pha_spec_hdr["RESPFILE"] = (
                    responsefile.name,
                    "Redistribution Matrix file (RMF)",
                )
                pha_spec_hdr["ANCRFILE"] = ("none", "Effective Area file (ARF)")
                pha_spec_hdr["QUALITY"] = (0, "Data quality flag")
                pha_spec_hdr["GROUPING"] = (0, "Spectra are not grouped")
                pha_spec_hdr["POISSERR"] = ("F", "Poisson errors do not apply")
                pha_spec_hdr["SYS_ERR"] = (0.0, "Systematic error value")
                pha_spec_hdr["DETCHANS"] = (
                    8,
                    "Total number of detector channels available",
                )
                pha_spec_hdr["CHANTYPE"] = ("PI", "Pulse height channel type")
                pha_spec_hdr["HDUCLAS2"] = ("NET", "Spectrum is background subtracted")
                pha_spec_hdr["HDUCLAS3"] = ("RATE", "Spectrum is count/s")
                pha_spec_hdr["PHAVERSN"] = ("1992a", "Version of spectrum format")
                pha_spec_hdr["HDUVERS"] = ("1.2.0", "Version of spectrum header")

                pha_spec_hdr["DATE-OBS"] = (
                    "2004-11-20T12:16:00",
                    "fake date-obs on UTC",
                )

                pha_ebound_hdr["HDUCLASS"] = ("OGIP", "Conforms to OGIP/GSFC standards")
                pha_ebound_hdr["HDUCLAS1"] = ("RESPONSE", "Contains spectrum")

                pha_hdulist.flush()
            except FileNotFoundError:
                print("The source %s was not found in the mosaiced image." % (ident))
