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

import numpy as np
from astropy.io import fits

from .batlib import datadir, dirtest, met2mjd, met2utc
from .batobservation import BatObservation

# for python>3.6
try:
    import heasoftpy.swift as hsp
    import heasoftpy.utils as hsp_util
except ModuleNotFoundError as err:
    # Error handling
    print(err)


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
        """

        # Set default energy ranges in keV and system errors
        self.channel = [1, 2, 3, 4, 5, 6, 7, 8]
        self.emin = [14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0]
        self.emax = [20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0]
        self.syserr = [0.6, 0.3, 0.15, 0.15, 0.15, 0.15, 0.15, 0.6]

        # make sure that the observation ID is a string
        if type(obs_id) is not str:
            obs_id = f"{int(obs_id)}"

        # initalize the pha filename list attribute
        self.pha_file_names_list = []

        # initialize super class
        super().__init__(obs_id, obs_dir)

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
        # print(os.getenv("PFILES"))

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

            if patt_noise_dir is None:
                patt_noise_dir = datadir().joinpath("noise_pattern_maps")
            else:
                # make a Path object
                patt_noise_dir = Path(patt_noise_dir)

            # read in the header of a file in the survey observation ID directory to get the MET start time and
            # convert to year/day of year
            input_file = sorted(
                self.obs_dir.joinpath("bat").joinpath("survey").glob("*")
            )[0]
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
                            np.datetime64(
                                datetime(int(i), 1, 1) + timedelta(int(j) - 1)
                            )
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

            if input_dict is None:
                input_dict_copy = dict(
                    indir=str(self.obs_dir),
                    outdir=str(
                        self.obs_dir.parent / f"{self.obs_dir.name}_surveyresult"
                    ),
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
                if "indir" not in input_dict_copy or input_dict_copy["indir"] is None:
                    input_dict_copy["indir"] = str(self.obs_dir)
                else:
                    # make this a fully resolved path
                    if not Path(input_dict_copy["indir"]).is_absolute():
                        input_dict_copy["indir"] = str(
                            Path.cwd().joinpath(input_dict_copy["indir"])
                        )

                if "outdir" not in input_dict_copy or input_dict_copy["outdir"] is None:
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
                        or input_dict_copy["detthresh"] is None
                ):
                    input_dict_copy["detthresh"] = "10000"

                if (
                        "detthresh2" not in input_dict_copy
                        or input_dict_copy["detthresh2"] is None
                ):
                    input_dict_copy["detthresh2"] = "10000"

                if (
                        "incatalog" not in input_dict_copy
                        or input_dict_copy["incatalog"] is None
                ):
                    input_dict_copy["incatalog"] = str(
                        Path(__file__).parent.joinpath("data/survey6b_2.cat")
                    )

                if (
                        "global_pattern_map" not in input_dict_copy
                        or input_dict_copy["global_pattern_map"] is None
                ):
                    input_dict_copy["global_pattern_map"] = str(patt_map_name)

                if (
                        "global_pattern_mask" not in input_dict_copy
                        or input_dict_copy["global_pattern_mask"] is None
                ):
                    input_dict_copy["global_pattern_mask"] = str(patt_mask_name)

                if (
                        "cleansnr" not in input_dict_copy
                        or input_dict_copy["cleansnr"] is None
                ):
                    input_dict_copy["cleansnr"] = 6

                if (
                        "cleanexpr" not in input_dict_copy
                        or input_dict_copy["cleanexpr"] is None
                ):
                    input_dict_copy["cleanexpr"] = "ALWAYS_CLEAN==T"

                # make sure that the output directory exists
                if not Path(input_dict_copy["outdir"]).parent.exists():
                    raise ValueError(
                        "The directory %s needs to exist for batsurvey to save its results."
                        % (os.path.split(input_dict_copy["outdir"])[0])
                    )

            # save what is passed to batsurvey
            self.survey_input = input_dict_copy

            # save result directory
            self.result_dir = Path(input_dict_copy["outdir"])

            # if the user has already done this calculation and wants to redo it, can set clobber to True in input_dict
            if recalc:
                input_dict_copy["clobber"] = "YES"
                if self.result_dir.exists():
                    shutil.rmtree(self.result_dir)

            # if the user wants to relaculate things or if recalc==False but the result directory specified doesnt exist
            # we need to recalculate things for further processing, IMPLEMENT LATER ON
            # call the heasoftpy command
            bs = self._call_batsurvey(input_dict_copy)
            self.batsurvey_result = bs
            # can print output of batsurvey with ba.stdout.split("\n")

            shutil.rmtree(self._local_pfile_dir)
            self._local_pfile_dir = self.result_dir.joinpath(".local_pfile")

            # make the local pfile dir if it doesnt exist and set this value
            self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
            try:
                hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
            except AttributeError:
                hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

            complete_file = self.result_dir.joinpath(".batsurvey_complete")

            # identify the pointings that have been created
            # all these pointings may not have the object of interest, so need to double-check that with the
            # merge_pointings method
            self.pointing_flux_files = sorted(
                self.result_dir.glob(f"point*/*_{bs.params['ncleaniter']}.cat")
            )  # glob.glob(f"{self.result_dir}/point*/*_{bs.params['ncleaniter']}.cat")

            # need to extract the respective pointing IDs
            self.pointing_ids = []
            for pointing in self.pointing_flux_files:
                self.pointing_ids.append(
                    pointing.parent.name.split("_")[-1]
                )

            # create dict of pointings ids and their respective information of time, exposure, etc which is the same
            # for each pointing
            self.pointing_info = dict.fromkeys(self.pointing_ids)
            for pointing, ident in zip(self.pointing_flux_files, self.pointing_ids):
                lc_fits = fits.open(pointing)
                lc_fits_data = lc_fits[1].data

                time_array = lc_fits_data.field("TIME")[0]  # MET start time
                exposure_array = lc_fits_data.field("EXPOSURE")[0]  # MET time in s

                # calculate times in UTC and MJD units as well
                mjdtime = met2mjd(time_array)
                utctime = met2utc(time_array, mjd_time=mjdtime)

                lc_fits.close()

                # open the status file to save the success code
                status_file = pointing.parent.joinpath(f"point_{ident}_status.txt")
                stat = self._batsurvey_error(status_file)[0]

                self.pointing_info[ident] = dict(
                    success=stat,
                    fail_code=None,
                    met_time=time_array,
                    exposure=exposure_array,
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
        file = self.result_dir.joinpath(
            "batsurvey.pickle"
        )
        with open(file, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
        print("A save file has been written to %s." % (str(file)))

    def _call_batsurvey(self, input_dict):
        """
        Calls heasoftpy's batsurvey with an error wrapper
        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batsurvey
        :return: heasoftpy Result object from batsurvey
        """
        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        # directly calls batsurvey
        try:
            return hsp.batsurvey(**input_dict)
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
            output_dir = self.result_dir.joinpath(
                "merged_pointings_lc"
            )
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
            output_dir = self.result_dir.joinpath(
                "PHA_files"
            )
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
            catalog = merge_output_path.joinpath(
                f"{ident}.cat"
            )
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
                    count_rate_band = (
                        []
                    )
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
                        ).joinpath(
                            f"{pointing_array[i]}.att"
                        )
                        dpifile = self.result_dir.joinpath(
                            f"{pointing_array[i]}"
                        ).joinpath(
                            f"{pointing_array[i]}_1.dpi"
                        )
                        detmask = self.result_dir.joinpath(
                            f"{pointing_array[i]}"
                        ).joinpath(
                            f"{pointing_array[i]}.detmask"
                        )
                        output_srcmask = output_dir.joinpath(
                            "src.mask"
                        )

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
                        pha_prime_hdr["MJDREFI"] = (51910.0, " Reference MJD Integer part")
                        pha_prime_hdr["MJDREFF"] = (0.00074287037, " Reference MJD fractional")
                        pha_prime_hdr["TIMEREF"] = ("LOCAL", " Time reference (barycenter/local)")
                        pha_prime_hdr["TASSIGN"] = ("SATELLITE", " Time assigned by clock")
                        pha_prime_hdr["TIMEUNIT"] = ("s", " Time unit")
                        pha_prime_hdr["TIERRELA"] = (1.0e-8, " [s/s] relative errors expressed as rate")
                        pha_prime_hdr["TIERABSO"] = (1.0, " [s] timing precision in seconds")
                        pha_prime_hdr["CLOCKAPP"] = ("F", "Is mission time corrected for clock drift?")

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

                        utc_starttime = met2utc(gti_starttime[0]).astype('datetime64[s]')
                        utc_stoptime = met2utc(gti_stoptime[0]).astype('datetime64[s]')
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
                        pha_spec_hdr["MJDREFI"] = (51910.0, " Reference MJD Integer part")
                        pha_spec_hdr["MJDREFF"] = (0.00074287037, " Reference MJD fractional")
                        pha_spec_hdr["TIMEREF"] = ("LOCAL", " Time reference (barycenter/local)")
                        pha_spec_hdr["TASSIGN"] = ("SATELLITE", " Time assigned by clock")
                        pha_spec_hdr["TIMEUNIT"] = ("s", " Time unit")
                        pha_spec_hdr["TIERRELA"] = (1.0e-8, " [s/s] relative errors expressed as rate")
                        pha_spec_hdr["TIERABSO"] = (1.0, " [s] timing precision in seconds")
                        pha_spec_hdr["CLOCKAPP"] = ("F", "Is mission time corrected for clock drift?")
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
                        raise RuntimeError("Found more than one matched time, please double check the time interval.\n"
                                           "This method does not add up the counts for more than one time intervals.")
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

                            self.set_pointing_info(point_id, "rate", rate_array, source_id=s)
                            self.set_pointing_info(
                                point_id, "rate_err", rate_err_array, source_id=s
                            )
                            self.set_pointing_info(
                                point_id, "bkg_var", bkg_var_array, source_id=s
                            )
                            self.set_pointing_info(point_id, "snr", snr_array, source_id=s)

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
                                    (self.pointing_info[point_id][s]["rate"], [rate_tot])
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
                                warn_str = (f"Observation ID: {self.obs_id} Pointing ID: {point_id} \n"
                                            f"There is no source {s} "
                                            f"found in the catalog file. Please double check the spelling.\nThis "
                                            f"source may also not be detected in this observation ID/pointing ID")
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

            default_input_dict = dict(
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
                passed_input_dict = default_input_dict.copy()
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
        hsp.ftmerge(infile=str(resulting_files), outfile=str(tmp_all_src_file))

        # get the coordinates from galactic to RA/DEC
        hsp.ftcoco(
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
        hsp.ftsort(
            infile=f"{all_src_file}[col *;FACET_DIST = ANGSEP(GLON_OBJ,GLAT_OBJ,CRVAL1,CRVAL2); DUP = F]",
            outfile=str(tmp_all_src_file),
            columns="CATNUM, FACET_DIST",
            clobber="YES",
        )
        hsp.ftcopy(
            infile=f"{tmp_all_src_file}[col *; DUP=(CATNUM == CATNUM{{-1}})?T:F;]",
            outfile=str(all_src_file),
            clobber="YES",
        )
        hsp.ftselect(
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
            output_dir = self.result_dir.joinpath(
                "PHA_files"
            )
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
        copied_responsefile = output_dir.joinpath(
            responsefile.name
        )
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
                catalog = merge_output_path.joinpath(
                    f"{ident}.cat"
                )
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
