"""

This file is meant specifically for the object that reads in and processes TTE data.

Tyler Parsotan April 5 2023

"""

import os
import shutil
import sys
from .batlib import datadir, dirtest, met2mjd, met2utc
from .batobservation import BatObservation
import glob
from astropy.io import fits
import numpy as np
import subprocess
import pickle
import sys
import re
from pathlib import Path
from astropy.time import Time
from datetime import datetime, timedelta
import re
import warnings

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatEvent(BatObservation):
    def __init__(
        self,
        obs_id,
        transient_name=None,
        obs_dir=None,
        input_dict=None,
        recalc=False,
        verbose=False,
        load_dir=None,
    ):
        # make sure that the observation ID is a string
        if type(obs_id) is not str:
            obs_id = f"{int(obs_id)}"

        # initialize super class
        super().__init__(obs_id, obs_dir)

        # See if a loadfile exists, if we dont want to recalcualte everything, otherwise remove any load file and
        # .batsurveycomplete file (this is produced only if the batsurvey calculation was completely finished, and thus
        # know that we can safely load the batsurvey.pickle file)
        if not recalc and load_dir is None:
            load_dir = sorted(self.obs_dir.parent.glob(obs_id + "_event*"))

            # see if there are any _surveyresult dir or anything otherwise just use obs_dir as a place holder
            if len(load_dir) > 0:
                load_dir = load_dir[0]
            else:
                load_dir = self.obs_dir
        elif not recalc and load_dir is not None:
            load_dir_test = sorted(Path(load_dir).glob(obs_id + "_event*"))
            # see if there are any _surveyresult dir or anything otherwise just use load_dir as a place holder
            if len(load_dir_test) > 0:
                load_dir = load_dir_test[0]
            else:
                load_dir = Path(load_dir)
        else:
            # just give dummy values that will be written over later
            load_dir = self.obs_dir

        load_file = load_dir.joinpath("batevent.pickle")
        complete_file = load_dir.joinpath(".batevent_complete")
        self._local_pfile_dir = load_dir.joinpath(".local_pfile")

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp.utils.local_pfiles(par_dir=str(self._local_pfile_dir))

        # if load_file is None:
        # if the user wants to recalculate things or if there is no batevent.pickle file, or if there is no
        # .batevent_complete file (meaning that the __init__ method didnt complete)
        if recalc or not load_file.exists() or not complete_file.exists():
            if (
                not self.obs_dir.joinpath("bat").joinpath("event").is_dir()
                or not self.obs_dir.joinpath("bat").joinpath("hk").is_dir()
                or not self.obs_dir.joinpath("bat").joinpath("rate").is_dir()
                or not self.obs_dir.joinpath("tdrss").is_dir()
                or not self.obs_dir.joinpath("auxil").is_dir()
            ):
                raise ValueError(
                    "The observation ID folder needs to contain the bat/event/, the bat/hk/, the bat/rate/, the auxil/, and tdrss/ subdirectories in order to "
                    + "analyze BAT event data. One or many of these folders are missing."
                )

            # save the necessary files that we will need through the processing/analysis steps
            self.enable_disable_file = list(
                self.obs_dir.joinpath("bat").joinpath("hk").glob("*bdecb*")
            )
            # the detector quality is combination of enable/disable detectors and currently (at tiem of trigger) hot detectors
            # https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html
            self.detector_quality_file = list(
                self.obs_dir.joinpath("bat").joinpath("hk").glob("*bdqcb*")
            )
            self.event_files = list(
                self.obs_dir.joinpath("bat").joinpath("event").glob("*bevsh*_uf*")
            )
            self.attitude_file = list(self.obs_dir.joinpath("auxil").glob("*sat.*"))
            self.tdrss_files = list(self.obs_dir.joinpath("tdrss").glob("*msb*.fits*"))
            self.gain_offset_file = list(
                self.obs_dir.joinpath("bat").joinpath("hk").glob("*bgocb*")
            )

            # make sure that there is only 1 attitude file
            if len(self.attitude_file) > 1:
                raise ValueError(
                    f"There seem to be more than one attitude file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            elif len(self.attitude_file) < 1:
                raise ValueError(
                    f"There seem to be no attitude file for this trigger with observation ID \
                                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            else:
                self.attitude_file = self.attitude_file[0]

            # make sure that there is at least one event file
            if len(self.event_files) < 1:
                raise FileNotFoundError(
                    f"There seem to be no event files for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )

            # make sure that we have an enable disable map
            if len(self.enable_disable_file) < 1:
                raise FileNotFoundError(
                    f"There seem to be no detector enable/disable file for this trigger with observation "
                    f"ID {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            elif len(self.enable_disable_file) > 1:
                raise ValueError(
                    f"There seem to be more than one detector enable/disable file for this trigger with observation ID "
                    f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            else:
                self.enable_disable_file = self.enable_disable_file[0]

            # make sure that we have a detector quality map
            if len(self.detector_quality_file) < 1:
                if verbose:
                    print(
                        f"There seem to be no detector quality file for this trigger with observation ID"
                        f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                    )

                # need to create this map can get to this if necessary, TODO improve on this later, for now just raise an error
                # self.create_detector_quality_map()
                raise FileNotFoundError(
                    f"There seem to be no detector quality file for this trigger with observation ID"
                    f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            elif len(self.detector_quality_file) > 1:
                raise ValueError(
                    f"There seem to be more than one detector quality file for this trigger with observation ID "
                    f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            else:
                self.detector_quality_file = self.detector_quality_file[0]

            # get the relevant information from the event file/TDRSS file

            # see if the event data has been energy calibrated
            if verbose:
                print("Checking to see if the event file has been energy calibrated.")

            # look at the header of the event file(s) and see if they have:
            # GAINAPP =                 T / Gain correction has been applied
            # and GAINMETH= 'FIXEDDAC'           / Cubic ground gain/offset correction using DAC-b
            for f in ev_file:
                with fits.open(f) as file:
                    hdr = file["EVENTS"].header
                if not hdr["GAINAPP"] or "FIXEDDAC" not in hdr["GAINMETH"]:
                    # need to run the energy conversion even though this should have been done by SDC
                    self.apply_energy_correction(f, verbose)

            # if we will be doing spectra/light curves we need to do the mask weighting. This may be done by the SDC already.
            # If the SDC already did this, there will be B
            # if not, the user can specify these values in the tdrss file

    def create_detector_quality_map(self):
        """
        This function creates a detector quality mask following the steps outlined here:
        https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html

        The resulting quality mask is placed in the bat/hk/directory with the appropriate observation ID and code=bdqcb

        This should be taken care of by the SDC but this funciton will document how this can be done incase a detector
        quality mask has not been created.

        :return: Path object to the detector quality mask
        """

        return None

    def apply_energy_correction(self, ev_file, verbose):
        """
        This function applies the proper energy correction to the event file following the steps outlined here:
        https://swift.gsfc.nasa.gov/analysis/threads/bateconvertthread.html

        This should be able to apply the energy correciton if needed (if the SDC didnt do this), which may entail figuring
        out how to get the relevant gain/offset file that is closest in time to the event data.

        If this needs to be done, the event files also need to be unzipped if they are zipped since the energy correction
        occurs in the event file itself.

        For now, the funciton just checks to see if there is a gain/offset file to do the energy correction and raises an error
        if the event file hasnt been energy corrected.

        :return:
        """

        # see if we have a gain/offset map
        if len(self.gain_offset_file) < 1:
            if verbose:
                print(
                    f"There seem to be no gain/offset file for this trigger with observation ID \
            {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing if an"
                    f"energy calibration needs to be applied."
                )
            # need to create this gain/offset file or get it somehow

        raise AttributeError(
            f"The event file {ev_file} has not had the energy calibration applied and there is no gain/offset "
            f"file for this trigger with observation ID \
            {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing since an"
            f"energy calibration needs to be applied."
        )

        return None
