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

    def __init__(self, obs_id, transient_name=None, obs_dir=None, input_dict=None, recalc=False, verbose=False, load_dir=None):

        # make sure that the observation ID is a string
        if type(obs_id) is not str:
            obs_id = f"{int(obs_id)}"

        # initialize super class
        super().__init__(obs_id, obs_dir)

        # See if a loadfile exists, if we dont want to recalcualte everything, otherwise remove any load file and
        # .batsurveycomplete file (this is produced only if the batsurvey calculation was completely finished, and thus
        # know that we can safely load the batsurvey.pickle file)
        if not recalc and load_dir is None:
            load_dir = sorted(self.obs_dir.parent.glob(obs_id + '_event*'))

            # see if there are any _surveyresult dir or anything otherwise just use obs_dir as a place holder
            if len(load_dir) > 0:
                load_dir = load_dir[0]
            else:
                load_dir = self.obs_dir
        elif not recalc and load_dir is not None:
            load_dir_test = sorted(Path(load_dir).glob(obs_id + '_event*'))
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

            if not self.obs_dir.joinpath("bat").joinpath("event").is_dir() or not self.obs_dir.joinpath("bat").joinpath("hk").is_dir() or\
                    not self.obs_dir.joinpath("bat").joinpath("rate").is_dir() or not self.obs_dir.joinpath(
                    "tdrss").is_dir() or not self.obs_dir.joinpath("auxil").is_dir():
                raise ValueError(
                    "The observation ID folder needs to contain the bat/event/, the bat/hk/, the bat/rate/, the auxil/, and tdrss/ subdirectories in order to " + \
                    "analyze BAT event data. One or many of these folders are missing.")

            #save the necessary files that we will need through the processing/analysis steps
            enable_disable_file=list(self.obs_dir.joinpath("bat").joinpath("hk").glob('*bdecb*'))
            #the detector quality is combination of enable/disable detectors and currently (at tiem of trigger) hot detectors
            # https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html
            detector_quality_file=list(self.obs_dir.joinpath("bat").joinpath("hk").glob('*bdqcb*'))
            event_file=list(self.obs_dir.joinpath("bat").joinpath("event").glob('*bevsh*_uf*'))
            attitude_file=list(self.obs_dir.joinpath("auxil").glob('*sat.*'))

            #make sure that there is only 1 attitude file
            if len(attitude_file)>1:
                raise ValueError("There seem to be more than one attitude file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                attitude_file=attitude_file[0]

            #make sure that there is at least one event file
            if len(event_file)<1:
                raise FileNotFoundError("There seem to be no event files for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")

            #make sure that we have an enable disable map
            if len(enable_disable_file) < 1:
                raise FileNotFoundError("There seem to be no detector enable/disable file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")

            #make sure that we have a detector quality map
            if len(detector_quality_file) < 1:
                if verbose:
                    print("There seem to be no detector quality file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
                #need to create this map
                self.create_detector_quality_map()



            if verbose:
                print('Checking to see if the event file has been energy corrected.')

            #look at the header of the event file(s) and see if they have:
            # GAINAPP =                 T / Gain correction has been applied
            # and GAINMETH= 'FIXEDDAC'           / Cubic ground gain/offset correction using DAC-b
            with fits.open(ev_file) as file:
                hdr=file['EVENTS'].header
            if not hdr["GAINAPP"] or  "FIXEDDAC" not in hdr["GAINMETH"]:
                #need to run the energy conversion even though this should have been done by ground software
                raise AttributeError(f'The event file {ev_file} has not had the energy correction applied')







    def create_detector_quality_map(self):
        """
        This function creates a detector quality mask following the steps outlined here:
        https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html

        The resulting quality mask is placed in the bat/hk/directory with the appropriate observation ID and code=bdqcb

        :return: Path object to the detector quality mask
        """



