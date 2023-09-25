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

    def __init__(self, obs_id, transient_name=None, ra="event", dec="event", obs_dir=None, input_dict=None, recalc=False, verbose=False, load_dir=None):

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
        self._set_local_pfile_dir = load_dir.joinpath(".local_pfile")

        #THIS SHOULDNT BE NECESSARY NOW WITH THE BATOBSERVATION GET/SET
        # make the local pfile dir if it doesnt exist and set this value
        #self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        #try:
        #    hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        #except AttributeError:
        #    hsp.utils.local_pfiles(par_dir=str(self._local_pfile_dir))

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
            self.enable_disable_file=list(self.obs_dir.joinpath("bat").joinpath("hk").glob('*bdecb*'))
            #the detector quality is combination of enable/disable detectors and currently (at tiem of trigger) hot detectors
            # https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html
            self.detector_quality_file=list(self.obs_dir.joinpath("bat").joinpath("hk").glob('*bdqcb*'))
            self.event_files=list(self.obs_dir.joinpath("bat").joinpath("event").glob('*bevsh*_uf*'))
            self.attitude_file=list(self.obs_dir.joinpath("auxil").glob('*sat.*'))
            self.tdrss_files=list(self.obs_dir.joinpath("tdrss").glob('*msb*.fits*'))
            self.gain_offset_file=list(self.obs_dir.joinpath("bat").joinpath("hk").glob('*bgocb*'))
            self.auxil_raytracing_file=list(self.obs_dir.joinpath("bat").joinpath("event").glob('*evtr*'))


            #make sure that there is only 1 attitude file
            if len(self.attitude_file)>1:
                raise ValueError(f"There seem to be more than one attitude file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            elif len(self.attitude_file) < 1:
                raise ValueError(f"There seem to be no attitude file for this trigger with observation ID \
                                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                self.attitude_file=self.attitude_file[0]

            #make sure that there is at least one event file
            if len(self.event_files)<1:
                raise FileNotFoundError(f"There seem to be no event files for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                self.event_files=self.event_files[0]


            #make sure that we have an enable disable map
            if len(self.enable_disable_file) < 1:
                raise FileNotFoundError(f"There seem to be no detector enable/disable file for this trigger with observation "
                                        f"ID {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            elif len(self.enable_disable_file) > 1:
                raise ValueError(f"There seem to be more than one detector enable/disable file for this trigger with observation ID "
                                 f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                self.enable_disable_file=self.enable_disable_file[0]

            #make sure that we have a detector quality map
            if len(self.detector_quality_file) < 1:
                if verbose:
                    print(f"There seem to be no detector quality file for this trigger with observation ID" \
                f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")

                #need to create this map can get to this if necessary, TODO improve on this later, for now just raise an error
                #self.detector_quality_file = self.create_detector_quality_map()
                raise FileNotFoundError(f"There seem to be no detector quality file for this trigger with observation ID" \
                                f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            elif len(self.detector_quality_file) > 1:
                raise ValueError(
                    f"There seem to be more than one detector quality file for this trigger with observation ID "
                    f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                self.detector_quality_file=self.detector_quality_file[0]

            # if we will be doing spectra/light curves we need to do the mask weighting. This may be done by the SDC already.
            # If the SDC already did this, there will be BAT_RA and BAT_DEC header keywords in the event file(s)
            # if not, the user can specify these values in the tdrss file or just pass them to this constructor
            # TODO: possible feature here is to be able to do mask weighting for multiple sources in the BAT FOV at the time
            # of the event data being collected.

            #get the relevant information from the event file/TDRSS file such as RA/DEC/trigger time. Should also make
            # sure that these values agree. If so good, otherwise need to choose a coordinate/use the user supplied coordinates
            # and then rerun the auxil ray tracing
            tdrss_centroid_file=[i for i in self.tdrss_files if "msbce" in str(i)]
            #get the tdrss coordinates if the file exists
            if len(tdrss_centroid_file) > 0:
                with fits.open(tdrss_centroid_file[0]) as file:
                    tdrss_ra = file[0].header["BRA_OBJ"]
                    tdrss_dec = file[0].header["BDEC_OBJ"]

            #get info from event file which must exist to get to this point
            with fits.open(self.event_files[0]) as file:
                event_ra = file[0].header["BAT_RA"]
                event_dec = file[0].header["BAT_DEC"]

            #by default, ra/dec="event" to use the coordinates set here by SDC but can use other coordinates
            if "tdrss" in ra or "tdrss" in dec:
                #use the TDRSS message value
                self.ra=tdrss_ra
                self.dec=tdrss_dec
            elif "event" in ra or "event" in dec:
                #use the event file RA/DEC
                self.ra=event_ra
                self.dec=event_dec
            else:
                if np.isreal(ra) and np.isreal(dec):
                    self.ra=ra
                    self.dec=dec
                else:
                    #the ra/dec values must be decimal degrees for the following analysis to work
                    raise ValueError(f"The passed values of ra and dec are not decimal degrees. Please set these to appropriate values.")

            #see if the RA/DEC that the user wants to use is what is in the event file
            # if not, then we need to do the mask weighting again
            coord_match=(self.ra == event_ra) and (self.dec == event_dec)

            #make sure that we have our auxiliary ray tracing file in order to do spectral fitting of the burst
            #also need to check of the coordinates we want are what is in the event file.
            if len(self.auxil_raytracing_file) < 1 or not coord_match:
                if verbose:
                    print(f"There seem to be no auxiliary ray tracing file for this trigger with observation ID" \
                f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")

                #need to create this map can get to this if necessary,
                #TODO: improve on this later, for now just raise an error
                #TODO: improvement will be that when BAT is slewing that this file will need to be remade for each time interval
                # and will also have to consider drmgen and mask weighing for each timestep

                #self.auxil_raytracing_file = self.apply_mask_weighting()
                raise FileNotFoundError(f"There seem to be no auxiliary ray tracing file for this trigger with observation ID" \
                                f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            elif len(self.auxil_raytracing_file) > 1:
                raise ValueError(
                    f"There seem to be more than one auxiliary ray tracing file for this trigger with observation ID "
                    f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                self.auxil_raytracing_file=self.auxil_raytracing_file[0]

            #see if the event data has been energy calibrated
            if verbose:
                print('Checking to see if the event file has been energy calibrated.')

            #look at the header of the event file(s) and see if they have:
            # GAINAPP =                 T / Gain correction has been applied
            # and GAINMETH= 'FIXEDDAC'           / Cubic ground gain/offset correction using DAC-b
            for f in self.event_files:
                with fits.open(f) as file:
                    hdr=file['EVENTS'].header
                if not hdr["GAINAPP"] or  "FIXEDDAC" not in hdr["GAINMETH"]:
                    #need to run the energy conversion even though this should have been done by SDC
                    self.apply_energy_correction(f, verbose)



            # at this point, we have set some things up and we can let the user define what they want to do for their light
            # curves and spctra

        else:
            load_file = Path(load_file).expanduser().resolve()
            self.load(load_file)


    def load(self, f):
        """
        Loads a saved BatEvent object
        :param f: String of the file that contains the previously saved BatSurvey object
        :return: None
        """
        with open(f, 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        self.__dict__.update(content)

    def save(self):
        """
        Saves the current BatEvent object
        :return: None
        """
        file = self.result_dir.joinpath('batevent.pickle')  # os.path.join(self.result_dir, "batsurvey.pickle")
        with open(file, 'wb') as f:
            pickle.dump(self.__dict__, f, 2)
        print("A save file has been written to %s." % (str(file)))

    def create_detector_quality_map(self):
        """
        This function creates a detector quality mask following the steps outlined here:
        https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html

        The resulting quality mask is placed in the bat/hk/directory with the appropriate observation ID and code=bdqcb

        This should be taken care of by the SDC but this funciton will document how this can be done incase a detector
        quality mask has not been created. Have confirmed that the bat/hk/*bdqcb* file is the same as what is outputted
        by the website linked above

        :return: Path object to the detector quality mask
        """
        try:
            #Create DPI
            #batbinevt bat/event/*bevshsp_uf.evt.gz grb.dpi DPI 0 u - weighted = no outunits = counts
            output_dpi=self.obs_dir.joinpath("bat").joinpath("hk").joinpath('detector_quality.dpi')
            input_dict=dict(infile=str(self.event_files[0]), outfile=str(output_dpi),
                            outtype="DPI", timedel=0.0, timebinalg = "uniform", energybins = "-", weighted = "no", outunits = "counts")
            binevt_return=self._call_batbinevt(input_dict)

            #Get list of known problematic detectors
            #bathotpix grb.dpi grb.mask detmask = bat/hk/sw01116441000bdecb.hk.gz
            #get the

            output_detector_quality=self.obs_dir.joinpath("bat").joinpath("hk").joinpath(f'sw{self.obs_id}bdqcb.hk.gz')
            input_dict = dict(infile=str(output_dpi),
                              outfile=str(output_detector_quality),
                              detmask = str(self.enable_disable_file)
                              )
            hotpix_return=self._call_bathotpix(input_dict)

            self.detector_quality_file=output_detector_quality
        except Exception as e:
            print(e)
            raise RuntimeError("There was a runtime error in either batbinevt or bathotpix while creating th detector quality mask.")

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
                print(f"There seem to be no gain/offset file for this trigger with observation ID \
            {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing if an"
                      f"energy calibration needs to be applied.")
            # need to create this gain/offset file or get it somehow

            raise AttributeError(f'The event file {ev_file} has not had the energy calibration applied and there is no gain/offset '
                                     f'file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing since an' \
                          f"energy calibration needs to be applied.")
        else:
            #if we do, then we need to call bateconvert
            input_dict=dict(infile=str(self.event_files), calfile=str(self.gain_offset_file),
                            residfile="CALDB", pulserfile="CALDB", fltpulserfile="CALDB")
            self._call_bateconvert(input_dict)

        return None

    def apply_mask_weighting(self):
        """
        This method is meant to apply mask weighting for a source that is located at a certain position on the sky.
        An associated, necessary file that is produced is the auxiliary ray tracing file which is needed for spectral fitting.

        Note that it modifies the event file and the event file needs to be uncompressed.

        :return:
        """

        #batmaskwtevt infile=bat/event/sw01116441000bevshsp_uf.evt attitude=auxil/sw01116441000sat.fits.gz detmask=grb.mask ra= dec=

        raise NotImplementedError("Applying the mask weighing has not yet been implemented.")

        return None

    def create_lightcurve(self, **kwargs):
        """
        This method returns a lightcurve object.

        :return:
        """

        raise NotImplementedError("Creating the lightcurve has not yet been implemented.")

        return None

    def create_spectrum(self, **kwargs):
        """
        This method returns a spectrum object.

        :param kwargs:
        :return:
        """

        raise NotImplementedError("Creating a spectrum has not yet been implemented.")

        return None

    def create_DPI(self, **kwargs):
        """
        This method creates and returns a detector plane image.

        :param kwargs:
        :return:
        """

        raise NotImplementedError("Creating the DPI has not yet been implemented.")

        return None

    def create_sky_image(self, **kwargs):
        """
        This method returns a sky image

        :param kwargs:
        :return:
        """

        raise NotImplementedError("Creating the sky image has not yet been implemented.")

        return None


