"""

This file is meant specifically for the object that reads in and processes TTE data.

Tyler Parsotan April 5 2023

"""

import os
import gzip
import shutil
import sys

import batanalysis

from .batlib import datadir, dirtest, met2mjd, met2utc
from .batobservation import BatObservation, Lightcurve
import glob
from astropy.io import fits
import numpy as np
import subprocess
import pickle
import sys
import re
from pathlib import Path
from astropy.time import Time
import astropy.units as u
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

    def __init__(self, obs_id, result_dir=None, transient_name=None, ra="event", dec="event", obs_dir=None, input_dict=None, recalc=False, verbose=False, load_dir=None):

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
        self._set_local_pfile_dir(load_dir.joinpath(".local_pfile"))

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

            #save the necessary files that we will need through the processing/analysis steps. See
            # https://swift.gsfc.nasa.gov/archive/archiveguide1_v2_2_apr2018.pdf for reference of files
            self.enable_disable_file=list(self.obs_dir.joinpath("bat").joinpath("hk").glob('*bdecb*'))
            #the detector quality is combination of enable/disable detectors and currently (at time of trigger) hot detectors
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
                #also make sure that the file is gunzipped
                if ".gz" in self.event_files.suffix:
                    with gzip.open(self.event_files, 'rb') as f_in:
                        with open(self.event_files.parent.joinpath(self.event_files.stem), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    self.event_files=test.parent.joinpath(test.stem)


            #make sure that we have an enable disable map
            if len(self.enable_disable_file) < 1:
                raise FileNotFoundError(f"There seem to be no detector enable/disable file for this trigger with observation "
                                        f"ID {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            elif len(self.enable_disable_file) > 1:
                raise ValueError(f"There seem to be more than one detector enable/disable file for this trigger with observation ID "
                                 f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing.")
            else:
                self.enable_disable_file=self.enable_disable_file[0]

            #make sure that we have gain offset file
            if len(self.gain_offset_file) < 1:
                warnings.warn(f"There seem to be no gain/offset file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing if an"
                      f"energy calibration needs to be applied.")
            elif len(self.gain_offset_file) > 1:
                warnings.warn(f"There seem to be too many gain/offset files for this trigger with observation ID \
                            {self.obs_id} located at {self.obs_dir}. One of these files is necessary for the remaining processing if an"
                              f"energy calibration needs to be applied.")
            else:
                self.gain_offset_file=self.gain_offset_file[0]

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

            #TODO: need to get the GTI? May not need according to software guide?

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
            with fits.open(self.event_files) as file:
                event_ra = file[0].header["RA_OBJ"]
                event_dec = file[0].header["DEC_OBJ"]

            #by default, ra/dec="event" to use the coordinates set here by SDC but can use other coordinates
            if "tdrss" in ra or "tdrss" in dec:
                if len(tdrss_centroid_file) > 0:
                    #use the TDRSS message value
                    self.ra=tdrss_ra
                    self.dec=tdrss_dec
                else:
                    raise ValueError("There is no TDRSS message coordinate. Please create a TDRSS file to use this option.")
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

                #self.auxil_raytracing_file = self.apply_mask_weighting(self.ra, self.dec)
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
            #also want to get the TSTART and TSTOP for use later
            with fits.open(self.event_files) as file:
                hdr=file['EVENTS'].header
                self.tstart_met=hdr["TSTART"]
                self.tstop_met=hdr["TSTOP"]
                self.telapse=hdr["TELAPSE"]
                self.trigtime_met=hdr["TRIGTIME"]
            if not hdr["GAINAPP"] or  "FIXEDDAC" not in hdr["GAINMETH"]:
                #need to run the energy conversion even though this should have been done by SDC
                self.apply_energy_correction(verbose)

            # at this point, we have made sure that the events are energy calibrated, the mask weighting has been done for
            # the coordinates of interest (assuming it is the triggered event)

            # see if the savedir=None, if so set it to the determined load_dir. If the directory doesnt exist create it.
            if result_dir is None:
                self.result_dir = self.obs_dir.parent.joinpath(f"{obs_id}_eventresult")
            else:
                self.result_dir = Path(result_dir)

            self.result_dir.mkdir(parents=True, exist_ok=True)

            #Now we can create the necessary directories to hold the files in the save_dir directory
            event_dir=self.result_dir.joinpath("events")
            gti_dir=self.result_dir.joinpath("gti")
            auxil_dir=self.result_dir.joinpath("auxil")
            dpi_dir=self.result_dir.joinpath("dpi")
            img_dir=self.result_dir.joinpath("gti")
            lc_dir=self.result_dir.joinpath("lc")
            pha_dir=self.result_dir.joinpath("pha")

            for i in [event_dir, gti_dir, auxil_dir, dpi_dir, img_dir, lc_dir, pha_dir]:
                i.mkdir(parents=True, exist_ok=True)

            #copy the necessary files over, eg the event file, the quality mask, the attitude file, etc
            shutil.copy(self.event_files, event_dir)
            shutil.copy(self.auxil_raytracing_file, event_dir)

            shutil.copy(self.enable_disable_file, auxil_dir)
            shutil.copy(self.detector_quality_file, auxil_dir)
            shutil.copy(self.attitude_file, auxil_dir)
            #move all tdrss files for reference
            for i in self.tdrss_files:
                shutil.copy(i, auxil_dir)

            shutil.copy(self.gain_offset_file, auxil_dir)

            #save the new location of the files as attributes
            self.event_files = event_dir.joinpath(self.event_files.name)
            self.auxil_raytracing_file = event_dir.joinpath(self.auxil_raytracing_file.name)
            self.enable_disable_file = auxil_dir.joinpath(self.enable_disable_file.name)
            self.detector_quality_file = auxil_dir.joinpath(self.detector_quality_file.name)
            self.attitude_file = auxil_dir.joinpath(self.attitude_file.name)
            # change paths for all tdrss files
            temp_tdrss_files=[]
            for i in self.tdrss_files:
                temp_tdrss_files.append(auxil_dir.joinpath(i.name))
            self.tdrss_files=temp_tdrss_files
            self.gain_offset_file = auxil_dir.joinpath(self.gain_offset_file.name)

            #also update the local pfile dir
            self._set_local_pfile_dir(self.result_dir.joinpath(".local_pfile"))

            #want to get some other basic information for use later, including all the photon data
            self._parse_event_file()

            # create the marker file that tells us that the __init__ method completed successfully
            complete_file.touch()

            # save the state so we can load things later
            self.save()

            # Now we can let the user define what they want to do for their light
            # curves and spctra. Need to determine how to organize this for any source in FOV to be analyzed.



        else:
            load_file = Path(load_file).expanduser().resolve()
            self.load(load_file)

    def _parse_event_file(self):
        """
        This funciton reads in the data from the event file
        :return:
        """

        self.data = {}
        with fits.open(self.event_files) as file:
            data=file[1].data
            for i in data.columns:
                self.data[i.name] = u.Quantity(data[i.name], i.unit)

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

            #Get list of known problematic detectors, do we need to do this? This might be handled by the SDC
            #eg batdetmask date=output_dpi outfile=master.detmask clobber=YES detmask= self.enable_disable_file
            #then master.detmask gets passed as detmask parameter in bathotpix call

            #get the hot pixels
            #bathotpix grb.dpi grb.mask detmask = bat/hk/sw01116441000bdecb.hk.gz
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

    def apply_energy_correction(self, verbose):
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
        elif len(self.gain_offset_file) > 1:
            raise AttributeError(
                f'The event file {ev_file} has not had the energy calibration applied and there are too many gain/offset '
                f'files for this trigger with observation ID \
                            {self.obs_id} located at {self.obs_dir}. One of these files is necessary for the remaining processing since an' \
                f"energy calibration needs to be applied.")
        else:
            #if we have the file, then we need to call bateconvert
            input_dict=dict(infile=str(self.event_files), calfile=str(self.gain_offset_file),
                            residfile="CALDB", pulserfile="CALDB", fltpulserfile="CALDB")
            self._call_bateconvert(input_dict)

        return None

    def apply_mask_weighting(self, ra=None, dec=None):
        """
        This method is meant to apply mask weighting for a source that is located at a certain position on the sky.
        An associated, necessary file that is produced is the auxiliary ray tracing file which is needed for spectral fitting.

        Note that it modifies the event file and the event file needs to be uncompressed.
        Note the event file RA_OBJ and DEC_OBJ header values are not modified with a call to batmaskwtevt

        :return:
        """

        #batmaskwtevt infile=bat/event/sw01116441000bevshsp_uf.evt attitude=auxil/sw01116441000sat.fits.gz detmask=grb.mask ra= dec=
        if ra is None and dec is None:
            ra=self.ra
            dec=self.dec
        else:
            #set the new ra/dec values
            self.ra=ra
            self.dec=dec

        input_dict=dict(infile=str(self.event_files), attitude=str(self.attitude_file), detmask=str(self.detector_quality_file),
                        ra=ra, dec=dec, auxfile=str(self.auxil_raytracing_file), clobber="YES")
        self._call_batmaskwtevt(input_dict)

        #modify the event file header with the RA/DEC of the weights that were applied, if they are different
        with fits.open(self.event_files, mode='update') as file:
            event_ra = file[0].header["RA_OBJ"]
            event_dec = file[0].header["DEC_OBJ"]
            if event_ra != self.ra or event_dec != self.dec:
                #update the event file RA/DEC_OBJ values everywhere
                for i in file:
                    i.header["RA_OBJ"]=self.ra
                    i.header["DEC_OBJ"]=self.dec
                file.flush()

        #reread in the event file data
        self._parse_event_file()

        #TODO how to handle a different auxiliary ray tracing file bieng produced here? what is there is none before?
        #how do we handle the name of the file?

        return None

    def create_lightcurve(self, lc_file=None, timedelta=np.timedelta64(64, 'ms'), tstart=None, tstop=None,
                          energybins=["15-25", "25-50", "50-100", "100-350", "15-350"], recalc=True, mask_weighting=True,
                          timebinalg="uniform"):
        """
        This method returns a lightcurve object which can be manipulated in different energies/timebins

        :return:
        """

        #raise NotImplementedError("Creating the lightcurve has not yet been implemented.")

        #batbinevt infile=sw00145675000bevshsp_uf.evt.gz outfile=onesec.lc outtype=LC
        # timedel=1.0 timebinalg=u energybins=15-150
        # detmask=../hk/sw00145675000bcbdq.hk.gz clobber=YES

        if lc_file is None:
            if not recalc:
                #make up a name for the light curve that hasnt been used already in the LC directory
                lc_files=list(self.result_dir.joinpath("lc").glob("*.lc"))
                base="lightcurve_"
                count=0
                while f"{base}{count}.lc" in lc_files:
                    count+=1
                lc_file=self.result_dir.joinpath("lc").joinpath(f"{base}{count}.lc")
            else:
                lc_files = list(self.result_dir.joinpath("lc").glob("*.lc"))
                if len(lc_files)==1:
                    outfile=lc_files[0]
                else:
                    raise ValueError(f"There are too many files which meet the criteria to be loaded. Please specify one of {lc_files}.")
        else:
            lc_file=Path(lc_file)


        if recalc or not lc_file.exists():
            # create a general light curve to modify or load one that was previously created
            input_dict = dict(infile=str(self.event_files), outfile=str(lc_file), outtype="LC",
                              energybins="15-350", weighted="YES", timedel=0.064,
                              detmask=str(self.detector_quality_file),
                              tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")
            self._call_batbinevt(input_dict)
            lc = Lightcurve(self.event_files, lc_file, self.detector_quality_file)
        else:
            lc=Lightcurve(self.event_files, lc_file, self.detector_quality_file)

        stop


        # error checking for times
        if type(timedelta) is not np.timedelta64:
            raise ValueError('The timedelta variable needs to be a numpy timedelta64 object.')
        else:
            timedel= timedelta / np.timedelta64(1, 's') #convert to seconds

        if tstart is not None:
            #test if its a number
            if type(tstart) is not TimeDelta:
                tstart=str(tstart)
                try:
                    tstart=float(tstart)
                except ValueError as e:
                    raise ValueError('The tstart variable needs to be an MET time float, a string of an MET time value, or a TimeDelta object.')
            else:
                tstart=self.trigtime_met+tstart.to_value("sec")
        else:
            #get the start time from the earliest MET time in the event file using default batbinevt values
            tstart="INDEF"

        if tstop is not None:
            #test if its a number
            if type(tstop) is not TimeDelta:
                tstop=str(tstop)
                try:
                    tstop=float(tstop)
                except ValueError as e:
                    raise ValueError('The tstop variable needs to be an MET time float, a string of an MET time value, or a TimeDelta object.')
            else:
                tstop=self.trigtime_met+tstop.to_value("sec")
        else:
            #get the end time from the latest MET time in the event file using default batbinevt values
            tstop="INDEF"

        #error checking for timebinalg
        if timebinalg not in ["uniform", "gti", "snr", "bayesian"]:
            raise ValueError("The timebinalg must be set to uniform, gti, snr, or bayesian.")

        # if we want custom intervals for binning:
        if "gti" in timebinalg:
            #create the gti file and feed that in
            times=np.arange(tstart, tstop, timedelta)

        else:
            gtifile = "NONE"

        #if we want bayesian blocks we need to first create a light curve and then run it though the bayesian blocks
        # algorithm and return that
        if "bayesian" in timebinalg:
            timebinalg="uniform"
            bayesian_reanalyze=True




        #error checking for energies
        if type(energybins) is not list:
            energybins=list(energybins)
            energybins=','.join(energybins)

        #error checking for weighting
        if type(mask_weighting) is bool:
            if mask_weighting:
                weighted = "YES"
            else:
                weighted = "NO"
        else:
            raise ValueError("The mask_weighting parameter should be a boolean value.")

        input_dict=dict(infile=str(self.event_files), outfile=str(outfile), outtype="LC",
                        energybins=energybins, weighted = weighted, timedel=timedel, detmask=str(self.detector_quality_file),
                        tstart=tstart, tstop=tstop, )



        return None

    def create_pha(self, **kwargs):
        """
        This method returns a spectrum object.

        TODO: apply keyword correction for spectrum file (using the auxfile) via batupdatephakw
        TODO: apply systematic error for spectrum file  via batphasyserr

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


