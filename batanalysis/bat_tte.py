"""

This file is meant specifically for the object that reads in and processes TTE data.

Tyler Parsotan April 5 2023

"""

import gzip
import pickle
import shutil
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.io import fits
from swifttools.swift_too import Clock

from .attitude import Attitude
from .bat_dph import BatDPH
from .bat_dpi import BatDPI
from .bat_skyview import BatSkyView
from .batobservation import BatObservation
from .batproducts import Lightcurve, Spectrum
from .tte_data import TimeTaggedEvents

# for python>3.6
try:
    import heasoftpy.swift as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatEvent(BatObservation):
    def __init__(
            self,
            obs_id,
            result_dir=None,
            transient_name=None,
            ra="event",
            dec="event",
            obs_dir=None,
            input_dict=None,
            recalc=False,
            verbose=False,
            load_dir=None,
            is_guano=False
    ):
        # make sure that the observation ID is a string
        if type(obs_id) is not str:
            obs_id = f"{int(obs_id)}"

        # initialize super class
        super(BatEvent, self).__init__(obs_id, obs_dir)

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

        # stop
        load_file = load_dir.joinpath("batevent.pickle")
        complete_file = load_dir.joinpath(".batevent_complete")
        self._set_local_pfile_dir(load_dir.joinpath(".local_pfile"))

        # if the user wants to recalculate things or if there is no batevent.pickle file, or if there is no
        # .batevent_complete file (meaning that the __init__ method didnt complete)
        if recalc or not load_file.exists() or not complete_file.exists():
            if is_guano:
                check = (
                        not self.obs_dir.joinpath("bat").joinpath("event").is_dir()
                        or not self.obs_dir.joinpath("bat").joinpath("hk").is_dir()
                        # or not self.obs_dir.joinpath("bat").joinpath("rate").is_dir()  # this may not really be needed
                        or not self.obs_dir.joinpath("auxil").is_dir()
                )
                if check:
                    raise ValueError(
                        "The observation ID folder needs to contain the bat/event/, the bat/hk/, the bat/rate/, and the auxil/ subdirectories in order to "
                        + "analyze BAT guano event data. One or many of these folders are missing."
                    )

            else:
                check = (
                        not self.obs_dir.joinpath("bat").joinpath("event").is_dir()
                        or not self.obs_dir.joinpath("bat").joinpath("hk").is_dir()
                        or not self.obs_dir.joinpath("bat").joinpath("rate").is_dir()
                        or not self.obs_dir.joinpath("tdrss").is_dir()
                        or not self.obs_dir.joinpath("auxil").is_dir()
                )
                if check:
                    raise ValueError(
                        "The observation ID folder needs to contain the bat/event/, the bat/hk/, the bat/rate/, the auxil/, and tdrss/ subdirectories in order to "
                        + "analyze BAT event data. One or many of these folders are missing."
                    )

            # save the necessary files that we will need through the processing/analysis steps. See
            # https://swift.gsfc.nasa.gov/archive/archiveguide1_v2_2_apr2018.pdf for reference of files
            self.enable_disable_file = list(
                self.obs_dir.joinpath("bat").joinpath("hk").glob("*bdecb*")
            )
            # the detector quality is combination of enable/disable detectors and currently (at time of trigger) hot detectors
            # https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html
            self.detector_quality_file = list(
                self.obs_dir.joinpath("bat").joinpath("hk").glob("*bdqcb*")
            )

            # if we have previously loaded in an obsid with 1 event file, then we probably gunzipped it. This
            # would lead us to doubly list the event files. We want to filter out all the gunzipped ones
            self.event_files = sorted(
                list(self.obs_dir.joinpath("bat").joinpath("event").glob("*bev*_uf*.gz"))
            )
            self.attitude_file = list(self.obs_dir.joinpath("auxil").glob("*sat.*"))
            self.tdrss_files = list(self.obs_dir.joinpath("tdrss").glob("*msb*.fits*"))
            self.gain_offset_file = list(
                self.obs_dir.joinpath("bat").joinpath("hk").glob("*bgocb*")
            )
            self.auxil_raytracing_file = list(
                self.obs_dir.joinpath("bat").joinpath("event").glob("*evtr*")
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
            else:
                if len(self.event_files) > 1:

                    # merge the files and make sure all events are unique with no duplicates
                    concat_eventfile = self.event_files[0].parent.joinpath("total_events.evt")
                    if not concat_eventfile.exists():
                        total_event = TimeTaggedEvents.concatenate_event(*self.event_files,
                                                                         output_event_file=concat_eventfile)
                        self.event_files = total_event
                    else:
                        self.event_files = concat_eventfile
                else:
                    self.event_files = self.event_files[0]

                # also make sure that the file is gunzipped
                if ".gz" in self.event_files.suffix:
                    with gzip.open(self.event_files, "rb") as f_in:
                        with open(
                                self.event_files.parent.joinpath(self.event_files.stem),
                                "wb",
                        ) as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    self.event_files = self.event_files.parent.joinpath(
                        self.event_files.stem
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

            # make sure that we have gain offset file
            if len(self.gain_offset_file) < 1:
                warnings.warn(
                    f"There seem to be no gain/offset file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing if an"
                    f"energy calibration needs to be applied."
                )
            elif len(self.gain_offset_file) > 1:
                warnings.warn(
                    f"There seem to be too many gain/offset files for this trigger with observation ID \
                            {self.obs_id} located at {self.obs_dir}. One of these files is necessary for the remaining processing if an"
                    f"energy calibration needs to be applied."
                )
            else:
                self.gain_offset_file = self.gain_offset_file[0]

            # make sure that we have a detector quality map
            if len(self.detector_quality_file) < 1:
                if verbose:
                    print(
                        f"There seem to be no detector quality file for this trigger with observation ID"
                        f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                    )

                # need to create this map can get to this if necessary
                self.create_detector_quality_map()
            elif len(self.detector_quality_file) > 1:
                raise ValueError(
                    f"There seem to be more than one detector quality file for this trigger with observation ID "
                    f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                )
            else:
                self.detector_quality_file = self.detector_quality_file[0]

            # if we will be doing spectra/light curves we need to do the mask weighting. This may be done by the SDC already.
            # If the SDC already did this, there will be BAT_RA and BAT_DEC header keywords in the event file(s)
            # if not, the user can specify these values in the tdrss file or just pass them to this constructor
            # TODO: possible feature here is to be able to do mask weighting for multiple sources in the BAT FOV at the time
            # of the event data being collected.

            # TODO: need to get the GTI? May not need according to software guide?

            # get the relevant information from the event file/TDRSS file such as RA/DEC/trigger time. Should also make
            # sure that these values agree. If so good, otherwise need to choose a coordinate/use the user supplied coordinates
            # and then rerun the auxil ray tracing
            tdrss_centroid_file = [i for i in self.tdrss_files if "msbce" in str(i)]
            # get the tdrss coordinates if the file exists
            if len(tdrss_centroid_file) > 0:
                with fits.open(tdrss_centroid_file[0]) as file:
                    if "deg" in file[0].header.comments["BRA_OBJ"]:
                        tdrss_ra = file[0].header["BRA_OBJ"] * u.deg
                        tdrss_dec = file[0].header["BDEC_OBJ"] * u.deg
                    else:
                        raise ValueError(
                            "The TDRSS msbce file BRA/BDEC_OBJ does not seem to be in the units of decimal degrees which is not supported.")

            # by default, ra/dec="event" to use the coordinates set here by SDC but can use other coordinates

            # set these by default so we can do coordinate comparison after this if-else statement
            event_ra = None
            event_dec = None
            if "tdrss" in ra or "tdrss" in dec:
                if len(tdrss_centroid_file) > 0:
                    # use the TDRSS message value
                    self.ra = tdrss_ra
                    self.dec = tdrss_dec
                else:
                    raise ValueError(
                        "There is no TDRSS message coordinate. Please create a TDRSS file to use this option."
                    )
            elif "event" in ra or "event" in dec:
                # get info from event file which must exist to get to this point,
                # for failed trigger TTE data, there is no RA_OBJ/DEC_OBJ keywords in the file so put this in a try except
                # and set self.ra/dec to None.
                try:
                    with fits.open(self.event_files) as file:
                        if "deg" in file[0].header.comments["RA_OBJ"]:
                            event_ra = file[0].header["RA_OBJ"] * u.deg
                            event_dec = file[0].header["DEC_OBJ"] * u.deg
                        else:
                            raise ValueError(
                                "The event file RA/DEC_OBJ does not seem to be in the units of decimal degrees which is not supported.")

                    # use the event file RA/DEC
                    self.ra = event_ra
                    self.dec = event_dec
                    if is_guano:
                        warnings.warn(
                            f"Since this is a GUANO dataset the RA/Dec coordinates may not be valid. These are currently being set to"
                            f"({self.ra}, {self.dec}). Please verify that these are correct for your analysis."
                        )
                except KeyError as e:
                    # get around the quantity_input wrapper for these properties
                    self._ra = None
                    self._dec = None
            else:
                if isinstance(ra, u.Quantity) and isinstance(dec, u.Quantity):
                    self.ra = ra
                    self.dec = dec
                else:
                    # the ra/dec values must be decimal degrees for the following analysis to work
                    raise ValueError(
                        f"The passed values of ra and dec are not astropy unit quantities. Please set these to appropriate values."
                    )

            # see if the RA/DEC that the user wants to use is what is in the event file
            # if not, then we need to do the mask weighting again. If self.ra/dec is None, then lets just skip this
            # since the user is most likely using failed trigger TTE data
            if self.ra is not None and self.dec is not None:
                coord_match = (self.ra == event_ra) and (self.dec == event_dec)

                # make sure that we have our auxiliary ray tracing file in order to do spectral fitting of the burst
                # also need to check of the coordinates we want are what is in the event file.
                if len(self.auxil_raytracing_file) < 1 or not coord_match:
                    if verbose:
                        print(
                            f"There seem to be no auxiliary ray tracing file for this trigger with observation ID "
                            f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining "
                            f"processing."
                        )

                    # set the default auxil ray tracing attribute to None for recreation in the apply_mask_weighting method
                    self.auxil_raytracing_file = None

                    # need to create this map can get to this if necessary,
                    self.apply_mask_weighting(self.ra, self.dec)
                elif len(self.auxil_raytracing_file) > 1:
                    raise ValueError(
                        f"There seem to be more than one auxiliary ray tracing file for this trigger with observation ID "
                        f"{self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing."
                    )
                else:
                    self.auxil_raytracing_file = self.auxil_raytracing_file[0]
            else:
                self.auxil_raytracing_file = None

            # see if the event data has been energy calibrated
            if verbose:
                print("Checking to see if the event file has been energy calibrated.")

            # look at the header of the event file(s) and see if they have:
            # GAINAPP =                 T / Gain correction has been applied
            # and GAINMETH= 'FIXEDDAC'           / Cubic ground gain/offset correction using DAC-b
            # also want to get the TSTART and TSTOP for use later
            with fits.open(self.event_files) as file:
                hdr = file["EVENTS"].header
                self.tstart_met = hdr["TSTART"]
                self.tstop_met = hdr["TSTOP"]
                self.telapse = hdr["TELAPSE"]
                # if we dont have a guano TTE dataset or a failed trigger dataset then this keyword should not exist,
                # though it is possible that it does. We can also have swifttime issues with connecting with the server
                # to do the conversion so still want to set self.trigtime as None and warn about the lack of connection
                # if not is_guano:
                try:
                    self.trigtime = Clock(met=hdr["TRIGTIME"])
                except KeyError as e:
                    # guano data/failed trigger has no trigger time
                    self.trigtime = None
                except requests.exceptions.ConnectionError as e:
                    self.trigtime = None
                    warnings.warn(f"Clock conversion was not possible: {e}")

            if not hdr["GAINAPP"] or "FIXEDDAC" not in hdr["GAINMETH"]:
                # need to run the energy conversion even though this should have been done by SDC
                self.apply_energy_correction(verbose)

            # at this point, we have made sure that the events are energy calibrated, the mask weighting has been done for
            # the coordinates of interest (assuming it is the triggered event)

            # see if the savedir=None, if so set it to the determined load_dir. If the directory doesnt exist create it.
            if result_dir is None:
                self.result_dir = self.obs_dir.parent.joinpath(f"{obs_id}_eventresult")
            else:
                self.result_dir = Path(result_dir)

            self.result_dir.mkdir(parents=True, exist_ok=True)

            # Now we can create the necessary directories to hold the files in the save_dir directory
            event_dir = self.result_dir.joinpath("events")
            gti_dir = self.result_dir.joinpath("gti")
            auxil_dir = self.result_dir.joinpath("auxil")
            dpi_dir = self.result_dir.joinpath("dpi")
            img_dir = self.result_dir.joinpath("gti")
            lc_dir = self.result_dir.joinpath("lc")
            pha_dir = self.result_dir.joinpath("pha")

            for i in [event_dir, gti_dir, auxil_dir, dpi_dir, img_dir, lc_dir, pha_dir]:
                i.mkdir(parents=True, exist_ok=True)

            # copy the necessary files over, eg the event file, the quality mask, the attitude file, etc
            shutil.copy(self.event_files, event_dir)

            if self.auxil_raytracing_file is not None:
                shutil.copy(self.auxil_raytracing_file, event_dir)

            shutil.copy(self.enable_disable_file, auxil_dir)
            shutil.copy(self.detector_quality_file, auxil_dir)
            shutil.copy(self.attitude_file, auxil_dir)
            # move all tdrss files for reference
            for i in self.tdrss_files:
                shutil.copy(i, auxil_dir)

            shutil.copy(self.gain_offset_file, auxil_dir)

            # save the new location of the files as attributes
            self.event_files = event_dir.joinpath(self.event_files.name)

            if self.auxil_raytracing_file is not None:
                self.auxil_raytracing_file = event_dir.joinpath(
                    self.auxil_raytracing_file.name
                )
            self.enable_disable_file = auxil_dir.joinpath(self.enable_disable_file.name)
            self.detector_quality_file = auxil_dir.joinpath(
                self.detector_quality_file.name
            )
            self.attitude_file = auxil_dir.joinpath(self.attitude_file.name)
            # change paths for all tdrss files
            temp_tdrss_files = []
            for i in self.tdrss_files:
                temp_tdrss_files.append(auxil_dir.joinpath(i.name))
            self.tdrss_files = temp_tdrss_files
            self.gain_offset_file = auxil_dir.joinpath(self.gain_offset_file.name)

            # also update the local pfile dir
            self._set_local_pfile_dir(self.result_dir.joinpath(".local_pfile"))

            # want to get some other basic information for use later, including all the photon data
            self._parse_event_file()

            # and the attitude data
            self.attitude = Attitude.from_file(self.attitude_file)

            # create the marker file that tells us that the __init__ method completed successfully
            complete_file = self.result_dir.joinpath(".batevent_complete")
            complete_file.touch()

            # Now we can let the user define what they want to do for their light
            # curves, spctra, etc. Need to determine how to organize this for any source in FOV to be analyzed.

            # initalize the properties to be None
            self.spectra = None
            self.lightcurves = None
            self.dphs = None
            self.dpis = None
            self.skyviews = None

            # save the state so we can load things later
            self.save()

        else:
            load_file = Path(load_file).expanduser().resolve()
            self.load(load_file)

    def _parse_event_file(self):
        """
        This function reads in the data from the event file
        :return: None
        """

        self.data = TimeTaggedEvents.from_file(self.event_files)

    def load(self, f):
        """
        Loads a saved BatEvent object
        :param f: String of the file that contains the previously saved BatSurvey object
        :return: None
        """

        with open(f, "rb") as pickle_file:
            content = pickle.load(pickle_file)
        self.__dict__.update(content)

    def save(self):
        """
        Saves the current BatEvent object
        :return: None
        """
        file = self.result_dir.joinpath(
            "batevent.pickle"
        )
        with open(file, "wb") as f:
            pickle.dump(self.__dict__, f, 2)
        print("A save file has been written to %s." % (str(file)))

    def create_detector_quality_map(self):
        """
        This function creates a detector quality mask following the steps outlined here:
        https://swift.gsfc.nasa.gov/analysis/threads/batqualmapthread.html

        The resulting quality mask is placed in the bat/hk/directory with the appropriate observation ID and code=bdqcb

        This should be taken care of by the SDC but this function will document how this can be done incase a detector
        quality mask has not been created. Have confirmed that the bat/hk/*bdqcb* file is the same as what is outputted
        by the website linked above

        :return: Path object to the detector quality mask
        """
        try:
            # Create DPI
            # batbinevt bat/event/*bevshsp_uf.evt.gz grb.dpi DPI 0 u - weighted = no outunits = counts
            output_dpi = (
                self.obs_dir.joinpath("bat")
                .joinpath("hk")
                .joinpath("detector_quality.dpi")
            )
            input_dict = dict(
                infile=str(self.event_files),
                outfile=str(output_dpi),
                outtype="DPI",
                timedel=0.0,
                timebinalg="uniform",
                energybins="-",
                weighted="no",
                outunits="counts",
                clobber="YES",
            )
            binevt_return = self._call_batbinevt(input_dict)
            if binevt_return.returncode != 0:
                raise RuntimeError(
                    f"The call to Heasoft batbinevt failed with message: {binevt_return.output}"
                )

            # Get list of known problematic detectors
            # eg batdetmask date=output_dpi outfile=master.detmask clobber=YES detmask= self.enable_disable_file
            # then master.detmask gets passed as detmask parameter in bathotpix call
            master_detmask = self.enable_disable_file.parent.joinpath("master.detmask")
            input_dict = dict(
                date=str(output_dpi),
                outfile=str(master_detmask),
                detmask=str(self.enable_disable_file),
                clobber="YES",
            )
            detmask_return = self._call_batdetmask(input_dict)
            if detmask_return.returncode != 0:
                raise RuntimeError(
                    f"The call to Heasoft batdetmask failed with message: {detmask_return.output}"
                )

            # get the hot pixels
            # bathotpix grb.dpi grb.mask detmask = bat/hk/sw01116441000bdecb.hk.gz
            output_detector_quality = (
                self.obs_dir.joinpath("bat")
                .joinpath("hk")
                .joinpath(f"sw{self.obs_id}bdqcb.hk.gz")
            )
            input_dict = dict(
                infile=str(output_dpi),
                outfile=str(output_detector_quality),
                detmask=str(master_detmask),
                clobber="YES",
            )
            hotpix_return = self._call_bathotpix(input_dict)
            if hotpix_return.returncode != 0:
                raise RuntimeError(
                    f"The call to Heasoft bathotpix failed with message: {hotpix_return.output}"
                )

            self.detector_quality_file = output_detector_quality
        except Exception as e:
            print(e)
            raise RuntimeError(
                "There was a runtime error in either batbinevt or bathotpix while creating th detector quality mask."
            )

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

        if type(self.gain_offset_file) is not list:
            go_file = [self.gain_offset_file]
        else:
            go_file = self.gain_offset_file

        # see if we have a gain/offset map
        if len(go_file) < 1:
            if verbose:
                print(
                    f"There seem to be no gain/offset file for this trigger with observation ID \
            {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing if an"
                    f"energy calibration needs to be applied."
                )
            # need to create this gain/offset file or get it somehow

            raise AttributeError(
                f"The event file {self.event_files} has not had the energy calibration applied and there is no gain/offset "
                f"file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing since an"
                f"energy calibration needs to be applied."
            )
        elif len(go_file) > 1:
            raise AttributeError(
                f"The event file {self.event_files} has not had the energy calibration applied and there are too many gain/offset "
                f"files for this trigger with observation ID \
                            {self.obs_id} located at {self.obs_dir}. One of these files is necessary for the remaining processing since an"
                f"energy calibration needs to be applied."
            )
        else:
            # if we have the single file, then we need to call bateconvert
            input_dict = dict(
                infile=str(self.event_files),
                calfile=str(go_file[0]),
                residfile="CALDB",
                pulserfile="CALDB",
                fltpulserfile="CALDB",
            )
            bateconvert_return = self._call_bateconvert(input_dict)

            if bateconvert_return.returncode != 0:
                raise RuntimeError(
                    f"The call to Heasoft bateconvert failed with message: {bateconvert_return.output}"
                )

        return None

    @property
    def trigtime(self):
        """
        The triggertime associated with the event file. If this is specified in the event file then this is loaded in
        as a swiftools Swift Clock object .

        If the event file corresponds to a GUANO data dump or a failed trigger dataset, then this will be None, but a
        user can set the trigtime property by setting it equal to a Swift Clock object.

        """
        return self._trigtime

    @trigtime.setter
    def trigtime(self, value):
        if not isinstance(value, Clock) and value is not None:
            raise ValueError("The trigtime property can only be set to a swifttools Clock object.")

        self._trigtime = value

    @property
    def ra(self):
        """The right ascension of the source and the associated weighting assigned to the event file"""
        return self._ra

    @ra.setter
    @u.quantity_input(value=u.deg)
    def ra(self, value):
        self._ra = value

    @property
    def dec(self):
        """The declination of the source and the associated weighting assigned to the event file"""
        return self._dec

    @dec.setter
    @u.quantity_input(value=u.deg)
    def dec(self, value):
        self._dec = value

    @u.quantity_input(ra=u.deg, dec=u.deg)
    def apply_mask_weighting(self, ra=None, dec=None):
        """
        This method is meant to apply mask weighting for a source that is located at a certain position on the sky.
        An associated, necessary file that is produced is the auxiliary ray tracing file which is needed for spectral fitting.

        Note that it modifies the event file and the event file needs to be uncompressed.
        Note the event file RA_OBJ and DEC_OBJ header values are not modified with a call to batmaskwtevt

        :return:
        """

        # TODO: what to do if self.auxil_raytracing_file has length 0 during init or if we are recreating this file?
        # TODO: create a mask weight image which can also be used for each RA/DEC coordinate and be passed to batbinevt
        #   without a need for reading in the MASK WEIGHT column of the event file

        # batmaskwtevt infile=bat/event/sw01116441000bevshsp_uf.evt attitude=auxil/sw01116441000sat.fits.gz detmask=grb.mask ra= dec=
        if ra is None and dec is None:
            if self.ra is None or self.dec is None:
                raise ValueError("The supplied RA/Dec cannot be None since there is not a default Ra/Dec already set.")

            ra = self.ra.to(u.deg)
            dec = self.dec.to(u.deg)
        else:
            # set the new ra/dec values
            self.ra = ra.to(u.deg)
            self.dec = dec.to(u.deg)

        # if this attribute is None, we need to define it and create it using the standard naming convention
        if self.auxil_raytracing_file is None:
            temp_auxil_raytracing_file = self.event_files.parent.joinpath(
                f"sw{self.obs_id}bevtr.fits"
            )
        else:
            temp_auxil_raytracing_file = self.auxil_raytracing_file

        input_dict = dict(
            infile=str(self.event_files),
            attitude=str(self.attitude_file),
            detmask=str(self.detector_quality_file),
            ra=ra.value,
            dec=dec.value,
            auxfile=str(temp_auxil_raytracing_file),
            clobber="YES",
        )
        batmaskwtevt_return = self._call_batmaskwtevt(input_dict)

        if batmaskwtevt_return.returncode != 0:
            raise RuntimeError(
                f"The call to Heasoft batmaskwtevt failed with message: {batmaskwtevt_return.output}"
            )

        # modify the event file header with the RA/DEC of the weights that were applied, if they are different
        with fits.open(self.event_files, mode="update") as file:
            try:
                if "deg" in file[0].header.comments["RA_OBJ"]:
                    event_ra = file[0].header["RA_OBJ"] * u.deg
                    event_dec = file[0].header["DEC_OBJ"] * u.deg
                else:
                    raise ValueError(
                        "The event file RA/DEC_OBJ does not seem to be in the units of decimal degrees which is not supported.")
            except KeyError as e:
                # we may have a failled trigger  event file and the keyword doesnt exist
                event_dec = None
                event_ra = None

            if event_ra != self.ra or event_dec != self.dec:
                # update the event file RA/DEC_OBJ values everywhere
                for i in file:
                    i.header["RA_OBJ"] = (self.ra.to(u.deg).value, "[deg] R.A. Object")
                    i.header["DEC_OBJ"] = (self.dec.to(u.deg).value, "[deg] Dec Object")

                    # the BAT_RA/BAT_DEC keys have to updated too since this is something
                    # that the software manual points out should be updated
                    i.header["BAT_RA"] = (self.ra.to(u.deg).value, "[deg] Right ascension of source")
                    i.header["BAT_DEC"] = (self.dec.to(u.deg).value, "[deg] Declination of source")

            file.flush()

        # reread in the event file data
        self._parse_event_file()

        # save the file as the attribute if everything else is successful
        self.auxil_raytracing_file = temp_auxil_raytracing_file

        # TODO how to handle a different auxiliary ray tracing file bieng produced here?

        return None

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"], energybins=["energy"])
    def create_lightcurve(
            self,
            lc_file=None,
            timebinalg="uniform",
            timedelta=np.timedelta64(64, "ms"),
            tstart=None,
            tstop=None,
            timebins=None,
            T0=None,
            is_relative=False,
            energybins=[15, 25, 50, 100, 350] * u.keV,
            mask_weighting=True,
            recalc=False,
    ):
        """
        This method returns a lightcurve object which can be manipulated in different energies/timebins. The lightcurve
        path may be provided, which can be a lightcurve that should be loaded (if created already), or the name of the
        lightcurve that will be created with the specified energy/time binning. If no lightcurve file name is provided,
        the method will determine a generic lightcurve name. By default, the lightcurves are saved in the
        OBSID_eventresult/lc/ directory.

        This method allows one to specify different energy/time binnings however since this method returns a Lightcurve
        class, the resulting Lightcurve class instance can be used to rebin the lightcurve however the user wants. The
        lightcurve is also saved to the BatEvent.lightcurve attribute when it is created thorugh this method.

        This method also returns the Lightcurve object.

        Any newly created Lightcurve objects are saved to the lightcurves property where they are stored in order based
        on their creation. If a lightcurve file is loaded in, then the Lightcurve object will not be saved to the
        lightcurves property by default. If a user wants to do so they can set the loaded Lightcurve object to the
        lightcurves property (ie self.lightcurves = loaded_lightcurve).


        :param lc_file: None or a path object of the lightcurve file that will be read in, if previously calculated,
            or the location/name of the new lightcurve file that will contain the newly calculated lightcurve. If set
            to None, the lightcurve filename will be dynamically determined from the other input parameters.
        :param timebinalg: a string that can be set to "uniform", "snr", "highsnr", or "bayesian"
            "uniform" will do a uniform time binning from the specified tmin to tmax with the size of the bin set by
                the timedelta parameter.
            "snr" will bin the lightcurve until a maximum snr threshold is achieved, as is specified by the snrthresh parameter,
                or the width of the timebin becomes the size of timedelta
            "highsnr" will bin the lightcurve with a minimum bin size specified by the timedelta parameter. Longer
                timebin widths will be used if the source is not deteted at the snr level specified by the snrthresh parameter
            "bayesian" will use the battblocks bayesian algorithm to calculate the timebins based off of the energy
                energy integrated lightcurve with 64 ms time binning. Then the lightcurve will be binned in time to the
                tiembins determined by the battblocks algorithm. Using this option also allows for the calculation of
                T90, T50, background time periods, etc if the save_durations parameter =True (more information can
                be found from the battblocks HEASoft documentation).
            NOTE: more information can be found by looking at the HEASoft documentation for batbinevt and battblocks
        :param timedelta: numpy.timedelta64 object denoting the size of the timebinning. This value is used when
            timebinalg is used in the binning algorithm
        :param tstart: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the lightcurve to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstart is 1 then this denotes the time when the binned lightcurve should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

        :param tstop: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the lightcurve to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstop is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param T0: float or an astropy.units.Quantity object with some time of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tmin/tmax that were passed in.
        :param energybins: astropy.units.Quantity denoting the energy bin edges for the energy resolved lightcurves that
            will be calculated
        :param mask_weighting: Boolean to denote if mask weighting should be applied. By default this is set to True,
            however if a source is out of the BAT field of view the mask weighting will produce a lightcurve of 0 counts.
            Setting mask_weighting=False in this case ignores the position of the source and allows the pure rates/counts
            to be calculated.
        :param recalc: Boolean to denote if the lightcurve specified by lightcurve_file should be recalculated with the
            specified time/energy binning. See the Lightcurve class for a list of these defaults.
        :return: Lightcurve class instance
        """
        # batbinevt infile=sw00145675000bevshsp_uf.evt.gz outfile=onesec.lc outtype=LC
        # timedel=1.0 timebinalg=u energybins=15-150
        # detmask=../hk/sw00145675000bcbdq.hk.gz clobber=YES

        lc_dir = self.result_dir.joinpath("lc")

        if self.ra is None and self.dec is None and mask_weighting:
            raise ValueError(
                "Mask weighted lightcurves cannot be created unless an ra/dec is specified. Please use the apply_mask_weighting method to do so.")

        if lc_file is None:
            # contruct the template name from the other inputs
            if timebins is not None or tstart is not None or tstop is not None:
                # try to access tstart/tmin first
                try:
                    min_t = tstart.min().value
                    max_t = tstop.max().value
                except AttributeError as e:
                    # now try to access timebins
                    min_t = timebins.min().value
                    max_t = timebins.max().value

                # add on the T0 if needed
                if is_relative:
                    if isinstance(T0, u.Quantity):
                        min_t += T0.value
                        max_t += T0.value

                    else:
                        min_t += T0
                        max_t += T0

                # finally add the energy channels
                lc_filename = Path(f"t_{min_t}-{max_t}_dt_custom_{len(energybins) - 1}chan.lc")
            else:
                # use th normal batbinevt related information with energy channel as distinguishing info
                lc_filename = Path(
                    f"t_{timebinalg}_dt_{timedelta / np.timedelta64(1, 's')}_{len(energybins) - 1}chan.lc")

        else:
            lc_filename = lc_file

        if not lc_filename.is_absolute():
            # assume that the user wants to put it in the lc directory, or load a file in the lc directory
            lc_filename = lc_dir.joinpath(lc_filename)
        else:
            lc_filename = Path(lc_filename).expanduser().resolve()

        # if the file exists and recalc=False, just load it in and return it. Dont need to add it to the list of
        # lightcurves via the self.lightcurves property
        do_t_energy_calc = not (lc_filename.exists() and not recalc)

        # within Lightcurve, we determine if we load things in or recalculate the Lightcurve with the passed in parameters
        lc = Lightcurve(
            lc_filename,
            self.event_files,
            self.detector_quality_file,
            recalc=recalc,
            mask_weighting=mask_weighting,
        )

        if do_t_energy_calc:
            lc.set_timebins(
                timebinalg=timebinalg,
                timedelta=timedelta,
                tmin=tstart,
                tmax=tstop,
                timebins=timebins,
                is_relative=is_relative,
                T0=T0,
            )
            lc.set_energybins(energybins=energybins)

            self.lightcurves = lc

        # TODO: how to deal with an event file being loaded in and a user wanting to load in a previously created
        #  Lightcurve object? they can use this method to get the lightcurve object and then do event.lightcurves=lc,
        #  will need to change the error for property only being able to be set to an empty list

        return lc

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"], energybins=["energy"])
    def create_pha(
            self,
            pha_file=None,
            tstart=None,
            tstop=None,
            timebins=None,
            T0=None,
            is_relative=False,
            energybins=None,
            recalc=False,
            mask_weighting=True,
            load_upperlims=False,
    ):
        """
        This function creates a pha file that spans a given time range. The pha filename can be specified for a file
        name that the user wants the created pha file to be saved as or this can be set as None to allow for existing
        files to be loaded/recreated with a new set of time/energy binings.

        The time bin of the created spectrum can be provided through the tstart and tstop parameters, which should
        specify the start and stop times of the timebins for which the spectrum can be constructed. Alternatively, the
        time bin can be specified through the timebins parameter which allows the user to pass in an array of time bin
        edges. A spectrum will be constructed for each timebin, similarly if an array of values are passed to the tstart
        and tstop parameters then multiple spectra will be constructed. The time bins can be specified relative to some
        time of interest, T0, which allows for maximal flexibilty. In this case, the user needs to specify T0 and set
        is_relative=True.

        The energy bins of the spectrum/spectra can be set, however we recommend leaving this parameter to None to allow
        for the normal 80 channel spectra to be constructed. Additionally, the spectra can be mask weighted or not, and
        this is set to be enabled by default.

        This method can load upper limit spectral files (spectra that allow users to construct flux upper limits when
        sources are not well detected). By default, these files are not loaded.

        This method also returns the Spectrum object or list of Spectrum
        objects that is/are created.

        Any newly created Spectrum objects are saved to the spectra property where they are stored in order based on their
        creation. If a pha file is loaded in, then the Spectrum will not be saved to the dphs property by default. If
        a user wants to do so they can set the loaded spectrum to the spectra property (ie self.spectra = loaded_spectrum).


        :param pha_file: None or a Path object denoting whether a new predetermined filename should be used, or if
            previous existing files should be loaded or written over (in conjunction with the recalc parameter). The
            file should end with ".pha". If a string is passed without an absolute filepath then it is assumed that the
            created pha file should be placed in the pha/ subdirectory  of the results directory. If set
            to None, the pha filename will be dynamically determined from the other input parameters.
        :param tstart: astropy Quantity scalar or array denoting the start MET time of timebins that the user would like
            to create pha files for. A pha file will be created for each time range specified by tstart and tstop. The
            times can be defined relative to some time of interest which can be specified with the T0 parameter.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

        :param tstop: astropy Quantity scalar or array denoting the end MET time of timebins that the user would like to
            create pha files for. A pha file will be created for each time range specified by tstart and tstop. The
            times can be defined relative to some time of interest which can be specified with the T0 parameter.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

        :param timebins: astropy Quantity  array denoting the MET timebin edges that the spectra should be constructed
            for. The times can be defined relative to some time of interest which can be specified with the T0 parameter
        :param T0: float or astropy Quantity scalar denoting the time that time bins may be defined relative to
        :param is_relative: boolean to denote if the specified timebins are relative times with respect to T0
        :param energybins: None or an astropy Quantity array of energy bin edges that the pha files should be created with. None defaults
            to the 80 channel CALDB energy bins.
        :param recalc: Boolean to denote if a set of pha files should be recreated or if they should be loaded in, if
            they already exist.
        :param mask_weighting: boolean, default True, to denote if the mask weighting should be applied in constructing the pha file.
        :param load_upperlims: boolean, default False, to denote if any of the upper limit pha files should be loaded
            from the pha directory within the results directory.
        :return: Spectrum object or list of Spectrum objects
        """
        # batbinevt infile=sw00145675000bevshsp_uf.evt.gz outfile=onesec.lc outtype=PHA
        # timedel=0.0 timebinalg=u energybins=CALDB
        # detmask=../hk/sw00145675000bcbdq.hk.gz clobber=YES

        pha_dir = self.result_dir.joinpath("pha")

        if self.ra is None and self.dec is None and mask_weighting:
            raise ValueError(
                "spectra cannot be created unless an ra/dec is specified. Please use the apply_mask_weighting method to do so.")

        input_tstart = None
        input_tstop = None

        # if the timebins is defined, will need to break it up into the tstart and tstop arrays to iterate over
        # if tstart/tstop are specified we will prefer to use those
        if timebins is not None:
            input_tstart = timebins[:-1]
            input_tstop = timebins[1:]

        # do error checking on tmin/tmax make sure both are defined and that they are the same length
        if (tstart is None and tstop is not None) or (
                tstart is None and tstop is not None
        ):
            raise ValueError("Both tstart and tstop must be defined.")

        if tstart is not None and tstop is not None:
            if tstart.size == tstop.size:
                input_tstart = tstart.copy()
                input_tstop = tstop.copy()

                # make sure that we can iterate over the times even if the user passed in a single scalar quantity
                if input_tstart.isscalar:
                    input_tstart = u.Quantity([input_tstart])
                    input_tstop = u.Quantity([input_tstop])
            else:
                raise ValueError("Both tstart and tstop must have the same length.")

        if energybins is None:
            nchannels = 80
        else:
            nchannels = len(energybins) - 1

        # if we are passing in a number of timebins, the user can pass in a number of pha files to load/create so make
        # sure that we have the same number of pha_files passed in or that it is None
        if pha_file is not None and type(pha_file) is not list:
            pha_file = [pha_file]

        # The user has not specified a set or single pha file to load in or create then we need to determine which ones
        # to load. By default, we load all of them and dont recalculate anything. If recalc=true, we load all of them
        # and then they get recalcualted with whatever paramters get passed to the Lightcurve object.

        if pha_file is None:
            # construct the template name from the spectral inputs
            pha_filename = []
            if input_tstop is not None:
                for start, end in zip(input_tstart.value, input_tstop.value):
                    if is_relative:
                        if isinstance(T0, u.Quantity):
                            start += T0.value
                            end += T0.value
                        else:
                            start += T0
                            end += T0

                    name = Path(f"t_{start}-{end}_{nchannels}chan.pha")
                    pha_filename.append(name)

            else:
                start = "start"
                end = "end"
                input_tstart = np.array([None])
                input_tstop = np.array([None])

                name = Path(f"t_{start}-{end}_{nchannels}chan.pha")
                pha_filename.append(name)

            # if we want to load the upper limits, need to see if we have this file. Assume we are looking for
            # files in the OBSID/pha directory
            # f not, then default to using the non-upper limit pha file and throw a warning
            if load_upperlims:
                new_phafilename = []
                for i in pha_filename:
                    upperlim_files = list(pha_dir.glob(f"{i.stem}_bkgnsigma_*_upperlimit.pha"))
                    if len(upperlim_files) > 0:
                        for j in upperlim_files:
                            new_phafilename.append(j)
                    else:
                        new_phafilename.append(i)
                        warnings.warn(
                            f"There is no associated upper limit pha file for {i}, will continue by just loading in this file.",
                            stacklevel=2)
                pha_filename = new_phafilename

        else:
            if type(pha_file) is list:
                pha_filename = pha_file
            else:
                pha_filename = [pha_file]

        # if a single file has been specified, assume that is should go in the event/pha directory unless
        # the user has passed in an absolute file path
        final_pha_files = [
            pha_dir.joinpath(f"{i}")
            if not Path(i).is_absolute()
            else Path(i).expanduser().resolve()
            for i in pha_filename
        ]

        # need to see if input_tstart/input_tstop is None. If not None, then need to check that the lengths are the
        # same
        if input_tstop is not None and len(final_pha_files) != input_tstart.size:
            raise ValueError(
                "The number of pha files does not match the number of timebins. Please make sure these are "
                "the same length or that pha_files is set to None"
            )

        spectrum_list = []
        do_t_energy_calc = []
        for i in range(input_tstart.size):
            # if the file exists and recalc=False, just load it in and return it. Dont need to add it to the list of
            # lightcurves via the self.lightcurves property
            do_t_energy_calc.append(not (final_pha_files[i].exists() and not recalc))

            # within this constructor we determine if we need to load things or not
            spectrum = Spectrum(
                final_pha_files[i],
                self.event_files,
                self.detector_quality_file,
                self.auxil_raytracing_file,
                mask_weighting=mask_weighting,
                recalc=recalc,
            )

            # if we needed to create a new pha file, then we should make sure we set the timebins/energybins
            # we will also need to add the Spectrum to the spectra property
            if do_t_energy_calc[i]:
                spectrum.set_timebins(
                    tmin=input_tstart[i],
                    tmax=input_tstop[i],
                    T0=T0,
                    is_relative=is_relative,
                )
                if energybins is not None:
                    # if energybins is None, then the default energybinning of "CALDB" is used
                    spectrum.set_energybins(energybins=energybins)

            spectrum_list.append(spectrum)

        # save the spectrum list as an attribute, if there is one spectrum then index it appropriately
        # if len(spectrum_list) == 1:
        #    self.spectra = spectrum_list[0]
        # else:
        #    self.spectra = spectrum_list
        for i, t_energy_calc in zip(spectrum_list, do_t_energy_calc):
            if t_energy_calc:
                self.spectra = i

        if len(spectrum_list) == 1:
            return spectrum_list[0]
        else:
            return spectrum_list

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"], energybins=["energy"])
    def create_dph(
            self,
            dph_file=None,
            tstart=None,
            tstop=None,
            timebins=None,
            T0=None,
            is_relative=False,
            energybins=None,
            recalc=False,
    ):
        """
        This method creates a detector plane histogram. By default, this method will create a single BatDPH object for
        the time/energy ranges that are specified. If tstart/tstop/timebins is set to None, the default time binning
        will be a single time bin extending from the start time to the end time of the event dataset. If energybins is
        set to None, the default energy binning will be a single energy bin from 14-195 keV.

        Any newly created BatDPH objects are saved to the dphs property where they are stored in order based on their
        creation. If a DPH is loaded in, then the BatDPH will not be saved to the dphs property by default. If
        a user wants to do so they can set the loaded BatDPH to the dphs property (ie self.dphs = loaded_dph).

        :param dph_file: None or a path object of the dph file that will be read in, if previously calculated,
            or the location/name of the new dph file that will contain the newly calculated dph. If set
            to None, the DPH filename will be dynamically determined from the other input parameters. If the file exists,
            then it will be either read in or recreated, depending on the recalc parameter. By default, the DPHs are
            placed in the dph/ directory unless a Path object is passed in with an absolute filepath.
        :param tstart: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the DPH to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstart is 1 then this denotes the time when the binned lightcurve should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.
        :param tstop: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the DPH to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstop is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param T0: float or an astropy.units.Quantity object with some time of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tstart/tstop that were passed in.
        :param energybins: astropy.units.Quantity denoting the energy bin edges for the DPH that will be produced. None
            sets the default energy binning to be 14-195 keV
        :param recalc: Boolean to denote if the DPH specified by dph_file should be recalculated with the
            specified time/energy binning. See the BatDPH class for a list of these defaults.
        :return: BatDPH object or list of BatDPH objects
        """

        dph_dir = self.result_dir.joinpath("dph")

        # make the directory if it doesnt exist, if it does then we are fine. This is done here because users dont
        # usually create DPHs with the event file but this creates a subdirectory to put them in  if a user wants to
        # do so
        dph_dir.mkdir(exist_ok=True)

        if energybins is None:
            nchannels = 1
        else:
            nchannels = len(energybins) - 1

        input_tstart = None
        input_tstop = None

        # if the timebins is defined, will need to break it up into the tstart and tstop arrays to iterate over
        # if tstart/tstop are specified we will prefer to use those
        if timebins is not None:
            input_tstart = timebins[:-1]
            input_tstop = timebins[1:]

        # do error checking on tmin/tmax make sure both are defined and that they are the same length
        if (tstart is None and tstop is not None) or (
                tstart is None and tstop is not None
        ):
            raise ValueError("Both tstart and tstop must be defined.")

        if tstart is not None and tstop is not None:
            if tstart.size == tstop.size:
                input_tstart = tstart.copy()
                input_tstop = tstop.copy()

            # make sure that we can iterate over the times even if the user passed in a single scalar quantity
            if input_tstart.isscalar:
                input_tstart = u.Quantity([input_tstart])
                input_tstop = u.Quantity([input_tstop])
            else:
                raise ValueError("Both tstart and tstop must have the same length.")

        if dph_file is None:
            # construct the template name from the inputs
            dph_filename = []
            if input_tstop is not None:
                if is_relative:
                    if isinstance(T0, u.Quantity):
                        start = (input_tstart + T0).min().value
                        end = (input_tstop + T0).max().value
                    else:
                        start = input_tstart.min().value + T0
                        end = input_tstop.max().value + T0

                ntbins = input_tstart.size
            else:
                start = "start"
                end = "end"
                ntbins = 1

            name = Path(f"t_{start}-{end}_{ntbins}tbins_{nchannels}chan.dph")
            dph_filename.append(name)


        else:
            dph_filename = dph_file

        final_dph_files = [
            dph_dir.joinpath(f"{i}")
            if not Path(i).is_absolute()
            else Path(i).expanduser().resolve()
            for i in dph_filename
        ]

        dph_list = []
        for i in range(len(final_dph_files)):
            # if the file exists and recalc=False, just load it in and return it. Dont need to add it to the list of
            # dphs via the self.dphs property
            do_t_energy_calc = not (final_dph_files[i].exists() and not recalc)

            dph = BatDPH(final_dph_files[i], event_file=self.event_files,
                         recalc=recalc)

            if do_t_energy_calc:
                dph.set_timebins(tmin=input_tstart, tmax=input_tstop, is_relative=is_relative, T0=T0)
                if energybins is not None:
                    dph.set_energybins(energybins=energybins)

                self.dphs = dph

                dph_list.append(dph)

        return dph_list[0]

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"], energybins=["energy"])
    def create_dpi(self,
                   dpi_file=None,
                   tstart=None,
                   tstop=None,
                   timebins=None,
                   T0=None,
                   is_relative=False,
                   energybins=[15, 350] * u.keV,
                   recalc=False,
                   ):
        """
        This method creates and returns a BatDPI object. Unlike the create_DPH method, one DPI created here
        corresponds to only 1 time bin and as many energybins as is specified by the user.

        If the user attempts to create a DPI file outside of the range of times for which there is event data, an error
        will be raised.

        Any newly created BatDPI objects are saved to the dpis property where they are stored in order based on their
        creation. If a DPI is loaded in, then the BatDPI will not be saved to the dpis property by default. If
        a user wants to do so they can set the loaded BatDPI to the dpis property (ie self.dpis = loaded_dpi).


        :param dpi_file: None or a path object of the DPI file that will be read in, if previously calculated,
            or the location/name of the new DPI file that will contain the newly calculated DPI. If set
            to None, the DPI filename will be dynamically determined from the other input parameters. If the file exists,
            then it will be either read in or recreated, depending on the recalc parameter. By default, the DPIs are
            placed in the dpi/ directory unless a Path object is passed in with an absolute filepath.
        :param tstart: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the DPI to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstart is 1 then this denotes the time when the binned lightcurve should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.
        :param tstop: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the DPI to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstop is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param T0: float or an astropy.units.Quantity object with some time of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tstart/tstop that were passed in.
        :param energybins: astropy.units.Quantity denoting the energy bin edges for the DPI that will be produced. None
            sets the default energy binning to be 14-195 keV
        :param recalc: Boolean to denote if the DPH specified by dph_file should be recalculated with the
            specified time/energy binning. See the BatDPH class for a list of these defaults.
        :return: BatDPI object or a list of BatDPI objects
        """

        dpi_dir = self.result_dir.joinpath("dpi")
        nchannels = len(energybins) - 1

        input_tstart = None
        input_tstop = None
        # if the timebins is defined, will need to break it up into the tstart and tstop arrays to iterate over
        # if tstart/tstop are specified we will prefer to use those
        if timebins is not None:
            input_tstart = timebins[:-1]
            input_tstop = timebins[1:]
        # do error checking on tmin/tmax make sure both are defined and that they are the same length
        if (tstart is None and tstop is not None) or (
                tstart is None and tstop is not None
        ):
            raise ValueError("Both tstart and tstop must be defined.")
        if tstart is not None and tstop is not None:
            if tstart.size == tstop.size:
                input_tstart = tstart.copy()
                input_tstop = tstop.copy()
            # make sure that we can iterate over the times even if the user passed in a single scalar quantity
            if input_tstart.isscalar:
                input_tstart = u.Quantity([input_tstart])
                input_tstop = u.Quantity([input_tstop])
            else:
                raise ValueError("Both tstart and tstop must have the same length.")

        if dpi_file is None:
            # construct the template names
            dpi_filename = []
            if input_tstop is not None:
                for start, end in zip(input_tstart, input_tstop):
                    if isinstance(T0, u.Quantity):
                        s = (start + T0).min().value
                        e = (end + T0).max().value
                    else:
                        if T0 is None:
                            s = start.min().value
                            e = end.max().value
                        else:
                            s = start.min().value + T0
                            e = end.max().value + T0

                    # make sure that the timebin does not extend past the min/max event data time
                    if s < self.data.time.min().value or e > self.data.time.max().value:
                        raise ValueError(
                            f"The bounds of the timebin {s}-{e} extend past the min/max event time in the event file.")

                    dpi_filename.append(Path(f"t_{s}-{e}_{nchannels}chan.dpi"))
            else:
                start = "start"
                end = "end"
                input_tstart = np.array([None])
                input_tstop = np.array([None])

                dpi_filename.append(Path(f"t_{start}-{end}_{nchannels}chan.dpi"))

        else:
            if type(dpi_file) is list:
                dpi_filename = dpi_file
            else:
                dpi_filename = [dpi_file]

        final_dpi_files = [
            dpi_dir.joinpath(f"{i}")
            if not Path(i).is_absolute()
            else Path(i).expanduser().resolve()
            for i in dpi_filename
        ]

        dpi_list = []
        for start, end, file in zip(input_tstart, input_tstop, final_dpi_files):
            # if the file exists and recalc=False, just load it in and return it. Dont need to add it to the list of
            # dphs via the self.dphs property
            do_t_energy_calc = not (file.exists() and not recalc)

            dpi = BatDPI(file, event_file=self.event_files,
                         detector_quality_file=self.detector_quality_file, recalc=recalc)

            if do_t_energy_calc:
                dpi.set_timebins(tmin=start, tmax=end, is_relative=is_relative, T0=T0)
                if energybins is not None:
                    dpi.set_energybins(energybins=energybins)

            self.dpis = dpi

            dpi_list.append(dpi)

        if len(dpi_list) == 1:
            return dpi_list[0]
        else:
            return dpi_list

    def create_skyview(self,
                       dpis=None,
                       tstart=None,
                       tstop=None,
                       timebins=None,
                       T0=None,
                       is_relative=False,
                       energybins=[15, 350] * u.keV,
                       input_dict=None,
                       recalc=False,
                       ):
        """
        This method returns a sky view for all the DPIs that have been specified. If no DPIs
        have been created which correspond to the input times/energies then this method will create them and then produce
        a BatSkyView object for all the DPIs.

        Any newly created BatSkyView objects are saved to the skyviews property where they are stored in order based on their
        creation. If a BatSkyView is loaded, then the BatSkyView will not be saved to the skyviews property by default. If
        a user wants to do so they can set the loaded BatSkyView to the skyviews property (ie self.skyviews = loaded_skyview).


        :param dpis: None, a BatDPI object, or a list of BatDPI objects that will be used to produce the BatSkyView object(s)
        :param tstart: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the DPI and resulting skyview to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstart is 1 then this denotes the time when the binned lightcurve should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.
        :param tstop: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the DPI and resulting skyview to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if tstart/tstop are specified then anything passed to the timebins parameter is ignored.

            If the length of tstop is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param T0: float or an astropy.units.Quantity object with some time of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tstart/tstop that were passed in.
        :param energybins: astropy.units.Quantity denoting the energy bin edges for the DPH that will be produced. None
            sets the default energy binning to be 14-195 keV
        :param input_dict: None or dict of key/value pairs that will be passed to BatSkyView (and then batfftimage).
            If this is set to None, the default batfftimage parameter values will be used. If a dictionary is passed in,
            it will overwrite the default values for the keys that are specified.
            eg input_dict=dict(aperture="CALDB:DETECTION") would cause batfftimg to use the CALDB detection-optimized
            aperture map to construct the sky view
        :param recalc: Boolean to denote if the DPH specified by dph_file should be recalculated with the
            specified time/energy binning. See the BatDPH class for a list of these defaults.
        :return: BatSkyView object or a list of BatSkyView objects
        """

        skyview_dir = self.result_dir.joinpath("img")

        # make the directory if it doesnt exist, if it does then we are fine. This is done here because users dont
        # usually create sky images with the event file but this creates a subdirectory to put them in  if a user wants to
        # do so
        skyview_dir.mkdir(exist_ok=True)

        # if dpis is None, we need to create/load the DPI(s) from the other parameters passed in
        if dpis is None:
            dpi_output = self.create_dpi(tstart=tstart, tstop=tstop, timebins=timebins, T0=T0, is_relative=is_relative,
                                         energybins=energybins, recalc=recalc)
            # make sure we have a list
            if not isinstance(dpi_output, list):
                dpi_output = [dpi_output]
        else:
            if not isinstance(dpis, list):
                dpi_output = [dpis]
            else:
                dpi_output = dpis

            if np.any([not isinstance(i, BatDPI) for i in dpi_output]):
                raise ValueError(
                    "dpi can only be set to a BatDPI object or a list of BatDPI objects.")

        # we have created/loaded the necessary information to now create the BatSkyView objects
        skyviews_list = []
        for dpi in dpi_output:
            # create the name of the skyview files since we want them to be in the
            # img folder
            skyview_file = skyview_dir.joinpath(f"{dpi.dpi_file.stem}.img")

            # determine if we need to save this object to the skyviews property
            # if the file exists and recalc=False, just load it in and return it. Dont need to add it to the list of
            # skyviews via the self.skyviews property
            save_property = not (skyview_file.exists() and not recalc)

            skyview = BatSkyView(skyimg_file=skyview_file, bat_dpi=dpi, attitude_file=self.attitude_file,
                                 create_bkg_stddev_img=True,
                                 create_snr_img=True, input_dict=input_dict, recalc=recalc)

            if save_property:
                self.skyviews = skyview

            skyviews_list.append(skyview)

        if len(skyviews_list) == 1:
            return skyviews_list[0]
        else:
            return skyviews_list

    @property
    def spectra(self):
        """A list of spectrum objects that have been created from the event object"""
        return self._spectra

    @spectra.setter
    def spectra(self, value):
        if value is None:
            self._spectra = value
        elif isinstance(value, Spectrum):
            if self._spectra is None:
                self._spectra = []
            self._spectra.append(value)
        elif isinstance(value, list):
            if len(value) > 0:
                raise ValueError(
                    "The spectra property can only be set to None, an empty list, or have a Spectrum object appended to it.")
            self._spectra = value
        else:
            raise ValueError(
                "The spectra property can only be set to None, an empty list, or have a Spectrum object appended to it.")

    @property
    def lightcurves(self):
        """A list of lightcurve objects that have been created from the event file"""
        return self._lightcurves

    @lightcurves.setter
    def lightcurves(self, value):
        if value is None:
            self._lightcurves = value
        elif isinstance(value, Lightcurve):
            if self._lightcurves is None:
                self._lightcurves = []
            self._lightcurves.append(value)
        elif isinstance(value, list):
            if len(value) > 0:
                raise ValueError(
                    "The lightcurves property can only be set to None, an empty list, or have a Lightcurve object appended to it.")
            self._lightcurves = value
        else:
            raise ValueError(
                "The lightcurves property can only be set to None, an empty list, or have a Lightcurve object appended to it.")

    @property
    def dphs(self):
        """A list of DPH objects that have been created from the event file"""
        return self._dphs

    @dphs.setter
    def dphs(self, value):
        if value is None:
            self._dphs = value
        elif isinstance(value, BatDPH):
            if self._dphs is None:
                self._dphs = []
            self._dphs.append(value)
        elif isinstance(value, list):
            if len(value) > 0:
                raise ValueError(
                    "The dphs property can only be set to None, an empty list, or have a BatDPH object appended to it.")
            self._dphs = value
        else:
            raise ValueError(
                "The dphs property can only be set to None, an empty list, or have a BatDPH object appended to it.")

    @property
    def dpis(self):
        """A list of DPI objects that have been created from the event file"""
        return self._dpis

    @dpis.setter
    def dpis(self, value):
        if value is None:
            self._dpis = value
        elif isinstance(value, BatDPI):
            if self._dpis is None:
                self._dpis = []
            self._dpis.append(value)
        elif isinstance(value, list):
            if len(value) > 0:
                raise ValueError(
                    "The dpis property can only be set to None, an empty list, or have a BatDPI object appended to it.")
            self._dpis = value
        else:
            raise ValueError(
                "The dpis property can only be set to None, an empty list, or have a BatDPI object appended to it.")

    @property
    def skyviews(self):
        """A list of BatSkyView objects that have been created from the event file"""
        return self._skyviews

    @skyviews.setter
    def skyviews(self, value):
        if value is None:
            self._skyviews = value
        elif isinstance(value, BatSkyView):
            if self._skyviews is None:
                self._skyviews = []
            self._skyviews.append(value)
        elif isinstance(value, list):
            if len(value) > 0:
                raise ValueError(
                    "The skyviews property can only be set to None, an empty list, or have a BatSkyView object appended to it.")
            self._skyviews = value
        else:
            raise ValueError(
                "The skyviews property can only be set to None, an empty list, or have a BatSkyView object appended to it.")
