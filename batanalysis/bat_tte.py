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
from astropy.io import fits

from .bat_dpi import BatDPI
from .batlib import dirtest, decompose_det_id
from .batobservation import BatObservation
from .batproducts import Lightcurve, Spectrum

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class TimeTaggedEvents(object):
    """
    This class encapsulates the event data that is obtained by the BAT instrument.

    TODO: add methods to add/concatenate event data, plot event data, etc
    """

    def __init__(
            self,
            times,
            detector_id,
            detx,
            dety,
            quality_flag,
            energy,
            pulse_height_amplitude,
            pulse_invariant,
            mask_weight=None,
    ):
        """
        This initalizes the TimeTaggedEvent class and allows for the data to be accessed easily.

        :param times:
        :param detector_id:
        :param detx:
        :param dety:
        :param quality_flag:
        :param energy:
        :param pulse_height_amplitude:
        :param pulse_invariant:
        :param mask_weight:
        """

        self.time = times
        self.detector_id = detector_id
        self.detx = detx
        self.dety = dety
        self.quality_flag = quality_flag
        self.energy = energy
        self.pha = pulse_height_amplitude
        self.pi = pulse_invariant
        self.mask_weight = mask_weight

        # get the block/DM/sandwich/channel info
        block, dm, side, channel = decompose_det_id(self.detector_id)
        self.detector_block = block
        self.detector_dm = dm
        self.detector_sand = side
        self.detector_chan = channel


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
            self.event_files = sorted(
                list(self.obs_dir.joinpath("bat").joinpath("event").glob("*bev*_uf*"))
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
                    tdrss_ra = file[0].header["BRA_OBJ"]
                    tdrss_dec = file[0].header["BDEC_OBJ"]

            # get info from event file which must exist to get to this point
            with fits.open(self.event_files) as file:
                event_ra = file[0].header["RA_OBJ"]
                event_dec = file[0].header["DEC_OBJ"]

            # by default, ra/dec="event" to use the coordinates set here by SDC but can use other coordinates
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
                # use the event file RA/DEC
                self.ra = event_ra
                self.dec = event_dec
            else:
                if np.isreal(ra) and np.isreal(dec):
                    self.ra = ra
                    self.dec = dec
                else:
                    # the ra/dec values must be decimal degrees for the following analysis to work
                    raise ValueError(
                        f"The passed values of ra and dec are not decimal degrees. Please set these to appropriate values."
                    )

            # see if the RA/DEC that the user wants to use is what is in the event file
            # if not, then we need to do the mask weighting again
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
                self.trigtime_met = hdr["TRIGTIME"]
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

            # create the marker file that tells us that the __init__ method completed successfully
            complete_file = self.result_dir.joinpath(".batevent_complete")
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
        This function reads in the data from the event file
        :return: None
        """

        all_data = {}
        with fits.open(self.event_files) as file:
            data = file[1].data
            for i in data.columns:
                all_data[i.name] = u.Quantity(data[i.name], i.unit)

        self.data = TimeTaggedEvents(
            all_data["TIME"],
            all_data["DET_ID"],
            all_data["DETX"],
            all_data["DETY"],
            all_data["EVENT_FLAGS"],
            all_data["ENERGY"],
            all_data["PHA"],
            all_data["PI"],
            mask_weight=all_data["MASK_WEIGHT"],
        )

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
        )  # os.path.join(self.result_dir, "batsurvey.pickle")
        with open(file, "wb") as f:
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
            # Create DPI
            # batbinevt bat/event/*bevshsp_uf.evt.gz grb.dpi DPI 0 u - weighted = no outunits = counts
            output_dpi = (
                self.obs_dir.joinpath("bat")
                .joinpath("hk")
                .joinpath("detector_quality.dpi")
            )
            input_dict = dict(
                infile=str(self.event_files[0]),
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
                f"The event file {self.event_files} has not had the energy calibration applied and there is no gain/offset "
                f"file for this trigger with observation ID \
                {self.obs_id} located at {self.obs_dir}. This file is necessary for the remaining processing since an"
                f"energy calibration needs to be applied."
            )
        elif len(self.gain_offset_file) > 1:
            raise AttributeError(
                f"The event file {self.event_files} has not had the energy calibration applied and there are too many gain/offset "
                f"files for this trigger with observation ID \
                            {self.obs_id} located at {self.obs_dir}. One of these files is necessary for the remaining processing since an"
                f"energy calibration needs to be applied."
            )
        else:
            # if we have the file, then we need to call bateconvert
            input_dict = dict(
                infile=str(self.event_files),
                calfile=str(self.gain_offset_file),
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
            ra = self.ra
            dec = self.dec
        else:
            # set the new ra/dec values
            self.ra = ra
            self.dec = dec

        # if this attribute is None, we need to define it and create it using the standard naming convention
        if self.auxil_raytracing_file is None:
            temp_auxil_raytracing_file = self.event_files.parent.join(
                f"sw{self.obs_id}bevtr.fits"
            )
        else:
            temp_auxil_raytracing_file = self.auxil_raytracing_file

        input_dict = dict(
            infile=str(self.event_files),
            attitude=str(self.attitude_file),
            detmask=str(self.detector_quality_file),
            ra=ra,
            dec=dec,
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
            event_ra = file[0].header["RA_OBJ"]
            event_dec = file[0].header["DEC_OBJ"]
            if event_ra != self.ra or event_dec != self.dec:
                # update the event file RA/DEC_OBJ values everywhere
                for i in file:
                    i.header["RA_OBJ"] = self.ra
                    i.header["DEC_OBJ"] = self.dec

                    # the BAT_RA/BAT_DEC keys have to updated too since this is something
                    # that the software manual points out should be updated
                    i.header["BAT_RA"] = self.ra
                    i.header["BAT_DEC"] = self.dec

            file.flush()

        # reread in the event file data
        self._parse_event_file()

        # save the file as the attribute if everything else is successful
        self.auxil_raytracing_file = temp_auxil_raytracing_file

        # TODO how to handle a different auxiliary ray tracing file bieng produced here?

        return None

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"])
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
            energybins=["15-25", "25-50", "50-100", "100-350"],
            mask_weighting=True,
            recalc=False,
    ):
        """
        This method returns a lightcurve object which can be manipulated in different energies/timebins. The lightcurve
        path may be provided, which can be a lightcurve that should be loaded (if created already), or the name of the
        lightcurve that will be created with the specified energy/time binning. If no lightcurve file name is provided,
        the method will determine a generic lightcurve name.

        This method allows one to specify different energy/time binnings however since this method returns a Lightcurve
        class, the resulting Lightcurve class instance can be used to rebin the lightcurve however the user wants. The
        lightcurve is also saved to the BatEvent.lightcurve attribute when it is created thorugh this method.

        :param lc_file: path object of the lightcurve file that will be read in, if previously calculated,
            or the location/name of the new lightcurve file that will contain the newly calculated lightcurve.
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
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.

            If the length of tmin is 1 then this denotes the time when the binned lightcurve should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param tstop: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the lightcurve to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.

            If the length of tmin is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param T0: float or an astropy.units.Quantity object with some tiem of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tmin/tmax that were passed in.
        :param energybins:
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

        if lc_file is None:
            if not recalc:
                # make up a name for the light curve that hasnt been used already in the LC directory
                lc_files = list(self.result_dir.joinpath("lc").glob("*.lc"))
                lc_files = [str(i) for i in lc_files]
                base = "lightcurve_"
                count = 0
                while np.any([f"{base}{count}.lc" in t for t in lc_files]):
                    count += 1
                lc_file = self.result_dir.joinpath("lc").joinpath(f"{base}{count}.lc")
            else:
                lc_files = list(self.result_dir.joinpath("lc").glob("*.lc"))
                if len(lc_files) == 1:
                    lc_file = lc_files[0]
                else:
                    raise ValueError(
                        f"There are too many files which meet the criteria to be loaded. Please specify one of {lc_files}."
                    )
        else:
            lc_file = Path(lc_file).expanduser().resolve()

        lc = Lightcurve(
            lc_file,
            self.event_files,
            self.detector_quality_file,
            recalc=recalc,
            mask_weighting=mask_weighting,
        )
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

        self.lightcurve = lc

        return self.lightcurve

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"])
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

        The spectrum/spectra get dynamically loaded to the spectrum attribute where the newly created spectrum/spectra
        replaces what was saved in this attribute. This method also returns the Spectrum object or list of Spectrum
        objects that is/are created.

        :param pha_file: None, string, or Path object denoting whether a new predetermied filename should be used, or if
            previous existing files should be loaded or written over (in conjunction with the recalc parameter). The
            file should end with ".pha". If a string is passed without an abolute filepath then it is assumed that the
            created pha file should be placed in the pha/ subdirectory  of the results directory
        :param tstart: astropy Quantity scalar or array denoting the start MET time of timebins that the user would like
            to create pha files for. A pha file will be created for each time range specified by tstart and tstop. The
            times can be defined relative to some time of interest which can be specified with the T0 parameter.
        :param tstop: astropy Quantity scalar or array denoting the end MET time of timebins that the user would like to
            create pha files for. A pha file will be created for each time range specified by tstart and tstop. The
            times can be defined relative to some time of interest which can be specified with the T0 parameter.
        :param timebins: astropy Quantity  array denoting the MET timebin edges that the spectra should be constructed
            for. The times can be defined relative to some time of interest which can be specified with the T0 parameter
        :param T0: float or astropy Quantity scalar denoting the time that time bins may be defined relative to
        :param is_relative: boolean to denote if the specified timebins are relative times with respect to T0
        :param energybins: None or an array of energy bin edges that the pha files should be created with. None defaults
            to the 80 channel CALDB energy bins.
        :param recalc: Boolean to denote if a set of
        :param mask_weighting: boolean, default True, to denote if the mask weighting should be applied in constructing the pha file.
        :param load_upperlims: boolean, default False, to denote if any of the upper limit pha files should be loaded
            from the pha directory within the results directory.
        :return: Spectrum object or list of Spectrum objects
        """
        # batbinevt infile=sw00145675000bevshsp_uf.evt.gz outfile=onesec.lc outtype=PHA
        # timedel=0.0 timebinalg=u energybins=CALDB
        # detmask=../hk/sw00145675000bcbdq.hk.gz clobber=YES

        input_tstart = None
        input_tstop = None
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

        # if the timebins is defined, will need to break it up into the tstart and tstop arrays to iterate over
        if timebins is not None:
            input_tstart = timebins[:-1]
            input_tstop = timebins[1:]

        # if we are passing in a number of timebins, the user can pass in a number of pha files to load/create so make
        # sure that we have the same number of pha_files passed in or that it is None
        if pha_file is not None and type(pha_file) is not list:
            pha_file = [pha_file]

        # The user has not specified a set or single pha file to load in or create then we need to determine which ones
        # to load. By default, we load all of them and dont recalculate anything. If recalc=true, we load all of them
        # and then they get recalcualted with whatever paramters get passed to the Lightcurve object.

        if pha_file is None:
            # identify any pha files that may already exist. If there are make a list of them. If not then create a
            # new set based on the timebins that the user has provided
            pha_files = [
                i.name for i in list(self.result_dir.joinpath("pha").glob("*.pha"))
            ]

            if not load_upperlims:
                pha_files = [i for i in pha_files if "upperlim" not in i]

            if not recalc:
                if len(pha_files) > 0:
                    final_pha_files = [
                        self.result_dir.joinpath("pha").joinpath(f"{i}")
                        for i in pha_files
                    ]

                    # can have some number of pha files that dont correspond to the number of timebins which we dont want to
                    # handle right now. In the future can create the necessary amount if input_tstart.size > len(pha_files),
                    # or choose a subset if input_tstart.size < len(pha_files)
                    if input_tstart is not None and input_tstart.size != len(
                            final_pha_files
                    ):
                        raise ValueError(
                            "The number of pha files that exist in the pha directory do not match the "
                            "specified time binning for the creation of spectra when recalc=False. "
                        )

                    # if the tstart/tstop is None, we need to fill these to iterate over them and load in the pha files
                    if input_tstart is None:
                        input_tstart = np.array([None] * len(final_pha_files))
                        input_tstop = input_tstart

                elif len(pha_files) == 0:
                    # iterate through the pha files that need to be created
                    final_pha_files = []
                    base = "spectrum_"
                    count = 0
                    for i in range(input_tstart.size):
                        final_pha_files.append(
                            self.result_dir.joinpath("pha").joinpath(
                                f"{base}{count}.pha"
                            )
                        )
                        count += 1

            else:
                # list the currently existing pha files
                final_pha_files = [
                    self.result_dir.joinpath("pha").joinpath(f"{i}") for i in pha_files
                ]

                # can have that the input_tstart is None, in which case what are we doing? can have different number of
                # input_tstart than final_pha_file, then just delete all files in pha
                # directory for this case
                if input_tstart is None:
                    raise ValueError(
                        "The number of pha files that exist in the pha directory do not match the "
                        "specified time binning for the creation of spectra when recalc=True. "
                    )

                if input_tstart is not None and input_tstart.size != len(
                        final_pha_files
                ):
                    warnings.warn(
                        f"Deleting all files in {self.result_dir.joinpath('pha')} and creating new"
                        f"pha files for the passed in timebins"
                    )
                    dirtest(self.result_dir.joinpath("pha"))
                    # iterate through the pha files that need to be created
                    final_pha_files = []
                    base = "spectrum_"
                    count = 0
                    for i in range(input_tstart.size):
                        final_pha_files.append(
                            self.result_dir.joinpath("pha").joinpath(
                                f"{base}{count}.pha"
                            )
                        )
                        count += 1

        else:
            # if a single file has been specified, assume that is should go in the event/pha directory unless
            # the user has passed in an absolute file path
            final_pha_files = [
                self.result_dir.joinpath("pha").joinpath(f"{i}")
                if not Path(i).is_absolute()
                else Path(i).expanduser().resolve()
                for i in pha_file
            ]

            # need to see if input_tstart/input_tstop is None. If not None, then need to check that the lengths are the
            # same
            if input_tstop is not None and len(pha_file) != input_tstart.size:
                raise ValueError(
                    "The number of pha files does not match the number of timebins. Please make sure these are "
                    "the same length or that pha_files is set to None"
                )

            # if input_stop/input_start is None, then we need to just set it to be the same length as the final_pha_files
            # list. this is to get the loop below going.
            if recalc and input_tstart is None and energybins is None:
                raise ValueError(
                    "recalc has been set to True, but there is not sufficient information for rebinning "
                    f"the following lightcurves {','.join([i.name for i in final_pha_files])}. Please enter"
                    "information related to a change in timebins or energy bins"
                )

        spectrum_list = []
        for i in range(input_tstart.size):
            spectrum = Spectrum(
                final_pha_files[i],
                self.event_files,
                self.detector_quality_file,
                self.auxil_raytracing_file,
                mask_weighting=mask_weighting,
                recalc=recalc,
            )

            # need to check about recalculating this if recalc=False
            if pha_file is None and recalc or pha_file is not None:
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
        if len(spectrum_list) == 1:
            self.spectrum = spectrum_list[0]
        else:
            self.spectrum = spectrum_list

        return self.spectrum

    @u.quantity_input(timebins=["time"], tstart=["time"], tstop=["time"])
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
            mask_weighting=True,
    ):
        """
        This method creates a detector plane histogram.

        :return:
        """

        raise NotImplementedError("Creating the DPH has not yet been implemented.")

        return None

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
        This method creates and returns a detector plane image.

        :param kwargs:
        :return:
        """

        dpi_dir = self.result_dir.joinpath("dpi")
        nebin = len(energybins) - 1

        dpi_list = []
        for start, end, i in zip(tstart, tstop, range(len(tstart))):
            dpi = BatDPI(dpi_dir.joinpath(f"t_{start.value}-{end.value}_{nebin}chan.dpi"), event_file=self.event_files,
                         detector_quality_file=self.detector_quality_file, recalc=recalc)

            dpi.set_timebins(tmin=start, tmax=end, is_relative=is_relative, T0=T0)
            dpi.set_energybins(energybins=energybins)

            dpi_list.append(dpi)
        # raise NotImplementedError("Creating the DPI has not yet been implemented.")

        return dpi_list

    def create_sky_image(self, **kwargs):
        """
        This method returns a sky image

        :param kwargs:
        :return:
        """

        raise NotImplementedError(
            "Creating the sky image has not yet been implemented."
        )

        return None
