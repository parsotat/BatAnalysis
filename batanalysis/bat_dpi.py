"""
This file holds the BAT Detector plane image class

Tyler Parsotan Feb 21 2024
"""

import gzip
import shutil
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from .batlib import create_gti_file
from .detectorplanehist import DetectorPlaneHistogram

try:
    import heasoftpy.swift as hsp
    import heasoftpy as hsp_core
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatDPI(DetectorPlaneHistogram):
    """
    This class encapsulates the BAT Detector Plane Image data product. This is a Detector Plane Histogram
    that has had batsurvey-erebin and baterebin and select good time intervals batsurvey-gti applied to it.

    This can also be constructed by batbinevt which this class handles. There can be multiple DPIs extensions in a file
    or there can be a table with multiple DPIs

    TODO: for images with exposures>100 seconds, should also create a background DPI see section 5.8.5 of
        https://swift.gsfc.nasa.gov/analysis/bat_swguide_v6_3.pdf
    """

    def __init__(
            self,
            dpi_file=None,
            event_file=None,
            detector_quality_file=None,
            event_data=None,
            input_dict=None,
            recalc=False,
            load_dir=None,
            tmin=None,
            tmax=None,
            emin=None,
            emax=None,
    ):
        """
        This method initalizes the Detector Plane Image (DPI) data product. This can be initalized based on the
        creation of a DPI using heasoftpy's batbinevt or the direct binning of event data.

        :param dpi_file: None or a pathlib Path object for the full path to a DPI file that will be created with a call
            to heasoftpy batbinevt.
        :param event_file: None or path object of the event file that will be rebinned in a call to heasoftpy batbinevt
        :param detector_quality_file: Path object for the detector quality mask that was constructed for the associated
            event file
        :param event_data: None or TimeTaggedEvents class that has been initialized with event data
        :param input_dict: None or a dict of values that will be passed to batbinevt in the creation of the DPI.
            If a DPI is being read in from one that was previously created, the prior parameters that were used to
            calculate the DPI will be read in.
            If input_dict is None then it is set to
                dict(
                        infile=str(event_file),
                        outfile=str(dpi_file),
                        outtype="DPI",
                        energybins="14-195",
                        weighted="NO",
                        timedel=0,
                        tstart="INDEF",
                        tstop="INDEF",
                        clobber="YES",
                        timebinalg="uniform",
                    )
            by default which accumulates a histogram in an energy range of 14-195 keV and the start and end time of the
            event data.
        :param recalc: Boolean to denote if the DPI specified by dpi_file should be recalculated with the
            input_dict values (either those passed in or those that are defined by default)
        :param load_dir: Path of the directory that holds the DPI file that will be loaded in
        :param tmin: None or an astropy Quantity array of the beginning timebin edges
        :param tmax: None or an astropy Quantity array of the end timebin edges
        :param emin: None or an astropy Quantity array of the beginning of the energy bins
        :param emax: None or an astropy Quantity array of the end of the energy bins
        """

        if dpi_file is not None:
            self.dpi_file = Path(dpi_file).expanduser().resolve()

        # if any of these below are None, produce a warning that we wont be able to modify the spectrum. Also do
        # error checking for the files existing, etc
        if event_file is not None:
            self.event_file = Path(event_file).expanduser().resolve()
            if not self.event_file.exists():
                raise ValueError(
                    f"The specified event file {self.event_file} does not seem to exist. "
                    f"Please double check that it does."
                )
        else:
            self.event_file = None
            if event_data is None:
                warnings.warn(
                    "No event file has been specified. The resulting DPI object will not be able "
                    "to be arbitrarily modified either by rebinning in energy or time.",
                    stacklevel=2,
                )

        if detector_quality_file is not None:
            self.detector_quality_file = Path(detector_quality_file).expanduser().resolve()
            if not self.detector_quality_file.exists():
                raise ValueError(f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                                 f"to exist. Please double check that it does.")
        else:
            self.detector_quality_file = None
            # warnings.warn("No detector quality mask file has been specified. The resulting DPI object "
            #              "will not be able to be modified either by rebinning in energy or time.", stacklevel=2)

        # if ther is event data passed in then we can directly bin that otherwise need to see if we have to create a DPI
        # or load one in
        if event_data is None:
            if (not self.dpi_file.exists() or recalc) and self.event_file is not None:
                # we need to create the file, default is no mask weighting if we want to include that then we need the
                # image mask weight
                if input_dict is None:
                    self.dpi_input_dict = dict(
                        infile=str(self.event_file),
                        outfile=str(self.dpi_file),
                        detmask=str(self.detector_quality_file),
                        outtype="DPI",
                        energybins="14-195",
                        weighted="NO",
                        timedel=0,
                        tstart="INDEF",
                        tstop="INDEF",
                        clobber="YES",
                        timebinalg="uniform",
                    )

                else:
                    self.dpi_input_dict = input_dict

                # create the DPI
                self.bat_dpi_result = self._call_batbinevt(self.dpi_input_dict)

                # make sure that this calculation ran successfully
                if self.bat_dpi_result.returncode != 0:
                    raise RuntimeError(
                        f"The creation of the DPI failed with message: {self.bat_dpi_result.output}"
                    )

            else:
                self.dpi_input_dict = None

            self._parse_dpi_file()

            # properly format the DPI here the tbins and ebins attributes get overwritten but we dont care
            super().__init__(
                tmin=self.tbins["TIME_START"],
                tmax=self.tbins["TIME_STOP"],
                histogram_data=self.data["DPH_COUNTS"],
                emin=self.ebins["E_MIN"],
                emax=self.ebins["E_MAX"],
            )
        else:
            super().__init__(
                tmin=tmin,
                tmax=tmax,
                event_data=event_data,
                emin=emin,
                emax=emax,
            )

    @classmethod
    def from_file(cls, dpi_file, event_file=None):
        """
        This method allows one to load in a DPI that was provided in an observation ID or one that was previously
        created.

        :param dpi_file: Pathlib Path object for the full path to a DPI file that has been created with a call to
            heasoftpy's batbinevt
        :param event_file: None or an event file to also be loaded alongside the DPI. This will allow the DPI to be
            rebinned at any time or energy binning.
        :return: BatDPI object
        """

        dpi_file = Path(dpi_file).expanduser().resolve()

        if not dpi_file.exists():
            raise ValueError(
                f"The specified DPI file {dpi_file} does not seem to exist. "
                f"Please double check that it does."
            )

        # also make sure that the file is gunzipped which is necessary the user wants to do spectral fitting
        if ".gz" in dpi_file.suffix:
            with gzip.open(dpi_file, "rb") as f_in:
                with open(dpi_file.parent.joinpath(dpi_file.stem), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            dpi_file = dpi_file.parent.joinpath(dpi_file.stem)

        return cls(dpi_file, event_file)

    def _parse_dpi_file(self):
        """
        This method parses through a dpi file that has been created by batbinevent. The information included in
        the dpi file is read into the data attribute (which holds the dpi information itself,
        errors,  etc), the ebins attribute which holds the energybins associated with
        the dpi file, the tbins attibute with the time bin edges and the time bin centers of the dpi, the gti attribute
        which highlights the true times when the DPI data is good.

        NOTE: A special value of timepixr=-1 (the default used when constructing light curves) specifies that
              timepixr=0.0 for uniformly binned light curves and
              timepixr=0.5 for non-unformly binned light curves. We recommend using the unambiguous TIME_CENT in the
              tbins attribute to prevent any confusion instead of the data["TIME"] values

        :return: None
        """
        dpi_file = self.dpi_file
        dpi_data = []
        dpi_headers = []
        with fits.open(dpi_file) as f:
            # for DPI we can have:
            # 1) DPI(s) as extensions. Will need to read in all time and all energy bins
            # 2) DPI in the form of a table (like DPHs). In this case the times from the table will be replicated
            try:
                header = f["BAT_DPH"].header
                data = f["BAT_DPH"].data
                is_dpi_table = True
            except KeyError:
                is_dpi_table = False
                header = f[0].header
                for i in f:
                    if "BAT_DPI" in i.name:
                        dpi_data.append(i.data)
                        dpi_headers.append(i.header)

            energies = f["EBOUNDS"].data
            energies_header = f["EBOUNDS"].header
            try:
                times = f["GTI"].data
            except KeyError as ke:
                times = f["STDGTI"].data

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy
        # arrays as values. There can be a table iwth multiple DPIs or multiple extensions each with a DPI
        self.data = {}
        if is_dpi_table:
            for i in data.columns:
                try:
                    self.data[i.name] = u.Quantity(data[i.name], i.unit)
                except TypeError:
                    # if the user wants to read in any of the stuff in the class _exclude_data_cols variable this will take
                    # care of that.
                    self.data[i.name] = data[i.name]
        else:
            # get the unit (may not be the best way)
            image_unit = u.Quantity(0, unit=dpi_headers[0]["BUNIT"]).unit
            time_unit = u.Quantity(0, unit=dpi_headers[0]["TIMEUNIT"]).unit

            # set up the arrays
            self.data["DPH_COUNTS"] = np.zeros((len(dpi_data), *dpi_data[0].data.shape))
            self.data["TIME"] = np.zeros(len(dpi_data))
            self.data["EXPOSURE"] = np.zeros(len(dpi_data))

            # fill in arrays
            for i in range(len(dpi_data)):
                self.data["DPH_COUNTS"][i, :, :] = dpi_data[i].data
                self.data["TIME"][i] = dpi_headers[i]["TSTART"]
                self.data["EXPOSURE"][i] = dpi_headers[i]["TSTOP"] - dpi_headers[i]["TSTART"]

            # apply units
            self.data["DPH_COUNTS"] *= image_unit
            self.data["TIME"] *= time_unit
            self.data["EXPOSURE"] *= time_unit

        # fill in the energy bin info
        self.ebins = {}
        for i in energies.columns:
            if "CHANNEL" in i.name:
                self.ebins["INDEX"] = energies[i.name]
            elif "E" in i.name:
                self.ebins[i.name] = u.Quantity(energies[i.name], i.unit)

        # fill in the time info separately
        self.tbins = {}
        self.gti = {}
        # we have the good itme intervals for the DPI and also the time when the DPI were accumulated
        for i in times.columns:
            self.gti[f"TIME_{i.name}"] = u.Quantity(times[i.name], i.unit)
        self.gti["TIME_CENT"] = 0.5 * (self.gti[f"TIME_START"] + self.gti[f"TIME_STOP"])

        # now do the time bins for the dpis
        self.tbins["TIME_START"] = np.unique(self.data["TIME"])
        self.tbins["TIME_STOP"] = np.unique(self.data["TIME"] + self.data["EXPOSURE"])
        self.tbins["TIME_CENT"] = 0.5 * (
                self.tbins["TIME_START"] + self.tbins["TIME_STOP"]
        )

        # see if we need to reshape the DPIs for creation of the Histogram object
        if np.shape(self.data["DPH_COUNTS"]) != (
                self.tbins["TIME_START"].size,
                *self.data["DPH_COUNTS"].shape[1:],
                self.ebins["E_MIN"].size,
        ):
            new_array = (
                    np.zeros(
                        (
                            self.tbins["TIME_START"].size,
                            *self.data["DPH_COUNTS"].shape[1:],
                            self.ebins["E_MIN"].size,
                        ),
                    )
                    * self.data["DPH_COUNTS"].unit
            )

            for j in range(self.ebins["E_MIN"].size):
                new_array[:, :, :, j] = self.data["DPH_COUNTS"][
                                        j:: self.ebins["E_MIN"].size, :, :
                                        ]

            self.data["DPH_COUNTS"] = new_array

        # if self.dpi_input_dict ==None, then we will need to try to read in the hisotry of parameters passed into
        # batbinevt to create the dpi file. thsi usually is needed when we first parse a file so we know what things
        # are if we need to do some sort of rebinning.

        # were looking for something like:
        #   START PARAMETER list for batbinevt_1.48 at 2023-11-16T18:47:52
        #
        #   P1 infile = /Users/tparsota/Documents/01116441000_eventresult/events/sw0
        #   P1 1116441000bevshsp_uf.evt
        #   P2 outfile = /Users/tparsota/Documents/01116441000_eventresult/pha/spect
        #   P2 rum_0.pha
        #   P3 outtype = PHA
        #   P4 timedel = 0.0
        #   P5 timebinalg = uniform
        #   P6 energybins = CALDB
        #   P7 gtifile = NONE
        #   P8 ecol = ENERGY
        #   P9 weighted = YES
        #   P10 outunits = INDEF
        #   P11 timepixr = -1.0
        #   P12 maskwt = NONE
        #   P13 tstart = INDEF
        #   P14 tstop = INDEF
        #   P15 snrthresh = 6.0
        #   P16 detmask = /Users/tparsota/Documents/01116441000_eventresult/auxil/sw
        #   P16 01116441000bdqcb.hk.gz
        #   P17 tcol = TIME
        #   P18 countscol = DPI_COUNTS
        #   P19 xcol = DETX
        #   P20 ycol = DETY
        #   P21 maskwtcol = MASK_WEIGHT
        #   P22 ebinquant = 0.1
        #   P23 delzeroes = no
        #   P24 minfracexp = 0.1
        #   P25 min_dph_frac_overlap = 0.999
        #   P26 min_dph_time_overlap = 0.0
        #   P27 max_dph_time_nonoverlap = 0.5
        #   P28 buffersize = 16384
        #   P29 clobber = yes
        #   P30 chatter = 2
        #   P31 history = yes
        #   P32 mode = ql
        #   END PARAMETER list for batbinevt_1.48

        if self.dpi_input_dict is None:
            # get the default names of the parameters for batbinevt including its name 9which should never change)
            test = hsp_core.HSPTask("batbinevt")
            default_params_dict = test.default_params.copy()
            taskname = test.taskname
            start_processing = None

            for i in header["HISTORY"]:
                if taskname in i and start_processing is None:
                    # then set a switch for us to start looking at things
                    start_processing = True
                elif taskname in i and start_processing is True:
                    # we want to stop processing things
                    start_processing = False

                if start_processing and "START" not in i and len(i) > 0:
                    values = i.split(" ")
                    # print(i, values, "=" in values)

                    parameter_num = values[0]
                    parameter = values[1]
                    if "=" not in values:
                        # this belongs with the previous parameter and is a line continuation
                        default_params_dict[old_parameter] = (
                                default_params_dict[old_parameter] + values[-1]
                        )
                        # assume that we need to keep appending to the previous parameter
                    else:
                        default_params_dict[parameter] = values[-1]

                        old_parameter = parameter

            self.dpi_input_dict = default_params_dict.copy()

    @u.quantity_input(energybins=["energy"], emin=["energy"], emax=["energy"])
    def set_energybins(self, energybins=None, emin=None, emax=None):
        """

        NOTE: This is copied from the BatDPH class, this can be done better
        :param energybins: astropy Quantity object outlining the energy bin edges that the DPI will be binned into
        :param emin: an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the user would like. NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param emax: an astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            energy bins that the user would like. It shoudl have the same number of elements as emin.
            NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :return: None
        """

        if self.event_file is None:
            super().set_energybins(energybins=energybins, emin=emin, emax=emax)

            # if we have event data that is binned directly we dont have the data attribute
            try:
                self.data["DPH_COUNTS"] = self.contents
            except AttributeError as ae:
                pass
        else:
            # if we have read in a file (and passed in a numpy array that is a histogram)
            # then call batbinevt and parse newly created file
            # NOTE: this is mostly copied from the Lightcurve object
            # see if the user specified either the energy bins directly or emin/emax separately
            if emin is None and emax is None:
                # need to get emin and emax values, assume that these are in keV already when converting to astropy quantities
                emin = energybins[:-1]
                emax = energybins[1:]

            else:
                # make sure that both emin and emax are defined and have the same number of elements
                if (emin is None and emax is not None) or (emax is None and emin is not None):
                    raise ValueError('Both emin and emax must be defined.')

                if emin.size != emax.size:
                    raise ValueError('Both emin and emax must have the same length.')

                if emin.shape == ():
                    emin = u.Quantity([emin])
                    emax = u.Quantity([emax])

            # create our energybins input to batbinevt
            energybins = []
            for min, max in zip(emin.to(u.keV), emax.to(u.keV)):
                energybins.append(f"{min.value}-{max.value}")

            # create the full string
            ebins = ','.join(energybins)

            # create a temp dict to hold the energy rebinning parameters to pass to heasoftpy. If things dont run
            # successfully then the updated parameter list will not be saved
            tmp_dpi_input_dict = self.dpi_input_dict.copy()

            # need to see if the energybins are different (and even need to be calculated), if so do the recalculation
            if not np.array_equal(emin, self.ebins['E_MIN']) or not np.array_equal(emax, self.ebins['E_MAX']):
                # the tmp_lc_input_dict wil need to be modified with new Energybins
                tmp_dpi_input_dict["energybins"] = ebins

                # the DPH _call_batbinevt method ensures that  outtype = DPH and that clobber=YES
                dpi_return = self._call_batbinevt(tmp_dpi_input_dict)

                # make sure that the dph_return was successful
                if dpi_return.returncode != 0:
                    raise RuntimeError(f'The creation of the DPI failed with message: {dpi_return.output}')
                else:
                    self.bat_dpi_result = dpi_return
                    self.dpi_input_dict = tmp_dpi_input_dict

                    # reparse the DPH file to get the info
                    self._parse_dpi_file()

                    # properly format the DPH here the tbins and ebins attributes get overwritten but we dont care
                    super().__init__(
                        tmin=self.tbins["TIME_START"],
                        tmax=self.tbins["TIME_STOP"],
                        histogram_data=self.data["DPH_COUNTS"],
                        emin=self.ebins["E_MIN"],
                        emax=self.ebins["E_MAX"],
                    )

    @u.quantity_input(timebins=["time"], tmin=["time"], tmax=["time"])
    def set_timebins(self, timebins=None, tmin=None, tmax=None, timebinalg="uniform", T0=None, is_relative=False,
                     timedelta=np.timedelta64(0, 's')):
        """
        This method allows for the dynamic rebinning of the DPI in time.

        NOTE: This is copied from the BatDPH class, this can be done better

        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True. If this parameter is passed in, then it supercedes the values of
            tmin and tmax.
        :param tmin: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the DPI to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if the timebins parameter is passed in then anything passed into tmin/tmax is ignored
        :param tmax: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the DPI to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified and the is_relative parameter should be True.
            NOTE: if the timebins parameter is passed in then anything passed into tmin/tmax is ignored
        :param timebinalg: string that can be "uniform or "snr" to specify the type of timebinning algorithm we may want
            to specify for batbinevt (see related documentation: https://heasarc.gsfc.nasa.gov/ftools/caldb/help/batbinevt.html)
            This is only relevant if the user  does not pass in any values for the timebins/tmin/tmax parameters.
            In this case, the method will bin the DPIs starting from the start time of the event file to the end time
            in the event file in time bin widths correspondent to the timedelta parameter below.
            In the case where tmin/tmax is specified and they are single values, the default behavior is to create a
            single DPI that is collected over the time interval specified by tmin/tmax.
        :param T0: float or an astropy.units.Quantity object with some time of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tmin/tmax that were passed in.
        :param timedelta: a numpy timedelta object that specifies the uniform/snr timebinning that may be used
            if the timebinalg parameter is passed to batbinevt. The default value of timebin=np.timedelta64(0, "s")
            accumulates the whole event dataset into a single DPI.
        :return: None

        """

        if type(is_relative) is not bool:
            raise ValueError("The is_relative parameter should be a boolean value.")

        if is_relative and T0 is None:
            raise ValueError('The is_relative value is set to True however there is no T0 that is defined ' +
                             '(ie the time from which the time bins are defined relative to is not specified).')

        # we can either rebin using the timebins that are already present in the histogram
        # OR we can rebin the event data
        # where the event data can be rebinned directly through the histogram object or through the batbinevt script)
        if self.event_file is None:
            if is_relative:
                if timebins is not None:
                    # see if T0 is Quantity class
                    if type(T0) is u.Quantity:
                        timebins += T0
                    else:
                        timebins += T0 * u.s
                else:
                    if type(T0) is u.Quantity:
                        tmin += T0
                        tmax += T0
                    else:
                        tmin += T0 * u.s
                        tmax += T0 * u.s

            super().set_timebins(timebins=timebins, tmin=tmin, tmax=tmax)

            # if we have event data that is binned directly we dont have the data attribute
            try:
                self.data["DPH_COUNTS"] = self.contents

                # set the time key again as well
                self.data["TIME"] = self.tbins["TIME_START"]

                # also set the exposure as well
                self.data["EXPOSURE"] = self.exposure
            except AttributeError as ae:
                pass
        else:
            # do error checking on tmin/tmax
            if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
                raise ValueError('Both tmin and tmax must be defined.')

            if tmin is not None and tmax is not None:
                if tmin.size != tmax.size:
                    raise ValueError('Both tmin and tmax must have the same length.')

            tmp_dpi_input_dict = self.dpi_input_dict.copy()

            # create a copy of the timebins if it is not None to prevent modifying the original array
            if timebins is not None:
                timebins = timebins.copy()

            if tmin is not None and tmax is not None:
                # now try to construct single array of all timebin edges in seconds
                timebins = np.zeros(tmin.size + 1) * u.s
                timebins[:-1] = tmin
                if tmin.size > 1:
                    timebins[-1] = tmax[-1]
                else:
                    timebins[-1] = tmax

            # See if we need to add T0 to everything
            if is_relative:
                # see if T0 is Quantity class
                if type(T0) is u.Quantity:
                    timebins += T0
                else:
                    timebins += T0 * u.s

            if (timebins is not None and timebins.size > 2):
                # tmin is not None and tmax.size > 1 and
                # already checked that tmin && tmax are not 1 and have the same size
                # if they are defined and they are more than 1 element then we have a series of timebins otherwise we just have the

                tmp_dpi_input_dict['tstart'] = "INDEF"
                tmp_dpi_input_dict['tstop'] = "INDEF"

                # start/stop times of the lightcurve
                self.timebins_file = self._create_custom_timebins(timebins)
                tmp_dpi_input_dict['timebinalg'] = "gti"
                tmp_dpi_input_dict['gtifile'] = str(self.timebins_file)
            else:
                tmp_dpi_input_dict['gtifile'] = "NONE"

                # should have everything that we need to do the rebinning for a uniform/snr related rebinning
                # first need to update the tmp_lc_input_dict
                if "uniform" in timebinalg or "snr" in timebinalg:
                    tmp_dpi_input_dict['timebinalg'] = timebinalg

                tmp_dpi_input_dict['timedel'] = timedelta / np.timedelta64(1, 's')  # convert to seconds

                tmp_dpi_input_dict['tstart'] = "INDEF"
                tmp_dpi_input_dict['tstop'] = "INDEF"

                # see if we have the min/max times defined
                if (tmin is not None and tmax.size == 1):
                    tmp_dpi_input_dict['timedel'] = 0
                    tmp_dpi_input_dict['tstart'] = timebins[0].value
                    tmp_dpi_input_dict['tstop'] = timebins[1].value

            # the DPH _call_batbinevt method ensures that  outtype = DPH and that clobber=YES
            dpi_return = self._call_batbinevt(tmp_dpi_input_dict)

            # make sure that the dph_return was successful
            if dpi_return.returncode != 0:
                raise RuntimeError(f'The creation of the DPI failed with message: {dpi_return.output}')
            else:
                self.bat_dpi_result = dpi_return
                self.dpi_input_dict = tmp_dpi_input_dict

                # reparse the DPH file to get the info
                self._parse_dpi_file()

                # properly format the DPH here the tbins and ebins attributes get overwritten but we dont care
                super().__init__(
                    tmin=self.tbins["TIME_START"],
                    tmax=self.tbins["TIME_STOP"],
                    histogram_data=self.data["DPH_COUNTS"],
                    emin=self.ebins["E_MIN"],
                    emax=self.ebins["E_MAX"],
                )

    def _call_batbinevt(self, input_dict):
        """
        Calls heasoftpy's batbinevt with an error wrapper, ensures that this bins the event data to produce a DPI
        NOTE: This is copied from the BatDPH class, this can be done better

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batbinevt
        :return: heasoftpy Result object from batbinevt
        """

        input_dict["clobber"] = "YES"
        input_dict["outtype"] = "DPI"

        try:
            return hsp.batbinevt(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"The call to Heasoft batbinevt failed with inputs {input_dict}."
            )

    def _create_custom_timebins(self, timebins, output_file=None):
        """
        This method creates custom time bins from a user defined set of time bin edges. The created fits file with the
        timebins of interest will by default have the same name as the dpi file, however it will have a "gti" suffix
        instead of a "dpi" suffix and it will be stored in the gti subdirectory of the event results directory.

        Note: This method is here so the call to create a gti file with custom timebins can be phased out eventually.
        NOTE: This is copied from the BatDPH class, this can be done better

        :param timebins: a astropy.unit.Quantity object with the edges of the timebins that the user would like
        :param output_file: None or a Path object to where the output *.gti file will be saved to. A value of None
            defaults to the above description
        :return: Path object of the created good time intervals file
        """

        if output_file is None:
            # use the same filename as for the dph file but replace suffix with gti and put it in gti subdir instead of survey
            new_path = self.dpi_file.parts
            new_name = self.dpi_file.name.replace("dpi", "gti")
            try:
                # tryto put it in the gti directory
                output_file = Path(*new_path[:self.dpi_file.parts.index('dpi')]).joinpath("gti").joinpath(
                    new_name)

            except ValueError:
                # otherwise just try to place it where the dph is with the gti suffix
                output_file = self.dpi_file.parent.joinpath(new_name)
        return create_gti_file(timebins, output_file, T0=None, is_relative=False, overwrite=True)
