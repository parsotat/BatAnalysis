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

from .bat_dph import DetectorPlaneHistogram

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatDPI(DetectorPlaneHistogram):
    """
    This class encapsulates the BAT Detector Plane Image data product. This is a Detector Plane Histogram
    that has had batsurvey-erebin and baterebin and select good time intervals batsurvey-gti applied to it.

    This can also be constructed by batbinevt which this class handles. There can be multiple DPIs extensions in a file
    or there can be a table with multiple DPIs
    """

    def __init__(
            self,
            dpi_file=None,
            event_file=None,
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
        :param event_data: None or Event data dictionary or event data class (to be created)
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
        :param emin: None or an or an astropy Quantity array of the beginning of the energy bins
        :param emax: None or an or an astropy Quantity array of the end of the energy bins
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
        with fits.open(dpi_file) as f:
            # for DPI we can have:
            # 1) DPI(s) as extensions. Will need to read in all time and all energy bins
            # 2) DPI in the form of a table (like DPHs). In this case the times from the table will be replicated
            header = f[1].header
            data = f[1].data
            energies = f["EBOUNDS"].data
            energies_header = f["EBOUNDS"].header
            try:
                times = f["GTI"].data
            except KeyError as ke:
                times = f["STDGTI"].data

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy
        # arrays as values. There can be a table iwth multiple DPIs or multiple extensions each with a DPI
        self.data = {}
        for i in data.columns:
            try:
                self.data[i.name] = u.Quantity(data[i.name], i.unit)
            except TypeError:
                # if the user wants to read in any of the stuff in the class _exclude_data_cols variable this will take
                # care of that.
                self.data[i.name] = data[i.name]

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
            new_array = np.zeros(
                (
                    self.tbins["TIME_START"].size,
                    *self.data["DPH_COUNTS"].shape[1:],
                    self.ebins["E_MIN"].size,
                ),
            ) * self.data["DPH_COUNTS"].unit

            for j in range(self.ebins["E_MIN"].size):
                new_array[:, :, :, j] = self.data["DPH_COUNTS"][j:: self.ebins["E_MIN"].size, :, :]

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
            test = hsp.HSPTask("batbinevt")
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
