"""
This file holds the BAT Detector plane histogram class

Tyler Parsotan Feb 21 2024
"""

import gzip
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from histpy import Histogram
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class DetectorPlaneHistogram(Histogram):
    """
    This class encapsulates data that is spread across the detector plane in an image like format. It allows for the
    rebinning of data in space, time, and/or energy.
    """

    det_x_edges = np.arange(286 + 1) - 0.5
    det_y_edges = np.arange(173 + 1) - 0.5

    @u.quantity_input(
        timebins=["time"],
        tmin=["time"],
        tmax=["time"],
        energybins=["energy"],
        emin=["energy"],
        emax=["energy"],
    )
    def __init__(
            self,
            event_data=None,
            histogram_data=None,
            timebins=None,
            tmin=None,
            tmax=None,
            energybins=None,
            emin=None,
            emax=None,
            weights=None,
    ):
        """
        This class can be initiated from individual event data that gets histogrammed in space/time/energy. It can be
        initalized from a default BAT DPH.

        By default, the spatial binning is in the detector x and y bins specified with the class det_x_edges and
        det_y_edges.

        The times/energybins do not need to be continuous but the constructor will create timebins/energybins for the
        unspecified times where the count across the detector plane is 0

        :param event_data:
        :param histogram_data:
        :param timebins:
        :param tmin:
        :param tmax:
        :param energybins:
        :param emin:
        :param emax:
        :param weights:
        """

        # do some error checking
        if event_data is None and histogram_data is None:
            raise ValueError(
                "Either event data or a histogram needs to be passed in to initalize a DetectorPlaneHistogram object"
            )

        if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
            raise ValueError("Both tmin and tmax must be defined.")

        if tmin is not None and tmax is not None:
            if tmin.size != tmax.size:
                raise ValueError("Both tmin and tmax must have the same length.")

        # determine the type of data that we have
        if event_data is not None:
            parse_data = deepcopy(event_data)
        else:
            parse_data = deepcopy(histogram_data)

        # determine the time binnings
        # can have dfault time binning be the start/end time of the event data or the times passed in by default
        # from a potential histpy Histogram object.
        # can also have timebins passed in with contiguous timebinning
        # or tmin/tmax passed in where the times are not continuous due to good time intervals etc
        if timebins is None and tmin is None and tmax is None:
            if event_data is not None:
                timebin_edges = u.Quantity(
                    [event_data["TIME"].min(), event_data["TIME"].max()]
                )
            else:
                if not isinstance(histogram_data, Histogram):
                    # if we dont have a histpy histogram, need to have the timebins
                    raise ValueError(
                        "For a general histogram that has been passed in, the timebins need to be specified"
                    )
                else:
                    timebin_edges = histogram_data.axes["TIME"].edges
        elif timebins is not None:
            timebin_edges = timebins
        else:
            # need to determine if the timebins are contiguous
            if np.all(tmin[1:] == tmax[:-1]):
                # if all the timebins are not continuous, then need to add buffer timebins to make them so
                combined_edges = np.concatenate((tmin, tmax))
                timebin_edges = np.unique(np.sort(combined_edges))

            else:
                # concatenate the tmin/tmax and sort and then select the unique values. This fill in all the gaps that we
                # may have had due to GTIs etc. Now we need to modify the input histogram if there was one
                combined_edges = np.concatenate((tmin, tmax))
                final_timebins = np.unique(np.sort(combined_edges))

                if histogram_data is not None:
                    idx = np.searchsorted(final_timebins[:-1], tmin)

                    # get the new array size
                    new_hist = np.zeros(
                        (final_timebins.size - 1, *histogram_data.shape[1:])
                    )
                    new_hist[idx, :] = histogram_data

                    parse_data = new_hist

                timebin_edges = final_timebins

        # determine the energy binnings
        if energybins is None and emin is None and emax is None:
            if event_data is not None:
                energybin_edges = u.Quantity(
                    [event_data["ENERGY"].min(), event_data["ENERGY"].max()]
                )
            else:
                if not isinstance(histogram_data, Histogram):
                    # if we dont have a histpy histogram, need to have the energybins
                    raise ValueError(
                        "For a general histogram that has been passed in, the energybins need to be specified"
                    )
                else:
                    energybin_edges = histogram_data.axes["ENERGY"].edges
        elif energybins is not None:
            energybin_edges = energybins
        else:
            # need to determine if the energybins are contiguous
            if np.all(emin[1:] == emax[:-1]):
                # if all the energybins are not continuous combine them directly
                combined_edges = np.concatenate((emin, emax))
                energybin_edges = np.unique(np.sort(combined_edges))

            else:
                # concatenate the emin/emax and sort and then select the unique values. This fill in all the gaps that we
                # may have had. Now we need to modify the input histogram if there was one
                combined_edges = np.concatenate((emin, emax))
                final_energybins = np.unique(np.sort(combined_edges))

                if histogram_data is not None:
                    idx = np.searchsorted(final_energybins[:-1], emin)

                    # get the new array size
                    new_hist = np.zeros(
                        (
                            *parse_data.shape[:-1],
                            final_energybins.size - 1,
                        )
                    )
                    new_hist[idx, :] = parse_data

                    parse_data = new_hist

                energybin_edges = final_energybins

        # get the good time intervals, the time intervals for the histogram, the energy intervals for the histograms as well
        # these need to be set for us to create the histogram edges
        self.gti = {}
        if tmin is not None:
            self.gti["TIME_START"] = tmin
            self.gti["TIME_STOP"] = tmax
        else:
            self.gti["TIME_START"] = timebin_edges[:-1]
            self.gti["TIME_STOP"] = timebin_edges[1:]
        self.gti["TIME_CENT"] = 0.5 * (self.gti["TIME_START"] + self.gti["TIME_STOP"])

        self.exposure = self.gti["TIME_STOP"] - self.gti["TIME_START"]

        self.tbins = {}
        self.tbins["TIME_START"] = timebin_edges[:-1]
        self.tbins["TIME_STOP"] = timebin_edges[1:]
        self.tbins["TIME_CENT"] = 0.5 * (
                self.tbins["TIME_START"] + self.tbins["TIME_STOP"]
        )

        self.ebins = {}
        if emin is not None:
            self.ebins["INDEX"] = np.arange(emin.size) + 1
            self.ebins["E_MIN"] = emin
            self.ebins["E_MAX"] = emax
        else:
            self.ebins["INDEX"] = np.arange(energybin_edges.size - 1) + 1
            self.ebins["E_MIN"] = energybin_edges[:-1]
            self.ebins["E_MAX"] = energybin_edges[1:]

        if histogram_data is not None:
            self._set_histogram(histogram_data=parse_data)
        else:
            self._set_histogram(event_data=parse_data)

    def _set_histogram(self, histogram_data=None, event_data=None, weights=None):
        """

        :param histogram_data:
        :param event_data:
        :param weights:
        :return:
        """
        # get the timebin edges
        timebin_edges = (
                np.zeros(self.tbins["TIME_START"].size + 1) * self.tbins["TIME_START"].unit
        )
        timebin_edges[:-1] = self.tbins["TIME_START"]
        timebin_edges[-1] = self.tbins["TIME_STOP"][-1]

        # get the energybin edges
        energybin_edges = (
                np.zeros(self.ebins["E_MIN"].size + 1) * self.ebins["E_MIN"].unit
        )
        energybin_edges[:-1] = self.ebins["E_MIN"]
        energybin_edges[-1] = self.ebins["E_MAX"][-1]

        # create our histogrammed data
        if histogram_data is not None:
            if isinstance(histogram_data, u.Quantity):
                hist_unit = histogram_data.unit
            else:
                hist_unit = u.count

            if not isinstance(histogram_data, Histogram):
                super().__init__(
                    [
                        timebin_edges,
                        self.det_y_edges,
                        self.det_x_edges,
                        energybin_edges,
                    ],
                    contents=histogram_data,
                    labels=["TIME", "DETY", "DETX", "ENERGY"],
                    sumw2=weights,
                    unit=hist_unit,
                )
            else:
                super().__init__(
                    [i.edges for i in histogram_data.axes],
                    contents=histogram_data.contents,
                    labels=histogram_data.axes.labels,
                    unit=hist_unit,
                )

        else:
            super().__init__(
                [timebin_edges, self.det_y_edges, self.det_x_edges, energybin_edges],
                labels=["TIME", "DETY", "DETX", "ENERGY"],
                sumw2=weights is not None,
                unit=u.count,
            )
            self.fill(
                event_data["TIME"],
                event_data["DETY"].value,
                event_data["DETX"].value,
                event_data["ENERGY"],
                weight=weights,
            )

    @u.quantity_input(timebins=["time"], tmin=["time"], tmax=["time"])
    def set_timebins(
            self,
            timebins=None,
            tmin=None,
            tmax=None,
    ):
        """

        :param timebins:
        :param tmin:
        :param tmax:
        :return:
        """

        # first make sure that we have a time binning axis of our histogram
        if "TIME" not in self.axes.labels or self.axes["TIME"].nbins == 1:
            raise ValueError(
                "The DPH either has  not timing information or there is only one time bin which means that"
                "the DPH cannot be rebinned in time."
            )

        # create a copy of the timebins if it is not None to prevent modifying the original array
        if timebins is not None:
            timebins = timebins.copy()

            tmin = timebins[:-1]
            tmax = timebins[1:]

        # check the inputs
        # do error checking on tmin/tmax
        if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
            raise ValueError("Both tmin and tmax must be defined.")

        if tmin is not None and tmax is not None:
            if tmin.size != tmax.size:
                raise ValueError("Both tmin and tmax must have the same length.")

        # make sure that tmin/tmax can be iterated over
        if tmin.shape == ():
            tmin = u.Quantity([tmin])
            tmax = u.Quantity([tmax])

        # make sure that the times exist in the array of timebins that the DPH has been binned into
        for s, e in zip(tmin, tmax):
            if not np.all(np.in1d(s, self.tbins["TIME_START"])) or not np.all(
                    np.in1d(e, self.tbins["TIME_STOP"])
            ):
                raise ValueError(
                    f"The requested time binning from {s}-{e} is not encompassed by the current timebins "
                    f"of the histogrammed data. Please choose the closest TIME_START and TIME_STOP values from"
                    f" the tbins or gti attributes"
                )

        # do the rebinning along those dimensions and modify the appropriate attibutes. We cannot use the normal
        # Histogram methods since we dont have continuous time bins. If needed, can ensure that only the good DPHs are
        # included in the rebinning
        new_hist_size = (len(tmin), *self.nbins[1:])
        histograms = np.zeros(new_hist_size)

        for i, s, e in zip(np.arange(tmin.size), tmin, tmax):
            idx = np.where(
                (self.tbins["TIME_START"] >= s) & (self.tbins["TIME_STOP"] <= e)
            )[0]
            histograms[i, :] = self.contents[idx].sum(axis=0)

        # save the new time bins
        self.tbins["TIME_START"] = tmin
        self.tbins["TIME_STOP"] = tmax
        self.tbins["TIME_CENT"] = 0.5 * (
                self.tbins[f"TIME_START"] + self.tbins[f"TIME_STOP"]
        )

        # get the intersection with the good time intervals for us to keep track
        idx = np.where(
            (self.tbins["TIME_START"] <= self.gti["TIME_STOP"])
            & (self.tbins["TIME_STOP"] >= self.gti["TIME_STOP"])
        )
        for i in self.gti.keys():
            self.gti[i] = self.gti[i][idx]

        # now we can reinitalize the info
        self._set_histogram(histogram_data=histograms)

        return None

    @u.quantity_input(energybins=["energy"], emin=["energy"], emax=["energy"])
    def set_energybins(self, energybins=None, emin=None, emax=None):
        """
        This method allows for the DPH(s) to be rebinned to different energy binnings. Here we require the enerybins to
        be contiguous.

        Could have energybins=[14.0, 20.0, 24.0, 35.0, 50.0, 75.0, 100.0, 150.0, 195.0]*u.keV, but the erebin batsurvey
        has to be run first to correct for energy.
        """

        # first make sure that we have a energy binning axis of our histogram
        if "ENERGY" not in self.axes.labels or self.axes["ENERGY"].nbins == 1:
            raise ValueError(
                "The DPH either has  no energy information or there is only one energy bin which means that"
                "the DPH cannot be rebinned in energy."
            )

        # do error checking on tmin/tmax
        if (emin is None and emax is not None) or (emin is None and emax is not None):
            raise ValueError("Both emin and emax must be defined.")

        if emin is not None and emax is not None:
            if emin.size != emax.size:
                raise ValueError("Both emin and emax must have the same length.")
        else:
            if energybins is not None:
                # create a copy of the enerbins if it is not None to prevent modifying the original array
                energybins = energybins.copy()

                emin = energybins[:-1]
                emax = energybins[1:]
            else:
                raise ValueError("No energy bins have been specified.")

        # make sure that emin/emax can be iterated over
        if emin.shape == ():
            emin = u.Quantity([emin])
            emax = u.Quantity([emax])

        # make sure that the energies exist in the array of energybins that the DPH has been binned into
        for s, e in zip(emin, emax):
            if not np.all(np.in1d(s, self.ebins["E_MIN"])) or not np.all(
                    np.in1d(e, self.ebins["E_MAX"])
            ):
                raise ValueError(
                    f"The requested energy binning from {s}-{e} is not encompassed by the current energybins "
                    f"of the loaded DPH. Please choose the closest E_MIN and E_MAX values from"
                    f" the ebin attribute"
                )

        # do the rebinning along those dimensions, here dont need to modify the appropriate attibutes since they dont
        # depend on energy. We cannot use the normal
        # Histogram methods since we may have non-uniform energy bins.
        new_hist_size = (*self.nbins[:-1], len(emin))
        histograms = np.zeros(new_hist_size)

        for i, s, e in zip(np.arange(emin.size), emin, emax):
            idx = np.where((self.ebins["E_MIN"] >= s) & (self.ebins["E_MAX"] <= e))[0]
            histograms[:, :, :, i] = self.contents[:, :, :, idx].sum(axis=-1)

        # set the new ebin attrubute
        self.ebins["E_MIN"] = emin
        self.ebins["E_MAX"] = emax
        self.ebins["INDEX"] = np.arange(len(emin)) + 1

        # now we can reinitalize the info
        self._set_histogram(histogram_data=histograms)

        return None

    @u.quantity_input(emin=["energy"], emax=["energy"], tmin=["time"], tmax=["time"])
    def plot(self, emin=None, emax=None, tmin=None, tmax=None, plot_rate=False):
        """
        This method allows the user to conveniently plot the DPH for a single energy bin and time interval.

        :param emin:
        :param emax:
        :return:
        """

        if emin is None and emax is None:
            plot_emin = self.ebins["E_MIN"].min()
            plot_emax = self.ebins["E_MAX"].max()
        elif emin is not None and emax is not None:
            plot_emin = emin
            plot_emax = emax
        else:
            raise ValueError(
                "emin and emax must either both be None or both be specified."
            )
        plot_e_idx = np.where(
            (self.ebins["E_MIN"] >= plot_emin) & (self.ebins["E_MAX"] <= plot_emax)
        )[0]

        if tmin is None and tmax is None:
            plot_tmin = self.tbins["TIME_START"].min()
            plot_tmax = self.tbins["TIME_STOP"].max()
        elif tmin is not None and tmax is not None:
            plot_tmin = tmin
            plot_tmax = tmax
        else:
            raise ValueError(
                "tmin and tmax must either both be None or both be specified."
            )
        plot_t_idx = np.where(
            (self.tbins["TIME_START"] >= plot_tmin)
            & (self.tbins["TIME_STOP"] <= plot_tmax)
        )[0]

        # now start to accumulate the DPH counts based on the time and energy range that we care about
        plot_data = self.contents[plot_t_idx, :, :, :]

        if len(plot_t_idx) > 0:
            plot_data = plot_data.sum(axis=0)
        else:
            raise ValueError(
                f"There are no DPH time bins that fall between {plot_tmin} and {plot_tmax}"
            )

        plot_data = plot_data[:, :, plot_e_idx]

        if len(plot_e_idx) > 0:
            plot_data = plot_data.sum(axis=-1)
        else:
            raise ValueError(
                f"There are no DPH energy bins that fall between {plot_emin} and {plot_emax}"
            )

        if plot_rate:
            # calcualte the totoal exposure
            exposure_tot = np.sum(self.exposure[plot_t_idx])
            plot_data /= exposure_tot

        # set any 0 count detectors to nan so they get plotted in black
        # this includes detectors that are off and "space holders" between detectors where the value is 0
        plot_data[plot_data == 0] = np.nan

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cmap = mpl.colormaps.get_cmap("viridis")
        cmap.set_bad(color="k")

        im = ax.imshow(plot_data.value, origin="lower", interpolation="none", cmap=cmap)

        fig.colorbar(im, cax=cax, orientation="vertical", label=plot_data.unit)

        return fig, ax


class BatDPH(DetectorPlaneHistogram):
    """
    This class encapsulates the BAT detector plane historgrams (DPH) that are collected onboard. It also allows for the
    creation of a DPH from event data and rebinning in time/energy.

    TODO: to get DPI need to apply batsurvey-erebin and baterebin and select good time intervals batsurvey-gti
    """

    # this class variable states which columns form the DPH files should be excluded from being read in
    _exclude_data_cols = [
        "GAIN_INDEX",
        "OFFSET_INDEX",
        "LDPNAME",
        "BLOCK_MAP",
        "NUM_DETS",
        "APID",
        "LDP",
    ]

    def __init__(
            self,
            dph_file,
            event_file,
            input_dict=None,
            recalc=False,
            verbose=False,
            load_dir=None,
    ):
        """

        :param dph_file:
        :param event_file:
        """
        self.dph_file = Path(dph_file).expanduser().resolve()

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
            warnings.warn(
                "No event file has been specified. The resulting DPH object will not be able "
                "to be arbitrarily modified either by rebinning in energy or time.",
                stacklevel=2,
            )

        # if ther is no event file we just have the instrument produced DPH or a previously calculated one
        # if self.event_file is None:
        #    self.dph_input_dict = None

        if (not self.dph_file.exists() or recalc) and self.event_file is not None:
            # we need to create the file, default is no mask weighting if we want to include that then we need the
            # image mask weight
            if input_dict is None:
                self.dph_input_dict = dict(
                    infile=str(self.event_file),
                    outfile=str(self.dph_file),
                    outtype="DPH",
                    energybins="14-195",
                    weighted="NO",
                    timedel=0,
                    tstart="INDEF",
                    tstop="INDEF",
                    clobber="YES",
                    timebinalg="uniform",
                )

            else:
                self.dph_input_dict = input_dict

            # create the DPH
            self.bat_dph_result = self._call_batbinevt(self.dph_input_dict)

            # make sure that this calculation ran successfully
            if self.bat_dph_result.returncode != 0:
                raise RuntimeError(
                    f"The creation of the DPH failed with message: {self.bat_dph_result.output}"
                )

        else:
            self.dph_input_dict = None

        self._parse_dph_file()

        # properly format the DPH here the tbins and ebins attributes get overwritten but we dont care
        super().__init__(
            tmin=self.tbins["TIME_START"],
            tmax=self.tbins["TIME_STOP"],
            histogram_data=self.data["DPH_COUNTS"],
            emin=self.ebins["E_MIN"],
            emax=self.ebins["E_MAX"],
        )

    @classmethod
    def from_file(cls, dph_file, event_file=None):
        """
        This method allows one to load in a DPH that was provided in an observation ID or one that was previously
        created.

        :param dph_file:
        :param event_file:
        :return:
        """

        dph_file = Path(dph_file).expanduser().resolve()

        if not dph_file.exists():
            raise ValueError(
                f"The specified DPH file {dph_file} does not seem to exist. "
                f"Please double check that it does."
            )

        # also make sure that the file is gunzipped which is necessary the user wants to do spectral fitting
        if ".gz" in dph_file.suffix:
            with gzip.open(dph_file, "rb") as f_in:
                with open(dph_file.parent.joinpath(dph_file.stem), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            dph_file = dph_file.parent.joinpath(dph_file.stem)

        return cls(dph_file, event_file)

    def _parse_dph_file(self):
        """
        This method parses through a pha file that has been created by batbinevent. The information included in
        the pha file is read into the RA/DEC attributes (and checked to make sure that this is the pha that
        the user wants to load in), the data attribute (which holds the pha information itself including rates/counts,
        errors,  etc), the ebins attribute which holds the energybins associated with
        the pha file, the tbins attibute with the time bin edges and the time bin centers

        NOTE: A special value of timepixr=-1 (the default used when constructing light curves) specifies that
              timepixr=0.0 for uniformly binned light curves and
              timepixr=0.5 for non-unformly binned light curves. We recommend using the unambiguous TIME_CENT in the
              tbins attribute to prevent any confusion instead of the data["TIME"] values

        :return: None
        """
        dph_file = self.dph_file
        with fits.open(dph_file) as f:
            header = f[1].header
            data = f[1].data
            energies = f["EBOUNDS"].data
            energies_header = f["EBOUNDS"].header
            times = f["GTI"].data

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy
        # arrays as values
        self.data = {}
        data_columns = [
            i for i in data.columns if i.name not in self._exclude_data_cols
        ]
        for i in data_columns:
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
        # we have the good itme intervals for the DPH and also the time when the DPH were accumulated
        for i in times.columns:
            self.gti[f"TIME_{i.name}"] = u.Quantity(times[i.name], i.unit)
        self.gti["TIME_CENT"] = 0.5 * (self.gti[f"TIME_START"] + self.gti[f"TIME_STOP"])

        # now do the time bins for the dphs
        self.tbins["TIME_START"] = self.data["TIME"]
        self.tbins["TIME_STOP"] = self.data["TIME"] + self.data["EXPOSURE"]
        self.tbins["TIME_CENT"] = 0.5 * (
                self.tbins["TIME_START"] + self.tbins["TIME_STOP"]
        )

        # if self.dph_input_dict ==None, then we will need to try to read in the hisotry of parameters passed into
        # batbinevt to create the dph file. thsi usually is needed when we first parse a file so we know what things
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
        #   P18 countscol = DPH_COUNTS
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

        if self.dph_input_dict is None:
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

            self.dph_input_dict = default_params_dict.copy()

    @u.quantity_input(energybins=["energy"], emin=["energy"], emax=["energy"])
    def set_energybins(self, energybins=None, emin=None, emax=None):
        super().set_energybins(energybins=energybins, emin=emin, emax=emax)
        self.data["DPH_COUNTS"] = self.contents

    @u.quantity_input(timebins=["time"], tmin=["time"], tmax=["time"])
    def set_timebins(self, timebins=None, tmin=None, tmax=None):
        super().set_timebins(timebins=timebins, tmin=tmin, tmax=tmax)
        self.data["DPH_COUNTS"] = self.contents

    def to_fits(self, fits_filename=None, overwrite=False):
        """
        This method allows the user to save the rebinned DPH to a file. If no file is specified, then the dph_file
        attribute is used.
        """

        # get the defualt file name otherwise use what was passed in and expand to absolute path
        if fits_filename is None and self.dph_file is not None:
            fits_filename = self.dph_file
        else:
            fits_filename = Path(fits_filename).expanduser().resolve()

        if fits_filename.exists():
            if overwrite:
                with fits.open(test.dph_file, mode="update") as f:
                    # code to modify the table here with times and DPHs
                    # header = f[1].header
                    data = f[1].data
                    data_columns = [
                        i for i in data.columns if i.name not in test._exclude_data_cols
                    ]
                    temp_t_table = fits.FITS_rec.from_columns(
                        data_columns, nrows=test.tbins["TIME_START"].size
                    )

                    for i in data_columns:
                        if "DPH_COUNT" not in i.name:
                            temp_t_table[i.name] = test.data[i.name]
                        else:
                            for time_idx in range(test.tbins["TIME_START"].size):
                                temp_t_table[i.name][time_idx] = test.data[i.name][
                                    time_idx
                                ]

                    f[1].data = temp_t_table

                    # code to modify the energy bins
                    energies = f["EBOUNDS"].data
                    temp_energy = fits.FITS_rec.from_columns(
                        energies.columns, nrows=test.ebins["E_MIN"].size
                    )
                    # energies_header = f["EBOUNDS"].header
                    for i in energies.columns:
                        if "CHANNEL" in i.name:
                            temp_energy[i.name] = test.ebins["INDEX"]
                        elif "E" in i.name:
                            temp_energy[i.name] = test.ebins[i.name].value

                    f["EBOUNDS"].data = temp_energy

                    # code to modify the header info pertaining to the start/stop time
                    for i in f:
                        i.header["TSTART"] = test.tbins["TIME_START"].min().value
                        i.header["TSTOP"] = test.tbins["TIME_STOP"].max().value

                    f.flush()
            else:
                raise ValueError(
                    f"The file {fits_filename} will not be overwritten if the overwrite parameter is not explicitly set to True."
                )
        else:
            raise NotImplementedError("Saving to a new file is not yet implemented.")

        return None

    def reset(self):
        """
        This method allows the DPH object to be reset to the inputs in the passed in DPH file
        """

        self._parse_dph_file()

    def _call_batbinevt(self, input_dict):
        """
        Calls heasoftpy's batbinevt with an error wrapper, ensures that this bins the event data to produce a DPH

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batbinevt
        :return: heasoftpy Result object from batbinevt
        """

        input_dict["clobber"] = "YES"
        input_dict["outtype"] = "DPH"

        try:
            return hsp.batbinevt(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"The call to Heasoft batbinevt failed with inputs {input_dict}."
            )
