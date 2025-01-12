"""
This file holds the general DetectorPlaneHistogram class.

Tyler Parsotan March 8 2024
"""

from copy import deepcopy

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from histpy import Histogram
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .tte_data import TimeTaggedEvents

try:
    import heasoftpy.swift as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class DetectorPlaneHistogram(Histogram):
    """
    This class encapsulates data that is spread across the detector plane in an image like format. It allows for the
    rebinning of data in space, time, and/or energy.
    """

    # these are the edges of detectors including the spaces between detector modules
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
        initalized from a prebinned detector plane histogram as well. The class will save the timebins for the
        histograms and any timebins that may be missing from a continuous set of timebins will be filled in with
        histograms of all zeros for those points. These times will not show up in the gti attribute.

        When event data is passed in, the default bining is a single time bin from the smallest time of the event data
        to the largest time of the event data. The energy binning by default will be a single energy bin for the min/max
        event energy. (Thus, care will need to be taken when event data is passed in to initalize this object and it
        has not been filtered).

        By default, the spatial binning is in the detector x and y bins specified with the class attributes
        det_x_edges and det_y_edges.


        :param event_data: None or TimeTaggedEvents class that has been initialized with event data
        :param histogram_data: None or histpy Histogram or a numpy array of N dimensions. This should be formatted
            such that it has the following dimensions: (T,Ny,Nx,E) where T is the number of timebins, Ny is the
            number of detectors in the y direction see the det_x_edges class attribute, Nx represents an identical
            quantity in the x direction, and E is the number of energy bins
        :param timebins: None or a numpy array of continuous timebin edges, should be T+1 in size
        :param tmin: None or an astropy Quantity array of the beginning timebin edges, should be length T in size
        :param tmax: None or an astropy Quantity array of the end timebin edges, should be length T in size
        :param energybins: None or an astropy Quantity array of the continuous energy bin edges, should be size E+1
            in size
        :param emin: None or an astropy Quantity array of the beginning of the energy bins
        :param emax: None or an astropy Quantity array of the end of the energy bins
        :param weights: None or the weights of the same size as data contained in the event_data class or histogram_data
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
            # make sure that it is a TimeTaggedEvents class
            if not isinstance(event_data, TimeTaggedEvents):
                raise ValueError("The passed in event_data needs to be an instance of the TimeTaggedEvents class.")

            parse_data = deepcopy(event_data)
            self._passed_data = "event"
            self._event_data = parse_data
        else:
            parse_data = deepcopy(histogram_data)
            self._passed_data = "histogram"

        # determine the time binnings
        # can have dfault time binning be the start/end time of the event data or the times passed in by default
        # from a potential histpy Histogram object.
        # can also have timebins passed in with contiguous timebinning
        # or tmin/tmax passed in where the times are not continuous due to good time intervals etc
        if timebins is None and tmin is None and tmax is None:
            if event_data is not None:
                timebin_edges = u.Quantity(
                    [event_data.time.min(), event_data.time.max()]
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
            # make sure that tmin/tmax can be iterated over
            if tmin.isscalar:
                tmin = u.Quantity([tmin])
                tmax = u.Quantity([tmax])

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
                    [event_data.energy.min(), event_data.energy.max()]
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
            # make sure that emin/emax can be iterated over
            if emin.isscalar:
                emin = u.Quantity([emin])
                emax = u.Quantity([emax])

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
            self._set_histogram(histogram_data=parse_data, weights=weights)
        else:
            self._set_histogram(event_data=parse_data, weights=weights)

    def _set_histogram(self, histogram_data=None, event_data=None, weights=None):
        """
        This method properly initalizes the Histogram parent class. it uses the self.tbins and self.ebins information
        to define the time and energy binning for the histogram that is initalized.

        :param histogram_data: None or histpy Histogram or a numpy array of N dimensions. Thsi should be formatted
            such that it has the following dimensions: (T,Ny,Nx,E) where T is the number of timebins, Ny is the
            number of detectors in the y direction see the det_x_edges class attribute, Nx represents an identical
            quantity in the x direction, and E is the number of energy bins. These should be the appropriate sizes for
            the tbins and ebins attributes
        :param event_data: None or TimeTaggedEvents class
        :param weights: None or the weights of the same size as data contained in event_data or histogram_data
        :return: None
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
                event_data.time,
                event_data.dety.value,
                event_data.detx.value,
                event_data.energy,
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
        This method rebins the histogram in time. Note: this doesnt properly take the weighting into account and will
        need to be refined later on, ideally using the Histogram methods available. The tmin and tmax should be
        specified if the timebinnings that are requested are not continuous.

        :param timebins: None or an astropy Quantity array of continuous timebin edges that the histogram will be
            rebinned into
        :param tmin: None or an astropy Quantity array of the starting time bin edges that the histogram will be
            rebinned into
        :param tmax: None or an astropy Quantity array of the end time bin edges that the histogram will be
            rebinned into
        :return: None
        """

        # first make sure that we have a time binning axis of our histogram
        if "hist" in self._passed_data and ("TIME" not in self.axes.labels or self.axes["TIME"].nbins == 1):
            raise ValueError(
                "The histogrammed data either has no timing information or there is only one time bin which means that"
                "the data cannot be rebinned in time."
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

        # make sure that the times exist in the array of timebins that the DPH has been binned into if we are dealing
        # with histogrammed data
        if "hist" in self._passed_data:
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

        # get the intersection with the good time intervals for us to keep track of these and the exposures
        if "hist" in self._passed_data:
            idx = np.where(
                (self.tbins["TIME_START"] < self.gti["TIME_STOP"])
                & (self.tbins["TIME_STOP"] >= self.gti["TIME_STOP"])
            )

            for i in self.gti.keys():
                self.gti[i] = self.gti[i][idx]
            self.exposure = self.exposure[idx]
        else:
            # for event data just make sure that the time intervals are within the event data times
            idx = np.where(
                (self.tbins["TIME_START"] > self._event_data.time.min())
                & (self.tbins["TIME_STOP"] <= self._event_data.time.max())
            )
            self.gti["TIME_START"] = self.tbins["TIME_START"][idx]
            self.gti["TIME_STOP"] = self.tbins["TIME_STOP"][idx]
            self.gti["TIME_CENT"] = 0.5 * (self.gti["TIME_START"] + self.gti["TIME_STOP"])

            self.exposure = self.gti["TIME_STOP"] - self.gti["TIME_START"]

        # now we can reinitalize the info
        if "hist" in self._passed_data:
            self._set_histogram(histogram_data=histograms)
        else:
            self._set_histogram(event_data=self._event_data)

        return None

    @u.quantity_input(energybins=["energy"], emin=["energy"], emax=["energy"])
    def set_energybins(self, energybins=None, emin=None, emax=None):
        """
        This method allows for the histogram to be rebinned to different energy binnings.
        If the specified energy ranges are not continuous, it is better to specify emin and emax.

        :param energybins: None or an astropy Quantity array of the continuous energy bin edges, should be size E+1
            in size
        :param emin: None or an astropy Quantity array of the beginning of the energy bins
        :param emax: None or an astropy Quantity array of the end of the energy bins
        :return: None
        """

        # first make sure that we have a energy binning axis of our histogram
        if "hist" in self._passed_data and ("ENERGY" not in self.axes.labels or self.axes["ENERGY"].nbins == 1):
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

        # make sure that the energies exist in the array of energybins that the DPH has been binned into for histogrammed
        # data that was passed in. otherwise for event data we can bin any which way
        if "hist" in self._passed_data:
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
        if "hist" in self._passed_data:
            self._set_histogram(histogram_data=histograms)
        else:
            self._set_histogram(event_data=self._event_data)

        return None

    @u.quantity_input(emin=["energy"], emax=["energy"], tmin=["time"], tmax=["time"])
    def plot(self, emin=None, emax=None, tmin=None, tmax=None, plot_rate=False):
        """
        This method allows the user to conveniently plot the histogram for a single energy bin and time interval.
        Any detectors with 0 counts (due to detectors being off or due to there being no detectors in the specified
        DETX and DETY coordinates) are blacked out.

        By default, the histogram is binned along the energy and time axes. This behavior can be changed by specifying
        emin/emax and/or tmin/tmax. These values should all exist in the ebins and tbins attributes.

        :param emin: None or an astropy Quantity array of the beginning of the energy bins
        :param emax: None or an astropy Quantity array of the end of the energy bins
        :param tmin: None or an astropy Quantity array of the starting time bin edges that the histogram will be
            rebinned into
        :param tmax: None or an astropy Quantity array of the end time bin edges that the histogram will be
            rebinned into
        :param plot_rate: Boolean to denote if the count rate should be plotted. The histogram gets divided by the
            exposure time of the plotted histogram
        :return: matplotlib figure and axis for the plotted histogram
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

        ax.set_ylabel("DETY")
        ax.set_xlabel("DETX")

        return fig, ax
