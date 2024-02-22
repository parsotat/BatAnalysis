"""
This file holds the BAT Detector plane histogram class

Tyler Parsotan Feb 21 2024
"""

import shutil
import numpy as np

from .batobservation import BatObservation
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gzip
from astropy.io import fits
from pathlib import Path
from astropy.time import Time
import astropy.units as u
import warnings

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatDPH(BatObservation):
    """
    This class encapsulates the BAT detector plane historgrams (DPH) that are collected onboard. It also allows for the
    creation of a DPH from event data and rebinning in time/energy.
    """

    # this class variable states which columns form the DPH files should be excluded from being read in
    _exclude_data_cols = ["GAIN_INDEX", "OFFSET_INDEX", "LDPNAME", "BLOCK_MAP", "NUM_DETS", "APID", "LDP"]

    def __init__(self, dph_file, event_file, input_dict=None, recalc=False, verbose=False, load_dir=None):
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
                raise ValueError(f"The specified event file {self.event_file} does not seem to exist. "
                                 f"Please double check that it does.")
        else:
            self.event_file = None
            warnings.warn("No event file has been specified. The resulting DPH object will not be able "
                          "to be arbitrarily modified either by rebinning in energy or time.", stacklevel=2)

        # if ther is no event file we just have the instrument produced DPH or a previously calculated one
        # if self.event_file is None:
        #    self.dph_input_dict = None

        if (not self.dph_file.exists() or recalc) and self.event_file is None:
            # we need to create the file
            stop

        else:
            self.dph_input_dict = None

        self._parse_dph_file()

        return None

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
            raise ValueError(f"The specified DPH file {dph_file} does not seem to exist. "
                             f"Please double check that it does.")

        # also make sure that the file is gunzipped which is necessary the user wants to do spectral fitting
        if ".gz" in dph_file.suffix:
            with gzip.open(dph_file, 'rb') as f_in:
                with open(dph_file.parent.joinpath(dph_file.stem), 'wb') as f_out:
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
        data_columns = [i for i in data.columns if i.name not in self._exclude_data_cols]
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
        self.tbins["TIME_CENT"] = 0.5 * (self.gti[f"TIME_START"] + self.gti[f"TIME_STOP"])

        # now do the time bins for the dphs
        self.tbins["TIME_START"] = self.data["TIME"]
        self.tbins["TIME_STOP"] = self.data["TIME"] + self.data["EXPOSURE"]
        self.tbins["TIME_CENT"] = 0.5 * (self.tbins["TIME_START"] + self.tbins["TIME_STOP"])

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
            test = hsp.HSPTask('batbinevt')
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
                        default_params_dict[old_parameter] = default_params_dict[old_parameter] + values[-1]
                        # assume that we need to keep appending to the previous parameter
                    else:
                        default_params_dict[parameter] = values[-1]

                        old_parameter = parameter

            self.dph_input_dict = default_params_dict.copy()

    @u.quantity_input(emin=['energy'], emax=['energy'], tmin=['time'], tmax=['time'])
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
            raise ValueError("emin and emax must either both be None or both be specified.")
        plot_e_idx = np.where((self.ebins["E_MIN"] >= plot_emin) & (self.ebins["E_MAX"] <= plot_emax))[0]

        if tmin is None and tmax is None:
            plot_tmin = self.tbins["TIME_START"].min()
            plot_tmax = self.tbins["TIME_STOP"].max()
        elif tmin is not None and tmax is not None:
            plot_tmin = tmin
            plot_tmax = tmax
        else:
            raise ValueError("tmin and tmax must either both be None or both be specified.")
        plot_t_idx = np.where((self.tbins["TIME_START"] >= plot_tmin) & (self.tbins["TIME_STOP"] <= plot_tmax))[0]

        # now start to accumulate the DPH counts based on the time and energy range that we care about
        plot_data = self.data["DPH_COUNTS"][plot_t_idx, :, :, :]

        if len(plot_t_idx) > 0:
            plot_data = plot_data.sum(axis=0)

        plot_data = plot_data[:, :, plot_e_idx]

        if len(plot_e_idx) > 0:
            plot_data = plot_data.sum(axis=-1)

        if plot_rate:
            # calcualte the totoal exposure
            exposure_tot = np.sum(self.data["EXPOSURE"][plot_t_idx])
            plot_data /= exposure_tot

        # set any 0 count detectors to nan so they get plotted in black
        # this includes detectors that are off and "space holders" between detectors where the value is 0
        plot_data[plot_data == 0] = np.nan

        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cmap = mpl.colormaps.get_cmap('viridis')
        cmap.set_bad(color='k')

        im = ax.imshow(plot_data.value, origin="lower", interpolation='none', cmap=cmap)

        fig.colorbar(im, cax=cax, orientation='vertical', label=plot_data.unit)

        return fig, ax
