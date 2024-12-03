"""
This file holds the BatDRM class which hold the detector response matrix at a given time.

Tyler Parsotan Sept 3 2024

"""
import os
from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
import numpy as np
from astropy.io import fits
from histpy import Histogram

try:
    import heasoftpy.swift as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)

from gdt.missions.swift.bat.headers import RspHeaders


class BatDRM(Histogram):

    @u.quantity_input(
        timebins=["time"],
        tmin=["time"],
        tmax=["time"],
        input_energybins=["energy"],
        input_emin=["energy"],
        input_emax=["energy"],
        output_energybins=["energy"],
        output_emin=["energy"],
        output_emax=["energy"],
    )
    def __init__(self,
                 drm_data,
                 tmin=None,
                 tmax=None,
                 timebins=None,
                 input_emin=None,
                 input_emax=None,
                 input_energybins=None,
                 output_emin=None,
                 output_emax=None,
                 output_energybins=None,
                 ):
        """
        Create a BatDRM object based on an input DRM numpy array or a pre-constructed Histogram object.

        :param drm_data: a numpy array with 3 axis correspondent to the timebins, input/output energybins below
            OR
            a Histogram object that has the detector response matrix with the appropriate "TIME", "E_IN", "E_OUT" axes
        :param tmin: astropy.units.Quantity denoting the minimum values of the timebin edges that the DRM corresponds
            to. Units will usually be in seconds for this.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param tmax: astropy.units.Quantity denoting the maximum values of the timebin edges that the DRM corresponds
            to. Units will usually be in seconds for this.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. If tmin/tmax is specified, this parameter is ignored.
        :param input_emin: an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            DRM "input" photon energy bins. NOTE: If input_emin/emax are specified, the input_energybins parameter is
            ignored.
        :param input_emax: an astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            DRM "input" photon energy bins. NOTE: If input_emin/emax are specified, the input_energybins parameter is
            ignored.
        :param input_energybins: an astropy.unit.Quantity object of 2 or more elements with the "input" photon energy
            bin edges that the DRM has been binned into. None of the energy ranges should overlap.
            If input_emin/emax is specified this parameter is ignored.
        :param output_emin: an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            DRM "output", or dispersed, photon energy bins.
            NOTE: If output_emin/emax are specified, the output_energybins parameter is ignored.
        :param output_emax: an astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            DRM "output", or dispersed, photon energy bins.
            NOTE: If output_emin/emax are specified, the output_energybins parameter is ignored.
        :param output_energybins: an astropy.unit.Quantity object of 2 or more elements with the "output", dispersed,
            photon energy bin edges that the DRM has been binned into. None of the energy ranges should overlap.
            If output_emin/emax is specified this parameter is ignored.

        """

        # do some error checking
        if not isinstance(drm_data, np.ndarray) and not isinstance(drm_data, Histogram):
            raise ValueError("The input DRM data has to be a numpy array or a Histogram object.")

        if not isinstance(drm_data, Histogram):
            if timebins is None and tmin is None and tmax is None:
                raise ValueError(
                    "The timebin for the DRM needs to specified using the timebins or the tmin/tmax parameters.")

            if input_energybins is None and input_emin is None and input_emax is None:
                raise ValueError(
                    "The input photon energybins for the DRM needs to specified using the input_energybins or the input_emin/input_emax parameters.")

            if output_energybins is None and output_emin is None and output_emax is None:
                raise ValueError(
                    "The output/measured photon energybins for the DRM needs to specified using the output_energybins or the output_emin/output_emax parameters.")
        else:
            # get the timebin/energy edges from the Histogram
            timebins = drm_data.axes["TIME"].edges
            input_energybins = drm_data.axes["E_IN"].edges
            output_energybins = drm_data.axes["E_OUT"].edges

        # see if we have thetmin/tmax otherwise use timebins  in that order of preference
        self.tbins = {}
        if tmin is not None or tmax is not None:
            # make sure that both tmin and tmax are defined and have the same number of elements
            if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
                raise ValueError('Both tmin and tmax must be defined.')

            self.tbins["TIME_START"] = tmin
            self.tbins["TIME_STOP"] = tmax

        else:
            self.tbins["TIME_START"] = timebins[:-1]
            self.tbins["TIME_STOP"] = timebins[1:]

        self.tbins["TIME_CENT"] = 0.5 * (
                self.tbins["TIME_START"] + self.tbins["TIME_STOP"]
        )

        self.input_ebins = {}
        if input_emin is not None or input_emax is not None:
            # make sure that both input_emin and input_emax are defined and have the same number of elements
            if (input_emin is None and input_emax is not None) or (input_emax is None and input_emin is not None):
                raise ValueError('Both input_emin and input_emax must be defined.')

            self.input_ebins["INDEX"] = np.arange(input_emin.size) + 1
            self.input_ebins["E_MIN"] = input_emin
            self.input_ebins["E_MAX"] = input_emax
        else:
            self.input_ebins["INDEX"] = np.arange(input_energybins.size - 1) + 1
            self.input_ebins["E_MIN"] = input_energybins[:-1]
            self.input_ebins["E_MAX"] = input_energybins[1:]

        self.output_ebins = {}
        if output_emin is not None or output_emax is not None:
            # make sure that both output_emin and output_emax are defined and have the same number of elements
            if (output_emin is None and output_emax is not None) or (output_emax is None and output_emin is not None):
                raise ValueError('Both output_emin and output_emax must be defined.')

            self.output_ebins["INDEX"] = np.arange(output_emin.size) + 1
            self.output_ebins["E_MIN"] = output_emin
            self.output_ebins["E_MAX"] = output_emax
        else:
            self.output_ebins["INDEX"] = np.arange(output_energybins.size - 1) + 1
            self.output_ebins["E_MIN"] = output_energybins[:-1]
            self.output_ebins["E_MAX"] = output_energybins[1:]

        self._set_histogram(histogram_data=drm_data)

    def _set_histogram(self, histogram_data):
        """
        This method properly initalizes the Histogram parent class. it uses the self.tbins and self.input_ebins,
        self.output_ebins information
        to define the time and input/output energy binning for the histogram that is initalized.

        :param histogram_data: None or histpy Histogram or a numpy array of N dimensions. This should be formatted
            such that it has the following dimensions: (T,NE_IN,NE_OUT) where T is the number of timebins, NE_IN is the
            number of binned input photon energies, NE_OUT represents the number of output/dispersed energy bins.
            These should be the appropriate sizes for the tbins, input_ebins and output_ebins attributes.
        :return: None
        """

        # get the timebin edges
        timebin_edges = (
                np.zeros(self.tbins["TIME_START"].size + 1) * self.tbins["TIME_START"].unit
        )
        timebin_edges[:-1] = self.tbins["TIME_START"]
        timebin_edges[-1] = self.tbins["TIME_STOP"][-1]

        # get the energybin edges
        input_energybin_edges = (
                np.zeros(self.input_ebins["E_MIN"].size + 1) * self.input_ebins["E_MIN"].unit
        )
        input_energybin_edges[:-1] = self.input_ebins["E_MIN"]
        input_energybin_edges[-1] = self.input_ebins["E_MAX"][-1]

        output_energybin_edges = (
                np.zeros(self.output_ebins["E_MIN"].size + 1) * self.output_ebins["E_MIN"].unit
        )
        output_energybin_edges[:-1] = self.output_ebins["E_MIN"]
        output_energybin_edges[-1] = self.output_ebins["E_MAX"][-1]

        # create our histogrammed data
        if isinstance(histogram_data, u.Quantity):
            hist_unit = histogram_data.unit
        else:
            hist_unit = u.cm * u.cm

        if not isinstance(histogram_data, Histogram):
            # if the historgram has the appropriate number of dimensions we are good otherwise we most  likely
            # ened to add a new axis for time
            if np.ndim(histogram_data) == 3:
                super().__init__(
                    [
                        timebin_edges,
                        input_energybin_edges,
                        output_energybin_edges,
                    ],
                    contents=histogram_data,
                    labels=["TIME", "E_IN", "E_OUT"],
                    sumw2=None,
                    unit=hist_unit,
                )
            else:
                super().__init__(
                    [
                        timebin_edges,
                        input_energybin_edges,
                        output_energybin_edges,
                    ],
                    contents=histogram_data[np.newaxis],
                    labels=["TIME", "E_IN", "E_OUT"],
                    sumw2=None,
                    unit=hist_unit,
                )

        else:
            super().__init__(
                [i.edges for i in histogram_data.axes],
                contents=histogram_data.contents,
                labels=histogram_data.axes.labels,
                unit=hist_unit,
            )

    @staticmethod
    def calc_drm(pha_file):
        """
        This calls heasoftpy's batdrmgen which produces the associated drm for fitting the PHA file.

        :param pha_file: a list of PHA path objects or a Path object to the PHA file that the DRM will be constructed for
        :return: Path object to the created DRM file or a list of Path objects to all the DRM files created
        """

        if type(pha_file) is not list:
            pha_file = [pha_file]

        # when passing in tht whole filename, the paths mess up the connection between the response file and the pha file
        # since there seems to be some character limit to this header value. Therefore, we need to cd to the directory
        # that the PHA file lives in and create the .rsp file and then cd back to the original location.

        # make sure that all elements are paths
        phafilename = [Path(i) for i in pha_file]

        # we are passing in a whole filepath or
        # we are already located in the PHA directory and are mabe calculating the upperlimit bkg spectrum

        # Check if the phafilename is a string and if it has an extension .pha. If NOT then exit
        # drm_file=[]
        for filename in phafilename:
            if ".pha" not in filename.name:
                raise ValueError(
                    f"The file name {filename} needs to be a string and must have an extension of .pha ."
                )

            # get the cwd
            current_dir = Path.cwd()

            # get the directory that we have to cd to and the name of the file
            pha_dir = filename.parent
            pha_file = filename.name

            # cd to that dir
            if str(pha_dir) != str(current_dir):
                os.chdir(pha_dir)

            # Split the filename by extension, so as to remove the .pha and replace it with .rsp
            # this is necessary since sources can have '.' in name
            out = filename.stem + ".rsp"

            # create drm
            output = hsp.batdrmgen(
                infile=pha_file, outfile=out, chatter=2, clobber="YES", hkfile="NONE"
            )

            if output.returncode != 0:
                raise RuntimeError(f"The call to Heasoft batdrmgen failed with output {output.stdout}.")

            # cd back
            if str(pha_dir) != str(current_dir):
                os.chdir(current_dir)

        drm_file = [i.parent.joinpath(f"{i.stem}.rsp") for i in phafilename]

        if len(drm_file) > 1:
            return drm_file
        else:
            return drm_file[0]

    def plot(self):
        """
        This method plots the DRM object.

        :return: matplotlib axis and mesh handles
        """

        plot_data = self.project("E_IN", "E_OUT").contents
        vmax = plot_data.max().value
        vmin = plot_data[plot_data > 0].min().value

        t = colors.LogNorm(vmin=vmin, vmax=vmax)

        ax, mesh = self.project("E_IN", "E_OUT").plot(norm=t)
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_ylim([1, self.axes["E_OUT"].edges.max().value])
        ax.set_xlim([self.axes["E_IN"].edges.min().value, self.axes["E_IN"].edges.max().value])

        # set the 0 DRM values to black
        cm = mesh.get_cmap()
        cm.set_bad((0, 0, 0))

        return ax, mesh

    @classmethod
    def from_file(cls, pha_file=None, drm_file=None):
        """
        This class method takes either a pha file of a drm file and either:
            1) create a drm file based on an input PHA file and load that into a BatDRM object
            2) create a BatDRM object from a preconstructed DRM file

        :param pha_file: a string or a Path object to a pha file that will be used to calculate an associated DRM file.
        :param drm_file: a string or a Path object to a DRM file that has already been created, which will then be
            loaded into a BatDRM object
        :return: BatDRM object
        """

        # make sure something is specified
        if pha_file is None and drm_file is None:
            raise ValueError("Either a Path object specifying a PHA file or a Path object specifying a DRM file needs"
                             " to be passed in.")

        # maks sure only 1 file is specified and do error checking
        if pha_file is not None and drm_file is not None:
            raise ValueError("Please only specify either a pha file or a drm file.")

        if pha_file is not None:

            pha_file = Path(pha_file).expanduser().resolve()
            if not pha_file.exists():
                raise ValueError(f"The specified file {pha_file} does not seem to exist. "
                                 f"Please double check that it does.")

        if drm_file is not None:

            drm_file = Path(drm_file).expanduser().resolve()
            if not drm_file.exists():
                raise ValueError(f"The specified file {drm_file} does not seem to exist. "
                                 f"Please double check that it does.")

        # first see if we have a pha file to create a drm for
        if pha_file is not None:
            drm_file = cls.calc_drm(pha_file)

        # if we had to first create a pha file, we still set drm_file so we enter this if statement and parse the file
        # otherwise the user passes in drm_file and we still execute this.
        if drm_file is not None:
            with fits.open(drm_file) as f:
                # get the size of things that we need to save
                n_out_ebins = f["EBOUNDS"].header["NAXIS2"]
                n_in_ebins = f[1].header["NAXIS2"]

                energy_unit = u.Quantity(f'1{f["EBOUNDS"].header["TUNIT2"]}')
                time_unit = u.Quantity(f'1{f[1].header["TIMEUNIT"]}')

                timebin = [f[1].header["TSTART"], f[1].header["TSTOP"]] * time_unit

                # get the output energy bin edges, therefore need the +1
                out_ebins = np.zeros(n_out_ebins + 1)
                in_ebins = np.zeros(n_in_ebins + 1)

                # create the arrays
                out_ebins[:-1] = f["EBOUNDS"].data["E_MIN"]
                out_ebins[-1] = f["EBOUNDS"].data["E_MAX"][-1]

                in_ebins[:-1] = f[1].data["ENERG_LO"]
                in_ebins[-1] = f[1].data["ENERG_HI"][-1]

                # get the full response matrix
                rsp = np.zeros((n_in_ebins, n_out_ebins))

                for count, mat in enumerate(f[1].data["MATRIX"]):
                    rsp[count, :] = mat

        return cls(drm_data=rsp, input_energybins=in_ebins * energy_unit, output_energybins=out_ebins * energy_unit,
                   timebins=timebin)

    @classmethod
    def concatenate(cls, drm_list, weights=None, drm_save_file=None):
        """
        This class method takes a list of BatDRM objects and combines them with weighting factors if they are provided.

        :param drm_list: list of BatDRM objects that will be combined
        :param weights: None, or a list of weightings for the drm_list objects. if the weights don't sum to 1, then the
            input weights will be divided by the sum of the weights. None defaults to weights
            being 1/N, where N is the length of the drm_list that is passed in
        :param drm_save_file: a Path object containing the location and name of where the constructed DRM will be saved
        :return: a BatDRM object
        """

        # want to verify that all inputs are BatDRMs and that weights are normalized
        if np.any([not isinstance(i, cls) for i in drm_list]):
            raise ValueError(
                "All elements of the list that is passed in need to be BatDRM objects.")

        # if weights are passed in, then they need to be normalized otherwise we just set the weights to 1 and add things
        if weights is not None:
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(drm_list)) / len(drm_list)

        if len(weights) != len(drm_list):
            raise ValueError(
                "The number of drm_list elements do not match the number of weights that have been passed in")

        input_drm_list = [i.project("E_IN", "E_OUT") * j for i, j in zip(drm_list, weights)]
        times = [i.axes["TIME"].edges for i in drm_list]
        time_unit = drm_list[0].axes["TIME"].edges.unit

        output_drm = Histogram.concatenate(np.unique(times) * time_unit, input_drm_list, label="TIME")

        output_drm = cls(output_drm.rebin([np.unique(times).size - 1, 1, 1]))

        # to save the output drm to a file, we need to also make sure that at least 1 drm in the drm list has the drm.
        if drm_save_file is not None:
            output_drm._save(drm_save_file)

        return output_drm

    def _save(self, drm_file):
        """
        This method saves a DRM object to a detector response file which can be used for spectral fitting.

        :param drm_file: string or a Path object denoting the location and name of the detector response file that will
            be saved.
        :return: None
        """

        drm_file = Path(drm_file).expanduser().resolve()

        # get the default headers
        rsp = RspHeaders()

        hdulist = fits.HDUList()

        primary_hdu = fits.PrimaryHDU(header=rsp[0])

        hdulist.append(primary_hdu)

        # modify them based on what we have in this object
        ehi_col = fits.Column(name='ENERG_HI', format='E',
                              array=self.input_ebins["E_MAX"], unit=self.input_ebins["E_MAX"].unit.name
                              )
        elo_col = fits.Column(name='ENERG_LO', format='E',
                              array=self.input_ebins["E_MIN"], unit=self.input_ebins["E_MIN"].unit.name
                              )
        nchan_col = fits.Column(name='N_CHAN', format='I',
                                array=np.ones_like(self.input_ebins["E_MAX"].value) * self.output_ebins["INDEX"][-1])
        ngrp_col = fits.Column(name='N_GRP', format='I', array=np.ones_like(self.input_ebins["E_MAX"].value))
        fchan_col = fits.Column(name='F_CHAN', format='I', array=np.zeros_like(self.input_ebins["E_MAX"].value))

        matrix_col = fits.Column(name='MATRIX', array=self.contents[0, :, :].value,
                                 format='{}E'.format(self.output_ebins["INDEX"][-1]))

        hdu = fits.BinTableHDU.from_columns([elo_col, ehi_col, ngrp_col,
                                             fchan_col, nchan_col, matrix_col],
                                            header=rsp[1])
        hdulist.append(hdu)

        # get the out energy card
        chan_col = fits.Column(name='CHANNEL', format='I', array=self.output_ebins["INDEX"] - 1)
        emin_col = fits.Column(name='E_MIN', format='E', array=self.output_ebins["E_MIN"],
                               unit=self.output_ebins["E_MIN"].unit.name)
        emax_col = fits.Column(name='E_MAX', format='E', array=self.output_ebins["E_MAX"],
                               unit=self.output_ebins["E_MAX"].unit.name)
        hdu = fits.BinTableHDU.from_columns([chan_col, emin_col, emax_col], header=rsp[2])
        hdulist.append(hdu)

        # add time header keywords
        hdulist[1].header["TSTART"] = self.tbins["TIME_START"].min().value
        hdulist[1].header["TSTOP"] = self.tbins["TIME_STOP"].max().value
        hdulist[1].header["TIMEUNIT"] = self.tbins["TIME_STOP"].unit.name

        hdulist.writeto(drm_file)
