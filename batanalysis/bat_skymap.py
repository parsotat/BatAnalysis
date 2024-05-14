"""
This file holds the BatSkyImage class which contains binned data from a skymap generated

Tyler Parsotan March 11 2024
"""
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from healpy.newvisufunc import projview
from histpy import Histogram, HealpixAxis
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reproject import reproject_to_healpix

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatSkyImage(Histogram):
    """
    This class holds the information related to a sky image, which is created from a detector plane image

    This is constructed by doing a FFT of the sky image with the detector mask. There can be one or many
    energy or time bins. We can also handle the projection of the image onto a healpix map.

    This hold data correspondent to BAT's view of the sky. This can be a flux map (sky image) created from a FFT
    decolvolution, a partial coding map, a significance map, or a background variance map. These can all be energy
    dependent except for the partial coding map.
    """

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
            image_data=None,
            timebins=None,
            tmin=None,
            tmax=None,
            energybins=None,
            emin=None,
            emax=None,
            weights=None,
            wcs=None
    ):
        """
        This class is meant to hold images of the sky that have been created from a deconvolution of the BAT DPI with
        the coded mask. The sky data can represent flux as a function of energy, background variance as a function of
        energy, the significance map as a function of energy, and the partial coding map.

        This class holds an image for a single time bin for simplicity.

        :param image_data:
        :param timebins:
        :param tmin:
        :param tmax:
        :param energybins:
        :param emin:
        :param emax:
        :param weights:
        """

        # do some error checking
        if image_data is None:
            raise ValueError(
                "A  needs to be passed in to initalize a BatSkyImage object"
            )

        if wcs is None:
            warnings.warn(
                "No astropy World Coordinate System has been specified the sky image is assumed to be in the detector "
                "tangent plane. No conversion to Healpix will be possible",
                stacklevel=2,
            )

        if not isinstance(wcs, WCS):
            raise ValueError("The wcs is not an astropy WCS object.")

        parse_data = deepcopy(image_data)

        if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
            raise ValueError("Both tmin and tmax must be defined.")

        if tmin is not None and tmax is not None:
            if tmin.size != tmax.size:
                raise ValueError("Both tmin and tmax must have the same length.")

        # determine the time binnings
        # can have dfault time binning be the start/end time of the event data or the times passed in by default
        # from a potential histpy Histogram object.
        # also want to make sure that the image is at a single timebin
        if timebins is None and tmin is None and tmax is None:
            if not isinstance(image_data, Histogram):
                # if we dont have a histpy histogram, need to have the timebins
                raise ValueError(
                    "For a general histogram that has been passed in, the timebins need to be specified"
                )
            else:
                timebin_edges = image_data.axes["TIME"].edges
        elif timebins is not None:
            timebin_edges = timebins
        else:
            # use the tmin/tmax
            timebin_edges = u.Quantity([tmin, tmax])

        # make sure that timebin_edges is only 2 elements (meaning 1 time bin)
        if len(timebin_edges) != 2:
            raise ValueError(
                "The BatSkyImage object should be initalized with only 1 timebin. This was initialized with"
                f"{len(timebin_edges) - 1} timebins.")

        # determine the energy binnings
        if energybins is None and emin is None and emax is None:
            if not isinstance(image_data, Histogram):
                # if we dont have a histpy histogram, need to have the energybins
                raise ValueError(
                    "For a general histogram that has been passed in, the energybins need to be specified"
                )
            else:
                energybin_edges = image_data.axes["ENERGY"].edges
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

                if image_data is not None:
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

        self._set_histogram(histogram_data=parse_data, weights=weights)
        self.wcs = wcs

    def _set_histogram(self, histogram_data=None, event_data=None, weights=None):
        """
        This method properly initalizes the Histogram parent class. it uses the self.tbins and self.ebins information
        to define the time and energy binning for the histogram that is initalized.

        COPIED from DetectorPlaneHist class, can be organized better.

        :param histogram_data: None or histpy Histogram or a numpy array of N dimensions. Thsi should be formatted
            such that it has the following dimensions: (T,Ny,Nx,E) where T is the number of timebins, Ny is the
            number of image pixels in the y direction, Nx represents an identical
            quantity in the x direction, and E is the number of energy bins. These should be the appropriate sizes for
            the tbins and ebins attributes
        :param event_data: None or Event data dictionary or event data class (to be created)
        :param weights: None or the weights of the same size as event_data or histogram_data
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
        if isinstance(histogram_data, u.Quantity):
            hist_unit = histogram_data.unit
        else:
            hist_unit = u.count

        if not isinstance(histogram_data, Histogram):
            # need to make sure that the histogram_data has the correct shape ie be 4 dimensional arranged as (T,Ny,Nx,E)
            if np.ndim(histogram_data) != 4:
                raise ValueError(f'The size of the input sky image is a {np.ndim(histogram_data)} dimensional array'
                                 f'which needs to be a 4D array arranged as (T,Ny,Nx,E) where T is the number of '
                                 f'timebins, Ny is the number of image y pixels, Nx is the number of image x pixels,'
                                 f' and E is the number of energy bins.')

            # see if the shape of the image data is what it should be
            if np.shape(histogram_data) != (
                    self.tbins["TIME_START"].size,
                    histogram_data.shape[1], histogram_data.shape[2],
                    self.ebins["E_MIN"].size,
            ):
                raise ValueError(f'The shape of the input sky image is {np.shape(histogram_data)} while it should be'
                                 f'{(self.tbins["TIME_START"].size, histogram_data.shape[1], histogram_data.shape[2], self.ebins["E_MIN"].size)}')
            super().__init__(
                [
                    timebin_edges,
                    np.arange(histogram_data.shape[1] + 1) - 0.5,
                    np.arange(histogram_data.shape[2] + 1) - 0.5,
                    energybin_edges,
                ],
                contents=histogram_data,
                labels=["TIME", "IMY", "IMX", "ENERGY"],
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

    def healpix_projection(self, coordsys="galactic", nside=128):
        """
        This creates a healpix projection of the image. The dimension of the array is

        :param img_header:
        :param coordsys:
        :param nside:
        :return:
        """

        # create our new healpix axis
        hp_ax = HealpixAxis(nside=nside, coordsys=coordsys, label="HPX")

        # create a new array to hold the projection of the sky image in detector tangent plane coordinates to healpix
        # coordinates
        new_array = np.zeros((self.axes['TIME'].nbins, hp_ax.nbins, self.axes["ENERGY"].nbins))

        # for each time/energybin do the projection (ie linear interpolation)
        for t in range(self.axes['TIME'].nbins):
            for e in range(self.axes["ENERGY"].nbins):
                array, footprint = reproject_to_healpix((self.project("IMY", "IMX").contents, self.wcs), coordsys,
                                                        nside=nside)
                new_array[t, :, e] = array

        # create the new histogram
        h = Histogram(
            [self.axes['TIME'], hp_ax, self.axes["ENERGY"]],
            contents=new_array)

        # can return the histogram or choose to modify the class histogram. If the latter, need to get way to convert back
        # to detector plane coordinates
        return new_array, footprint, h

    def calc_radec(self):
        """

        :param img_header:
        :return:
        """
        from .mosaic import convert_xy2radec

        x = np.arange(self.axes["IMX"].nbins)
        y = np.arange(self.axes["IMY"].nbins)
        xx, yy = np.meshgrid(x, y)

        ra, dec = convert_xy2radec(xx, yy, self.wcs)

        c = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")

        return c.ra, c.dec

    def calc_glatlon(self):
        """

        :param img_header:
        :return:
        """

        ra, dec = self.calc_radec()

        c = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")

        return c.galactic.l, c.galactic.b

    @u.quantity_input(emin=["energy"], emax=["energy"], tmin=["time"], tmax=["time"])
    def plot(self, emin=None, emax=None, tmin=None, tmax=None, projection=None, coordsys="galactic", nside=128):
        """
        This is a convenience plotting function that allows for quick and easy plotting of a sky image. It allows for
        energy and time (where applicable) slices and different representations of the sky image.

        :param emin:
        :param emax:
        :param tmin:
        :param tmax:
        :param projection:
        :param coordsys:
        :param nside:
        :return:
        """

        # do some error checking
        if emin is None and emax is None:
            emin = self.axes["ENERGY"].lo_lim
            emax = self.axes["ENERGY"].hi_lim
        elif emin is not None and emax is not None:
            if emin not in self.axes["ENERGY"].edges or emax not in self.axes["ENERGY"].edges:
                raise ValueError(
                    f'The passed in emin or emax value is not a valid ENERGY bin edge: {self.axes["ENERGY"].edges}')
        else:
            raise ValueError("emin and emax must either both be None or both be specified.")

        if tmin is None and tmax is None:
            tmin = self.axes["TIME"].lo_lim
            tmax = self.axes["TIME"].hi_lim
        elif tmin is not None and tmax is not None:
            if tmin not in self.axes["TIME"].edges or tmax not in self.axes["TIME"].edges:
                raise ValueError(
                    f'The passed in tmin or tmax value is not a valid TIME bin edge: {self.axes["TIME"].edges}')
        else:
            raise ValueError("tmin and tmax must either both be None or both be specified.")

        # get the bin index to then do slicing and projecting for plotting. Need to make sure that the inputs are
        # single numbers (not single element array) which is why we do __.item()
        tmin_idx = self.axes["TIME"].find_bin(tmin.item())
        tmax_idx = self.axes["TIME"].find_bin(tmax.item())

        emin_idx = self.axes["ENERGY"].find_bin(emin.item())
        emax_idx = self.axes["ENERGY"].find_bin(emax.item())

        # now do the plotting
        if projection is None:
            # use the default spatial axes of the histogram
            # need to determine what this is
            if "IMX" in self.axes.labels:
                ax, mesh = self.slice[tmin_idx:tmax_idx + 1, :, :, emin_idx:emax_idx + 1].project("IMX", "IMY").plot()
                ret = (ax, mesh)
            elif "HPX" in self.axes.labels:
                if "galactic" in coordsys.lower():
                    coord = ["G"]
                elif "icrs" in coordsys.lower():
                    coord = ["G", "C"]
                else:
                    raise ValueError('This plotting function can only plot the healpix map in galactic or icrs '
                                     'coordinates.')
                mesh = projview(self.slice[tmin_idx:tmax_idx + 1, :, emin_idx:emax_idx + 1].project("HPX").contents,
                                coord=coord, graticule=True, graticule_labels=True,
                                projection_type="mollweide", reuse_axes=False)
                ret = (mesh)
            else:
                raise ValueError("The spatial projection of the sky image is currently not accepted as a plotting "
                                 "option. Please convert to IMX/IMY or a HEALPix map.")
        else:
            # the user has specified different options, can be ra/dec or healpix (with coordsys of galactic or icrs)
            # if we want Ra/Dec
            if "ra/dec" in projection.lower():
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1, projection=self.wcs)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=maxes.Axes)
                cmap = mpl.colormaps.get_cmap("viridis")
                cmap.set_bad(color="w")

                ax.grid(color='k', ls='solid')
                im = ax.imshow(
                    self.slice[tmin_idx:tmax_idx + 1, :, :, emin_idx:emax_idx + 1].project("IMY", "IMX").contents.value,
                    origin="lower")
                cbar = fig.colorbar(im, cax=cax, orientation="vertical", label=self.unit, ticklocation="right",
                                    location="right")
                ax.coords["ra"].set_axislabel("RA")
                ax.coords["dec"].set_axislabel("Dec")
                ax.coords["ra"].set_major_formatter("d.ddd")
                ax.coords["dec"].set_major_formatter("d.ddd")

                # to overplot here need to do eg. ax.plot(244.979,-15.6400,'bs', transform=ax.get_transform('world'))

                ret = (fig, ax)
            elif "healpix" in projection.lower():
                new_array, footprint, hist = self.healpix_projection(coordsys="galactic", nside=nside)
                if "galactic" in coordsys.lower():
                    coord = ["G"]
                elif "icrs" in coordsys.lower():
                    coord = ["G", "C"]
                else:
                    raise ValueError('This plotting function can only plot the healpix map in galactic or icrs '
                                     'coordinates.')
                mesh = projview(hist.slice[tmin_idx:tmax_idx + 1, :, emin_idx:emax_idx + 1].project("HPX").contents,
                                coord=coord, graticule=True, graticule_labels=True,
                                projection_type="mollweide", reuse_axes=False)
                ret = (mesh)
            else:
                raise ValueError("The projection value only accepts ra/dec or healpix as inputs.")

        return ret


class BatSkyView(object):
    """
    This class holds the information related to a sky image, which is created from a detector plane image

    This is constructed by doing a FFT of the sky image with the detector mask. There can be one or many
    energy or time bins. We can also handle the projection of the image onto a healpix map.

    TODO: create a python FFT deconvolution. This primarily relies on the batfftimage to create the data.
    """

    def __init__(
            self,
            skyimg_file=None,
            dpi_file=None,
            detector_quality_file=None,
            attitude_file=None,
            dpi_data=None,
            input_dict=None,
            recalc=False,
            load_dir=None,
            bkg_dpi_file=None,
            create_bkgvar_map=True,
            create_pcode_map=True
    ):
        """

        :param skyimg_file:
        :param dpi_file:
        :param detector_quality_file:
        :param attitude_file:
        :param dpi_data:
        :param input_dict:
        :param recalc:
        :param load_dir:
        :param bkg_dpi_file:
        """

        if dpi_data is not None:
            raise NotImplementedError(
                "Dealing with the DPI data directly to calculate the sky image is not yet supported.")

        if skyimg_file is not None:
            self.skyimg_file = Path(skyimg_file).expanduser().resolve()

        # do some error checking
        if detector_quality_file is not None:
            self.detector_quality_file = Path(detector_quality_file).expanduser().resolve()
            if not self.detector_quality_file.exists():
                raise ValueError(
                    f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                    f"to exist. Please double check that it does.")
        else:
            self.detector_quality_file = None
            # warnings.warn("No detector quality mask file has been specified. The resulting DPI object "
            #              "will not be able to be modified either by rebinning in energy or time.", stacklevel=2)

    def _call_batfftimage(self, input_dict):
        """
        Calls heasoftpy's batfftimage with an error wrapper, ensures that no runtime errors were encountered

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batfftimage
        :return: heasoftpy Result object from batfftimage

        :param input_dict:
        :return:
        """

        input_dict["clobber"] = "YES"
        input_dict["outtype"] = "DPI"

        try:
            return hsp.batfftimage(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"The call to Heasoft batfftimage failed with inputs {input_dict}."
            )
