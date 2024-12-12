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
from astropy.io import fits
from astropy.wcs import WCS
from healpy.newvisufunc import projview
from histpy import Histogram, HealpixAxis
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reproject import reproject_to_healpix

# list out the image extensions that we can read in. These should be lowercase.
# note that the snr and background stddev images dont have gti or energy extensions. These are gotten from the
# headers of the images themselves
_file_extension_names = ["image", "pcode", "signif", "varmap"]
_accepted_image_types = ["flux", "pcode", "snr", "stddev", "exposure"]


class BatSkyImage(Histogram):
    """
    This class holds the information related to a sky image, which is created from a detector plane image

    This is constructed by doing a FFT of the sky image with the detector mask. There can be one or many
    energy or time bins. We can also handle the projection of the image onto a healpix map.

    This hold data correspondent to BAT's view of the sky. This can be a flux map (sky image) created from a FFT
    deconvolution, a partial coding map, a significance map, or a background variance map. These can all be energy
    dependent including the partial coding map (though the partial coding map itself is independent of energy).

    This class can also hold information related to a mosaiced image such as the intermediate images or the final
    transformed images.

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
            wcs=None,
            is_mosaic_intermediate=False,
            image_type=None
    ):
        """
        This class is meant to hold images of the sky that have been created from a deconvolution of the BAT DPI with
        the coded mask. The sky data can represent flux as a function of energy, background stddev as a function of
        energy, the significance map as a function of energy, and the partial coding map.

        This class holds an image for a single time bin for simplicity. Since this class is generally written to hold
        sky images that can be fluxes, background stddev, signal to noise ratios, etc, the direct summation of energy 
        binned quantities to produce energy integrated quantities are not always valid. ie flux in different energy bins
        can be added directly but the background stddev in each energy bin has to be added in quadrature. To account for
        these differences, the user must specify what type of image the object contains. This has to be a string 
        correspondent to one of the following: "flux", "pcode", "snr", "stddev", "exposure"

        :param image_data: None or, a numpy array, an astropy Quantity array, or a Histogram object. If a numpy array is passed
            in the units will be assumed to be counts. If a Quantity array or a Histogram is passed in then the
            units will be obtained from those objects. If a Histogram object is passed in, the time, space, and energy
            axes will be inherited from the Histogram and passing in other parameters to initalize a BatSkyImage will be
            ignored.
        :param timebins: None or an astropy Quantity array with the timebin edges that define the timebinning of the image_data
            that is passed in.
            NOTE: if a Histogram object is passed in for image_data, then anything passed into this parameter is ignored
        :param tmin: None or an astropy.units.Quantity denoting the minimum values of the timebin edges that the image_data was
            binned with respect to.
            NOTE: if the timebins parameter is passed in then anything passed into tmin/tmax is ignored
            NOTE: if a Histogram object is passed in for image_data, then anything passed into this parameter is ignored
        :param tmax: None or an astropy.units.Quantity denoting the maximum values of the timebin edges that the image_data was
            binned with respect to.
            NOTE: if the timebins parameter is passed in then anything passed into tmin/tmax is ignored
            NOTE: if a Histogram object is passed in for image_data, then anything passed into this parameter is ignored
        :param energybins: None or an astropy Quantity object outlining the energy bin edges that the sky image has been binned into
        :param emin: None or an an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the sky image has been binned into.
            NOTE: If the energybins is specified, the emin/emax parameters are ignored.
            NOTE: if a Histogram object is passed in for image_data, then anything passed into this parameter is ignored
        :param emax: None or an an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the sky image has been binned into.
            NOTE: If the energybins is specified, the emin/emax parameters are ignored.
            NOTE: if a Histogram object is passed in for image_data, then anything passed into this parameter is ignored
        :param weights: None or a numpy array with the same shape as image_data that defines the weight in each pixel
            of the sky image
        :param wcs: None or an astropy WCS object that defines the world coordinate system for the BAtSkyImage
        :param is_mosaic_intermediate: Boolean to denote if this BatSkyImage object contains intermediate mosaic images.
            This helps with defining how projections are done in energy. 
        :param image_type: string to denote the type of image that is contained in the object. This is necessary 
            information to define how projections in energy are done in the BatSkyImage. 
        """

        # do some error checking
        if image_data is None:
            raise ValueError(
                "A numpy array or a Histpy.Histogram needs to be passed in to initalize a BatSkyImage object"
            )

        if image_type is not None:
            # make sure it is one of the strings that we recognize internally
            if type(image_type) is not str or not np.any([i == image_type for i in _accepted_image_types]):
                raise TypeError(
                    f"The image_type must be a string that corresponds to one of the following: {_accepted_image_types}")

        if wcs is None:
            warnings.warn(
                "No astropy World Coordinate System has been specified the sky image is assumed to be in the detector "
                "tangent plane. No conversion to Healpix or RA/Dec & galactic coordinate systems will be possible.",
                stacklevel=2,
            )
        else:
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

        # set whether we have a mosaic intermediate image and what type of image we have
        self.is_mosaic_intermediate = is_mosaic_intermediate
        self.image_type = image_type

    def _set_histogram(self, histogram_data=None, event_data=None, weights=None):
        """
        This method properly initalizes the Histogram parent class. it uses the self.tbins and self.ebins information
        to define the time and energy binning for the histogram that is initalized.

        COPIED from DetectorPlaneHist class, can be organized better.

        :param histogram_data: None or histpy Histogram or a numpy array of N dimensions. This should be formatted
            such that it has the following dimensions: (T,Ny,Nx,E) where T is the number of timebins, Ny is the
            number of image pixels in the y direction, Nx represents an identical
            quantity in the x direction, and E is the number of energy bins. These should be the appropriate sizes for
            the tbins and ebins attributes
        :param event_data: None or TimeTaggedEvents class that has been initialized with event data (Not used currently)
        :param weights: None or the weights with the same size as event_data or histogram_data
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
        if isinstance(histogram_data, u.Quantity) or isinstance(histogram_data, Histogram):
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
                labels=["TIME", "SKYY", "SKYX", "ENERGY"],
                sumw2=weights,
                unit=hist_unit,
            )
        else:
            # we need to explicitly set the units, so if we have a Quantity object need to do
            # histogram_data.contents.value
            if isinstance(histogram_data.contents, u.Quantity):
                super().__init__(
                    histogram_data.axes,
                    contents=histogram_data.contents.value,
                    labels=histogram_data.axes.labels,
                    unit=hist_unit,
                )
            else:
                super().__init__(
                    histogram_data.axes,
                    contents=histogram_data.contents,
                    labels=histogram_data.axes.labels,
                    unit=hist_unit,
                )
            # for some reason if we try to initialize the parent class when there is a healpix axis in the Histogram that
            # we are using for the intialization, then the self.axes wont have the "HPX" axis as a healpixaxis and we
            # wont be able to access any of the relevant methods for that axis. Therefore try to set the axes explicitly
            if "HPX" in histogram_data.axes.labels:
                self._axes = histogram_data.axes

    def healpix_projection(self, coordsys="galactic", nside=128):
        """
        This creates a healpix projection of the image. This currently works for going from the batfftimage sky pixel to
        the healpix projection.

        TODO: how to deal with having a sky image that is a healpix projection and wanting to change that
            healpix projection resolution? What about changing the coord sys?

        :param coordsys: str defining the coordinate system of the output healpix map. This can be "galactic" or "icrs"
        :param nside: int representing the resolution of the healpix map
        :return: a BatSkyImage object with the healpix projection of the original sky image
        """

        """
        Note for converting image(s) to mhealpy. only works for a single time/energybin
        eg:
        from mhealpy import HealpixMap, HealpixBase
        snr_hpx=skyview.snr_img.healpix_projection(nside=512)
        snr_hpx_map=HealpixMap(data=snr_hpx.contents[0,:,0], coordsys="G")
        fov_pix = np.nonzero(np.logical_not(np.isnan(snr_hpx_map)))[0]
        m_moc = HealpixMap.moc_from_pixels(nside = snr_hpx_map.nside, 
                                   nest = snr_hpx_map.is_nested, 
                                   pixels = fov_pix, coordsys="G")
        m_moc[:] = 1
        m_moc*=snr_hpx_map
        """

        if "HPX" not in self.axes.labels:

            # create our new healpix axis
            hp_ax = HealpixAxis(nside=nside, coordsys=coordsys, label="HPX")

            # create a new array to hold the projection of the sky image in detector tangent plane coordinates to healpix
            # coordinates
            new_array = np.zeros((self.axes['TIME'].nbins, hp_ax.nbins, self.axes["ENERGY"].nbins))

            # for each time/energybin do the projection (ie linear interpolation)
            # reproject_to_healpix does not support passing in a multidimensional array where the coordinates in the
            # extra dimension is assumed to be the same (as of 2024)
            for t in range(self.axes['TIME'].nbins):
                for e in range(self.axes["ENERGY"].nbins):
                    array, footprint = reproject_to_healpix(
                        (self.slice[t, :, :, e].project("SKYY", "SKYX").contents, self.wcs), coordsys,
                        nside=nside)
                    new_array[t, :, e] = array

            # create the new histogram
            h = BatSkyImage(Histogram(
                [self.axes['TIME'], hp_ax, self.axes["ENERGY"]],
                contents=new_array, unit=self.unit), image_type=self.image_type,
                is_mosaic_intermediate=self.is_mosaic_intermediate)

            # can return the histogram or choose to modify the class histogram. If the latter, need to get way to convert back
            # to detector plane coordinates
            # return new_array, footprint, h
        else:
            # need to verify that we have the healpix axis in the correct coordinate system and with correct nsides
            if self.axes["HPX"].nside != nside:
                raise ValueError(
                    "The requested healpix nsides for the BatSkyImage is different from what is contained in the object.")

            if self.axes["HPX"].coordsys.name != coordsys:
                raise ValueError(
                    "The requested healpix coordinate system of the BatSkyImage object is different from what is contained in the object.")

            h = BatSkyImage(image_data=Histogram(self.axes, contents=self.contents, unit=self.unit),
                            image_type=self.image_type, is_mosaic_intermediate=self.is_mosaic_intermediate)

        return h

    def calc_radec(self):
        """
        Calculates the RA/Dec from the sky image pixels using the associated WCS that was passed in.
        The units are degrees. The shapes are the same as that of the sky image.

        :return: ra, dec
        """
        from .mosaic import convert_xy2radec

        if self.wcs is None:
            raise ValueError(
                "Cannot convert from the sky image pixel coordinates to RA/Dec without the WCS information.")

        x = np.arange(self.axes["SKYX"].nbins)
        y = np.arange(self.axes["SKYY"].nbins)
        xx, yy = np.meshgrid(x, y)

        ra, dec = convert_xy2radec(xx, yy, self.wcs)

        c = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")

        return c.ra, c.dec

    def calc_glatlon(self):
        """
        Calculates the galatic latitude/longitude from the sky image pixels using the associated WCS that was passed in.
        The units are degrees. The shapes are the same as that of the sky image.

        :return:galactic latitude, galactic longitude
        """
        if self.wcs is None:
            raise ValueError(
                "Cannot convert from the sky image pixel coordinates to galactic coordinates without the WCS information.")

        ra, dec = self.calc_radec()

        c = SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg")

        return c.galactic.l, c.galactic.b

    @u.quantity_input(emin=["energy"], emax=["energy"], tmin=["time"], tmax=["time"])
    def plot(self, emin=None, emax=None, tmin=None, tmax=None, projection=None, coordsys="galactic", nside=128):
        """
        This is a convenience plotting function that allows for quick and easy plotting of a sky image. It allows for
        energy and time (where applicable) slices and different representations of the sky image.

        TODO: consider if the image is already a healpix projection with some nside and the user wants to plot with a
            different nside value

        :param emin: None or an astropy Quantity object that defines the minimum energy range of the sky image
            that will be plotted. If this is None, it will default to the minimum energy defined in the sky image itself.
        :param emax: None or an astropy Quantity object that defines the maximum energy range of the sky image
            that will be plotted. If this is None, it will default to the maximum energy defined in the sky image itself.
        :param tmin: None or an astropy Quantity object that defines the minimum time range of the sky image
            that will be plotted. If this is None, it will default to the minimum time defined in the sky image itself.
            NOTE: This parameter currently does not do anything as sky images only hold images for a single timebin
        :param tmax: None or an astropy Quantity object that defines the maximum time range of the sky image
            that will be plotted. If this is None, it will default to the maximum time defined in the sky image itself.
            NOTE: This parameter currently does not do anything as sky images only hold images for a single timebin
        :param projection: None or a string that denotes if the plotted sky image should be reprojected onto a healpix map
            of a plot with ra/dec coordinates. This is only possible if the sky image is that of
            the sky pixel values (direct from batfftimage) The accepted
            strings are: "healpix" or "ra/dec". If None is passed in then the projection will be that of the sky image
            itself. ie if the sky image is already in a healpix map, that is the projection that will be plotted.
        :param coordsys: string denoting if the galactic or icrs coordinate system should be used for a healpix map.
            Valid options for this parameter are "galactic" and "icrs"
        :param nside: if projection="healpix", then this defines the healpix projection's resolution. If the sky image
            that is being plotted is already a healpix projection then this parameter does nothing and the native
            healpix resolution of the sky image is used.
        :return: (matplotlib axis object, matplotlib Quadmesh object) or just the matplotlib Quadmesh object
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

        # for mosaic images, cannot do normal projection with summing up
        if self.is_mosaic_intermediate:
            if tmax_idx - tmin_idx > 1 or emax_idx - emin_idx > 1:
                raise ValueError(
                    f"Cannot do normal addition of a mosaiced image. Please choose a single time/energy bin to plot.")

        # now do the plotting
        if projection is None:
            # use the default spatial axes of the histogram
            # need to determine what this is
            if "SKYX" in self.axes.labels:
                ax, mesh = BatSkyImage(image_data=self.slice[tmin_idx:tmax_idx, :, :, emin_idx:emax_idx],
                                       image_type=self.image_type,
                                       is_mosaic_intermediate=self.is_mosaic_intermediate).project("SKYX",
                                                                                                   "SKYY").plot()
                ret = (ax, mesh)
            elif "HPX" in self.axes.labels:
                if "galactic" in coordsys.lower():
                    coord = ["G"]
                elif "icrs" in coordsys.lower():
                    coord = ["G", "C"]
                else:
                    raise ValueError('This plotting function can only plot the healpix map in galactic or icrs '
                                     'coordinates.')
                plot_quantity = BatSkyImage(image_data=self.slice[tmin_idx:tmax_idx, :, emin_idx:emax_idx],
                                            image_type=self.image_type,
                                            is_mosaic_intermediate=self.is_mosaic_intermediate).project(
                    "HPX").contents
                if isinstance(plot_quantity, u.Quantity):
                    plot_quantity = plot_quantity.value

                mesh = projview(plot_quantity,
                                coord=coord, graticule=True, graticule_labels=True,
                                projection_type="mollweide", reuse_axes=False)
                ret = (mesh)
            else:
                raise ValueError("The spatial projection of the sky image is currently not accepted as a plotting "
                                 "option. Please convert to SKYX/SKYY or a HEALPix map.")
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
                    BatSkyImage(image_data=self.slice[tmin_idx:tmax_idx, :, :, emin_idx:emax_idx],
                                image_type=self.image_type,
                                is_mosaic_intermediate=self.is_mosaic_intermediate).project("SKYY",
                                                                                            "SKYX").contents.value,
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
                hist = self.healpix_projection(coordsys="galactic", nside=nside)
                if "galactic" in coordsys.lower():
                    coord = ["G"]
                elif "icrs" in coordsys.lower():
                    coord = ["G", "C"]
                else:
                    raise ValueError('This plotting function can only plot the healpix map in galactic or icrs '
                                     'coordinates.')

                plot_quantity = BatSkyImage(image_data=hist.slice[tmin_idx:tmax_idx, :, emin_idx:emax_idx],
                                            image_type=self.image_type,
                                            is_mosaic_intermediate=self.is_mosaic_intermediate).project("HPX").contents
                if isinstance(plot_quantity, u.Quantity):
                    plot_quantity = plot_quantity.value

                mesh = projview(plot_quantity,
                                coord=coord, graticule=True, graticule_labels=True,
                                projection_type="mollweide", reuse_axes=False)
                ret = (mesh)
            else:
                raise ValueError("The projection value only accepts ra/dec or healpix as inputs.")

        return ret

    @classmethod
    def from_file(cls, file):
        """
        This convenience class method allows a pre-constructed sky image to be read in and used to construct a new
        BatSkyImage object. This method can currently process images that have these extension header
        names: "image", "pcode", "signif", "varmap"

        TODO: be able to parse the skyfacet files for mosaicing images
        
        :param file: string of Path object to the sky image file that will be processed
        :return: BatSkyImage object
        """

        input_file = Path(file).expanduser().resolve()
        if not input_file.exists():
            raise ValueError(f"The specified sky image file {input_file} does not seem to exist. "
                             f"Please double check that it does.")

        # read the file header
        img_headers = []
        energy_header = None
        time_header = None
        with fits.open(input_file) as f:
            for i in range(len(f)):
                header = f[i].header
                # if we have an image, save it to our list of image headers
                # if "image" in header["EXTNAME"].lower():
                if np.any([name in header["EXTNAME"].lower() for name in _file_extension_names]):
                    img_headers.append(header)
                elif "ebounds" in header["EXTNAME"].lower():
                    energy_header = header
                elif "stdgti" in header["EXTNAME"].lower():
                    time_header = header
                else:
                    raise ValueError(
                        f'An unexpected header extension name {header["EXTNAME"]} was encountered. This class can '
                        f'only parse sky image files that have {_file_extension_names}, EBOUNDS, and STDGTI header extensions. ')

        # now we can construct the data for the time bins, the energy bins, the total sky image array, and the WCS
        w = WCS(img_headers[0])

        # the partial coding image has no units so make sure that only when we are reading in a pcoding or snr file we
        # have this set
        if np.all(["pcode" in i["EXTNAME"].lower() for i in img_headers]) or np.all(
                ["signif" in i["EXTNAME"].lower() for i in img_headers]):
            img_unit = 1 * u.dimensionless_unscaled
        else:
            img_unit = u.Quantity(f'1{img_headers[0]["BUNIT"]}')

        if time_header is not None:
            time_unit = u.Quantity(f'1{time_header["TUNIT1"]}')  # expect seconds
            n_times = time_header["NAXIS2"]
        else:
            time_unit = 1 * u.s
            n_times = 1

        if energy_header is not None:
            energy_unit = u.Quantity(f'1{energy_header["TUNIT2"]}')  # expect keV
            n_energies = energy_header["NAXIS2"]
        else:
            energy_unit = 1 * u.keV
            n_energies = len(img_headers)

        # make sure that we only have 1 time bin (want to enforce this for mosaicing)
        if n_times > 1:
            raise NotImplementedError("The number of timebins for the sky images is greater than 1, which is "
                                      "currently not supported.")

        # maek sure that the number of energy bins is equal to the number of images to read in
        if len(img_headers) != n_energies:
            raise ValueError(
                f'The number of energy bins, {n_energies}, is not equal to the number of images to read in {len(img_headers)}.')

        img_data = np.zeros((n_times, img_headers[0]["NAXIS2"], img_headers[0]["NAXIS1"],
                             n_energies))

        # here we assume that the images are ordered in energy and only have 1 timebin
        energy_data = None
        time_data = None
        with fits.open(input_file) as f:
            for i in range(len(f)):
                data = f[i].data
                header = f[i].header
                # if we have an image, save it to our list of image headers
                # if "image" in header["EXTNAME"].lower():
                if np.any([name in header["EXTNAME"].lower() for name in _file_extension_names]):
                    img_data[:, :, :, i] = data
                elif "ebounds" in header["EXTNAME"].lower():
                    energy_data = data
                elif "stdgti" in header["EXTNAME"].lower():
                    time_data = data
                else:
                    raise ValueError(
                        f'An unexpected header extension name {header["EXTNAME"]} was encountered. This class can '
                        f'only parse sky image files that have IMAGE, EBOUNDS, and STDGTI header extensions. ')

        # set the unit for the sky image
        img_data *= img_unit.unit

        # parse the time/energy to initalize our BatSkyImage
        if time_data is not None:
            min_t = np.squeeze(time_data["START"] * time_unit.unit)
            max_t = np.squeeze(time_data["STOP"] * time_unit.unit)
        else:
            min_t = img_headers[0]["TSTART"] * time_unit.unit
            max_t = img_headers[0]["TSTOP"] * time_unit.unit

        if energy_data is not None:
            min_e = energy_data["E_MIN"] * energy_unit.unit
            max_e = energy_data["E_MAX"] * energy_unit.unit
        else:
            min_e = [i["E_MIN"] for i in img_headers] * energy_unit.unit
            max_e = [i["E_MAX"] for i in img_headers] * energy_unit.unit

        # define the image type, we have confirmed that is is one of the accepted types, just need to ID which
        # then have to convert the weird file extension to a string that is accepted by the constructor.
        # Note exposure doesnt have an equivalent
        # _file_extension_names = ["image", "pcode", "signif", "varmap"]
        # _accepted_image_types = ["flux", "pcode", "snr", "stddev", "exposure"]

        imtype = [name for name in _file_extension_names if name in img_headers[0]["EXTNAME"].lower()][0]
        if "image" in imtype:
            imtype = "flux"
        elif "signif" in imtype:
            imtype = "snr"
        elif "varmap" in imtype:
            imtype = "stddev"

        return cls(image_data=img_data, tmin=min_t, tmax=max_t, emin=min_e, emax=max_e, wcs=w, image_type=imtype)

    def project(self, *axis):
        """
        This overwrites the Histogram class' project method.
            1) If we have a non-intermediate-mosaic background stddev/snr image, we need to add quantities in quadrature
                instead of just adding the energy bins.
            2) If we have a mosaic intermediate image or a flux image, we can just add directly and calls the Histogram
                project method as normal on the object itself.
            3) If we have an exposure image or a pcode image, then a projection over energy is irrelevant and we just
                want to return a slice of the Histogram (if there is more than 1 energy)
            4) If "ENERGY" is a value specified in axes, then we dont need to worry about any of this
        
        :param axis: gets passed into the Histogram project method.
        :return: Histogram object with the proper projection done and the axes that were requested
        """

        # if energy is not specified as a remaining axis OR if there is only 1 energy bin then we dont need to worry
        # about all these nuances. If the image type is not specified, then also go to the normal behavior
        if ("ENERGY" not in [i for i in axis] and self.axes["ENERGY"].nbins > 1) and self.image_type is not None:
            # check to see if we have images that are not intermediate mosaic images and they are stddev/snr quantities
            if not self.is_mosaic_intermediate and np.any([self.image_type == i for i in ["snr", "stddev"]]):
                # because the self.project is recursive below, we need to create a new Histogram object so we call that
                temp_hist = Histogram(edges=self.axes, contents=(self * self).contents.value)
                ax = [self.axes[i] for i in axis]

                hist = Histogram(edges=ax, contents=np.sqrt(temp_hist.project(*axis)), unit=self.unit)

            elif np.any([self.image_type == i for i in ["pcode", "exposure"]]):
                # this gets executed even if self.is_mosaic_intermediate is True
                if "HPX" in self.axes.labels:
                    # only have 1 spatial dimension
                    hist = self.slice[:, :, self.end - 1].project(*axis)
                else:
                    # have 2 spatial dimensions
                    hist = self.slice[:, :, :, self.end - 1].project(*axis)
            elif self.is_mosaic_intermediate or self.image_type == "flux":
                hist = super().project(*axis)
            else:
                # capture all other cases with error
                raise ValueError("Cannot do normal sum over energy bins for this type of image.")
        else:
            # warn the user that this image_type isnt set and that we will be using the default behavior to sum
            # over energy
            if self.image_type is None and ("ENERGY" not in [i for i in axis] and self.axes["ENERGY"].nbins > 1):
                warnings.warn(
                    "The image type for this object has not been specified. Defaulting to summing up the Histogram values over the ENERGY axis",
                    stacklevel=2,
                )

            hist = super().project(*axis)

        return hist
