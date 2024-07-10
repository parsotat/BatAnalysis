"""
This file holds the BatSkyView object which contains all the necessary sky map information that can be generated from
batfftimage (flux sky image, background variation map, partial coding map).

Tyler Parsotan May 15 2024
"""
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from histpy import Histogram, HealpixAxis, Axis

from .bat_skyimage import BatSkyImage
from .mosaic import _pcodethresh

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatSkyView(object):
    """
    This class holds the information related to a sky image, which is created from a detector plane image

    This is constructed by doing a FFT of the sky image with the detector mask. There can be one or many
    energy or time bins. We can also handle the projection of the image onto a healpix map.

    TODO: create a python FFT deconvolution. This primarily relies on the batfftimage to create the data.
    TODO: make class compatible with mosaiced skyview
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
            create_pcode_img=True,
            create_snr_img=False,
            create_bkg_stddev_img=False,
            sky_img=None,
            bkg_stddev_img=None,
            snr_img=None,
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
        """

        if dpi_data is not None:
            raise NotImplementedError(
                "Dealing with the DPI data directly to calculate the sky image is not yet supported.")

        # do some error checking
        # if the user specified a sky image then use it, otherwise set the sky image to be the same name as the dpi
        # and same location
        if skyimg_file is not None:
            self.skyimg_file = Path(skyimg_file).expanduser().resolve()
        else:
            if sky_img is None:
                self.skyimg_file = dpi_file.parent.joinpath(f"{dpi_file.stem}.img")

        # by defautl set the sky_img attribute to None
        self.sky_img = None

        if dpi_file is not None:
            self.dpi_file = Path(dpi_file).expanduser().resolve()
            if not self.dpi_file.exists():
                raise ValueError(
                    f"The specified DPI file {self.dpi_file} does not seem "
                    f"to exist. Please double check that it does.")
        else:
            # the user could have passed in just a sky image that was previously created and then the dpi file doesnt
            # need to be passed in
            self.dpi_file = dpi_file
            if sky_img is None and (not self.skyimg_file.exists() or recalc):
                raise ValueError("Please specify a DPI file to create the sky image from.")

        if detector_quality_file is not None:
            self.detector_quality_file = Path(detector_quality_file).expanduser().resolve()
            if not self.detector_quality_file.exists():
                raise ValueError(
                    f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                    f"to exist. Please double check that it does.")
        else:
            self.detector_quality_file = "NONE"  # should be replaced with None
            warnings.warn("No detector quality mask file has been specified. Sky images will be constructed assuming "
                          "that all detectors are on.", stacklevel=2)

        # make sure that we have an attitude file (technically we dont need it for batfft, but for BatSkyImage object
        # we do)
        if attitude_file is not None:
            self.attitude_file = Path(attitude_file).expanduser().resolve()
            if not self.attitude_file.exists():
                raise ValueError(
                    f"The specified attitude file {self.attitude_file} does not seem "
                    f"to exist. Please double check that it does.")
        else:
            if sky_img is None and (not self.skyimg_file.exists() or recalc):
                raise ValueError("Please specify an attitude file associated with the DPI.")

        # get the default names of the parameters for batfftimage including its name (which should never change)
        test = hsp.HSPTask("batfftimage")
        default_params_dict = test.default_params.copy()

        if sky_img is None and (not self.skyimg_file.exists() or recalc):
            # fill in defaults, which can be overwritten if values are passed into the input_dict parameter
            self.skyimg_input_dict = default_params_dict
            self.skyimg_input_dict["infile"] = str(self.dpi_file)
            self.skyimg_input_dict["outfile"] = str(self.skyimg_file)
            self.skyimg_input_dict["attitude"] = str(self.attitude_file)
            self.skyimg_input_dict["detmask"] = str(self.detector_quality_file)

            if create_bkg_stddev_img:
                self.bkg_stddev_img_file = self.skyimg_file.parent.joinpath(
                    f"{dpi_file.stem}_bkg_stddev.img")
                self.skyimg_input_dict["bkgvarmap"] = str(self.bkg_stddev_img_file)
            else:
                self.bkg_stddev_img_file = None

            if create_snr_img:
                self.snr_img_file = self.skyimg_file.parent.joinpath(
                    f"{dpi_file.stem}_snr.img")
                self.skyimg_input_dict["signifmap"] = str(self.snr_img_file)
            else:
                self.snr_img_file = None

            if input_dict is not None:
                for key in input_dict.keys():
                    if key in self.skyimg_input_dict.keys():
                        self.skyimg_input_dict[key] = input_dict[key]

            # create all the images that were requested
            self.batfftimage_result = self._call_batfftimage(self.skyimg_input_dict)

            # make sure that this calculation ran successfully
            if self.batfftimage_result.returncode != 0:
                raise RuntimeError(
                    f"The creation of the skyimage failed with message: {self.batfftimage_result.output}"
                )

            # if we want to create the partial coding map then we need to rerun the batfftimage calculation to produce a
            # pcode map that will be able to be passed into batcelldetect
            if create_pcode_img:
                temp_pcodeimg_file = self.skyimg_file.parent.joinpath(
                    f"{dpi_file.stem}.pcodeimg")
                pcodeimg_input_dict = self.skyimg_input_dict.copy()
                pcodeimg_input_dict["pcodemap"] = "YES"
                pcodeimg_input_dict["outfile"] = str(temp_pcodeimg_file)
                # needto make sure that the snr and bkgvar maps are None
                pcodeimg_input_dict["bkgvarmap"] = "NONE"
                pcodeimg_input_dict["signifmap"] = "NONE"

                batfftimage_pcode_result = self._call_batfftimage(pcodeimg_input_dict)

                # make sure that this calculation ran successfully
                if batfftimage_pcode_result.returncode != 0:
                    raise RuntimeError(
                        f"The creation of the associated partial coding map failed with message: {batfftimage_pcode_result.output}"
                    )

                self.pcodeimg_file = temp_pcodeimg_file
            else:
                self.pcodeimg_file = None

        else:
            # set defaults for different attributes
            self.skyimg_input_dict = None
            self.pcodeimg_file = None
            self.snr_img_file = None
            self.bkg_stddev_img_file = None
            self.sky_img = sky_img

        # parse through all the images and get the previous input to batfftimage
        self._parse_skyimages()

        # if the parsing of sky images didnt produce any skymaps b/c we only passed in the images directly
        if self.bkg_stddev_img is None:
            self.bkg_stddev_img = bkg_stddev_img

        if self.snr_img is None:
            self.snr_img = snr_img

        # set the default attribute for projection where we can add the skyviews to be the healpix
        self.projection = "healpix"
        self.healpix_nside = 128
        self.healpix_coordsys = "galactic"

    def _call_batfftimage(self, input_dict):
        """
        Calls heasoftpy's batfftimage with an error wrapper, ensures that no runtime errors were encountered

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batfftimage
        :return: heasoftpy Result object from batfftimage

        :param input_dict:
        :return:
        """

        input_dict["clobber"] = "YES"

        try:
            return hsp.batfftimage(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"The call to Heasoft batfftimage failed with inputs {input_dict}."
            )

    def _call_batcelldetect(self, input_dict):
        """
        Call heasoftpy batcelldetect.

        :param input_dict: dictionary of inputs to pass to batcelldetet.
        :return: heasoft output object
        """

        input_dict["clobber"] = "YES"

        try:
            return hsp.batcelldetect(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f"The call to Heasoft batcelldetect failed with inputs {input_dict}."
            )

    def _parse_skyimages(self):
        """
        This method goes through the sky image file that was produced by batfftimage and reads in all the sky images'
        fits files and saves them as BatSkyImage objects to the appropriate attributes

        batgrbproducts doesnt append the partial coding map to the output, so users can load this file in separately
            by specifying the value of the pcodeimg_file attribute.
        """

        # make sure that the skyimage exists
        if self.sky_img is None and not self.skyimg_file.exists():
            raise ValueError(
                f'The sky image file {self.skyimg_file} does not seem to exist. An error must have occured '
                f'in the creation of this file.')

        # read in the skyimage file and create a SkyImage object. Note that the BatSkyImage.from_file() method
        # read in the first N hdus in the file where N is the number of energy bins that sky images were created for
        # by default, the partial coding map which is set to append_last is not read in
        if self.sky_img is None:
            self.sky_img = BatSkyImage.from_file(self.skyimg_file)

            # read in the history of the sky image that was created
            with fits.open(self.skyimg_file) as f:
                header = f[0].header

            if self.skyimg_input_dict is None:
                # get the default names of the parameters for batbinevt including its name 9which should never change)
                test = hsp.HSPTask("batfftimage")
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

                self.skyimg_input_dict = default_params_dict.copy()

            # see if there is an associated pcode image and that it exists for us to read in
            # since the pcode file can be specified separately, need to verify that it is for the same time range
            if self.pcodeimg_file is not None and self.pcodeimg_file.exists():
                self.pcode_img = BatSkyImage.from_file(self.pcodeimg_file)

            else:
                # see if we can guess the partial coding file's name and see if it exists
                temp_pcodeimg_file = self.skyimg_file.parent.joinpath(
                    f"{self.skyimg_file.stem}.pcodeimg")
                if temp_pcodeimg_file.exists():
                    self.pcodeimg_file = temp_pcodeimg_file
                    self.pcode_img = BatSkyImage.from_file(self.pcodeimg_file)
                else:
                    self.pcode_img = None

            if self.pcode_img is not None:
                # do the time check
                for i in self.pcode_img.tbins.keys():
                    if self.pcode_img.tbins[i] != self.sky_img.tbins[i]:
                        raise ValueError("The timebin of the partial coding image does not align with the sky image."
                                         f"for {i} {self.pcode_img.tbins[i]} != {self.sky_img.tbins[i]}.")

            # see if there are background/snr images for us to read in
            # this can be defined in the history of the batfftimage call or in the constructor method
            # we prioritize anything that was set in the constructor
            if self.snr_img_file is None and self.skyimg_input_dict["signifmap"] != "NONE":
                self.snr_img_file = Path(self.skyimg_input_dict["signifmap"]).expanduser().resolve()

            # now read in the file
            if self.snr_img_file is not None:
                self.snr_img = BatSkyImage.from_file(self.snr_img_file)
            else:
                self.snr_img = None

            if self.bkg_stddev_img_file is None and self.skyimg_input_dict["bkgvarmap"] != "NONE":
                self.bkg_stddev_img_file = Path(self.skyimg_input_dict["bkgvarmap"]).expanduser().resolve()

            # now read in the file
            if self.bkg_stddev_img_file is not None:
                self.bkg_stddev_img = BatSkyImage.from_file(self.bkg_stddev_img_file)
            else:
                self.bkg_stddev_img = None

    @property
    def pcodeimg_file(self):
        return self._pcodeimg_file

    @pcodeimg_file.setter
    def pcodeimg_file(self, value):
        if value is not None:
            temp_value = Path(value).expanduser().resolve()
            if temp_value.exists():
                self._pcodeimg_file = temp_value
                self._parse_skyimages()
            else:
                raise ValueError(f"The file {temp_value} does not exist")
        else:
            self._pcodeimg_file = value

    @property
    def projection(self):
        return self._projection

    @projection.setter
    def projection(self, value):
        if value is not None and "healpix" not in value:
            raise ValueError("The projection attribute can only be set to None or healpix.")

        self._projection = value

    @property
    def healpix_nside(self):
        return self._healpix_nside

    @healpix_nside.setter
    def healpix_nside(self, value):
        if self.projection is not None and "healpix" not in self.projection:
            raise ValueError("The projection attribute needs to be set to healpix before healpix_nside can be set.")

        self._healpix_nside = value

    @property
    def healpix_coordsys(self):
        return self._healpix_coordsys

    @healpix_coordsys.setter
    def healpix_coordsys(self, value):
        if self.projection is not None and "healpix" not in self.projection:
            raise ValueError("The projection attribute needs to be set to healpix before healpix_coordsys can be set.")

        self._healpix_coordsys = value

    @classmethod
    def from_file(cls, skyimg_file, pcodeimg_file=None):
        """

        :param skyimg_file:
        :param pcodeimg_file:
        :return:
        """

        skyview = cls(skyimg_file=skyimg_file)
        skyview.pcodeimg_file = pcodeimg_file

        return skyview

    def detect_sources(self, catalog_file=None, input_dict=None):
        """

        :param catalog_file:
        :param input_dict:
        :return:
        """

        # need the pcode file
        if self.pcodeimg_file is None:
            raise ValueError("Please specify a partial coding file associated with the sky image in order to conduct "
                             "source detection.")

        # use the default catalog is none is specified
        if catalog_file is None:
            catalog_file = (
                Path(__file__).parent.joinpath("data").joinpath("survey6b_2.cat")
            )
        else:
            catalog_file = Path(catalog_file).expanduser().resolve()

        # get the default names of the parameters for batcelldetect including its name (which should never change)
        test = hsp.HSPTask("batcelldetect")
        default_params_dict = test.default_params.copy()

        # fill in defaults, which can be overwritten if values are passed into the input_dict parameter
        self.src_detect_input_dict = default_params_dict
        self.src_detect_input_dict["outfile"] = self.skyimg_file.parent.joinpath(f"{self.skyimg_file.stem}.cat")
        self.src_detect_input_dict["regionfile"] = self.skyimg_file.parent.joinpath(f"{self.skyimg_file.stem}.reg")

        self.src_detect_input_dict["infile"] = str(self.skyimg_file)
        self.src_detect_input_dict["incatalog"] = str(catalog_file)
        self.src_detect_input_dict["pcodefile"] = str(self.pcodeimg_file)
        self.src_detect_input_dict["snrthresh"] = 6
        self.src_detect_input_dict["pcodethresh"] = 0.05
        self.src_detect_input_dict["vectorflux"] = "YES"

        if input_dict is not None:
            for key in input_dict.keys():
                if key in self.skyimg_input_dict.keys():
                    self.skyimg_input_dict[key] = input_dict[key]

        # create all the images that were requested
        self.batcelldetect_result = self._call_batcelldetect(self.src_detect_input_dict)

        # make sure that this calculation ran successfully
        if self.batcelldetect_result.returncode != 0:
            raise RuntimeError(
                f"The detection of sources in the skyimage failed with message: {self.batcelldetect_result.output}"
            )

    def __add__(self, other):
        """
        If we are adding 2 skyviews we can either do
            1) a "simple" add if we want the healpix projection. Here we take the
        partial coding, and variance weighting into account.
            2) a reprojection onto the skyfacets taking the partial coding, the variance weighting, the off-axis
                corrections into account

        :param other:
        :return:
        """
        # make sure that we are adding 2 skyviews
        if not isinstance(other, type(self)):
            raise TypeError(
                "unsupported operand for +: "
                f"'{type(self).__name__}' and '{type(other).__name__}'"
            )

        # make sure we also have the same energy bins
        if not np.array_equal(self.sky_img.ebins["E_MIN"], other.sky_img.ebins["E_MIN"]) or not np.array_equal(
                self.sky_img.ebins["E_MAX"], other.sky_img.ebins["E_MAX"]):
            raise ValueError('Ensure that the two BatSkyView objects have the same energy ranges. ')

        # make sure that both have background stddev and pcode images
        if None in self.pcode_img or None in other.pcode_img:
            raise ValueError('All BatSkyView objects need to have an associated partial coding image to be added')

        if None in self.bkg_stddev_img or None in other.bkg_stddev_img:
            raise ValueError(
                'All BatSkyView objects need to have an associated background standard deviation image to be added')

        if self.projection is not None:
            # we are using the healpix projection to add images, need to get the greater value of healpix_nside and verify that both objects arent None
            nsides = [self.healpix_nside, other.healpix_nside]
            if np.all(None in nsides):
                nsides = 128
            else:
                # need to id the max when we can have None as one of the values in the array
                nsides = np.nanmax(np.array(nsides, dtype=np.float64))

            # now create the histograms that we need to do the calculations, need flux and std dev to be in units of cts/s
            exposure = []
            tstart = []
            tstop = []
            for i, count in zip([self, other], range(2)):
                exposure.append(i.sky_img.exposure)
                tstart.append(i.sky_img.tbins["TIME_START"])
                tstop.append(i.sky_img.tbins["TIME_STOP"])

                # do the healpix projection calculation and get rid of time axis since it is irrelevant now
                flux_hist = i.sky_img.healpix_projection(coordsys=self.healpix_coordsys,
                                                         nside=self.healpix_nside)  # .project("HPX", "ENERGY")
                pcode_hist = i.pcode_img.healpix_projection(coordsys=self.healpix_coordsys,
                                                            nside=self.healpix_nside)  # .project("HPX", "ENERGY")
                bkg_stddev_hist = i.bkg_stddev_img.healpix_projection(coordsys=self.healpix_coordsys,
                                                                      nside=self.healpix_nside)  # .project("HPX", "ENERGY")

                exposure_hist = Histogram(flux_hist.axes,
                                          contents=flux_hist.contents.value * 0 + i.sky_img.exposure.value,
                                          unit=i.sky_img.exposure.unit)

                # correct the units
                if flux_hist.unit != u.count / u.s:
                    flux_hist /= u.s
                if bkg_stddev_hist.unit != u.count / u.s:
                    bkg_stddev_hist /= u.s

                # construct the quality map for each energy and for the total energy images
                energy_quality_mask = np.zeros_like(
                    flux_hist.contents.value)
                good_idx = np.where(
                    (pcode_hist.contents > _pcodethresh
                     )
                    & (bkg_stddev_hist.contents > 0)
                    & np.isfinite(flux_hist.contents)
                    & np.isfinite(bkg_stddev_hist.contents)
                )
                energy_quality_mask[good_idx] = 1
                # make the intermediate images to do operations with
                if count == 0:
                    tot_exposure_hist = deepcopy(exposure_hist) * energy_quality_mask
                    interim_pcode_hist = deepcopy(pcode_hist) * tot_exposure_hist.contents.value
                    interm_inv_var_hist = (1 / (bkg_stddev_hist * bkg_stddev_hist)) * energy_quality_mask
                    interm_flux_hist = flux_hist * interm_inv_var_hist
                else:
                    tot_exposure_hist += deepcopy(exposure_hist) * energy_quality_mask
                    interim_pcode_hist += deepcopy(pcode_hist) * tot_exposure_hist.contents.value
                    interm_inv_var_hist += (1 / (bkg_stddev_hist * bkg_stddev_hist)) * energy_quality_mask
                    interm_flux_hist += flux_hist * interm_inv_var_hist

            # convert to the normal values for flux and bkg std dev
            flux = interm_flux_hist / interm_inv_var_hist
            bkg_stddev = 1 / np.sqrt(interm_inv_var_hist)  # because of the np.sqrt this turns into a u.Quantity array
            snr = flux_hist / bkg_stddev_hist

            tmin = u.Quantity(tstart).min()
            tmax = u.Quantity(tstop).max()
            energybin_ax = self.sky_img.axes["ENERGY"]
            hp_ax = HealpixAxis(nside=self.healpix_nside, coordsys=self.healpix_coordsys, label="HPX")
            t_ax = Axis(u.Quantity([tmin, tmax]), label="TIME")

            # create the SkyImages for each quantity
            flux = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=flux.contents, unit=flux.unit))
            bkg_stddev = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=bkg_stddev.value, unit=bkg_stddev.unit))
            snr = BatSkyImage(image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=snr.contents, unit=snr.unit))

            tot_exposure = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=tot_exposure_hist.contents,
                                     unit=tot_exposure_hist.unit))
            pcode = BatSkyImage(image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=interim_pcode_hist.contents,
                                                     unit=interim_pcode_hist.unit))

            interm_flux = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=interm_flux_hist.contents,
                                     unit=interm_flux_hist.unit), is_mosaic=True)
            interm_inv_var = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=interm_inv_var_hist.contents,
                                     unit=interm_inv_var_hist.unit), is_mosaic=True)

            test_mosaic = BatMosaicSkyView(interm_flux, interm_inv_var, pcode, tot_exposure)

            return flux, bkg_stddev, snr, tot_exposure, pcode, test_mosaic
        else:
            raise NotImplementedError("Adding Sky Images with the template sky facets is not yet implemented.")


class BatMosaicSkyView(BatSkyView):
    """
    This is a special case of a skyview where the data that is held is the intermediate maps that can be easily added
    """

    def __init__(self, interim_sky_img, interim_var_img, pcode_img, exposure_img):
        self.interim_sky_img = interim_sky_img
        self.interim_var_img = interim_var_img
        self.pcode_img = pcode_img
        self.exposure_img = exposure_img

        super().__init__(sky_img=self.sky_img, bkg_stddev_img=self.bkg_stddev_img, snr_img=self.snr_img)

    @property
    def sky_img(self):
        return self.interim_sky_img / self.interim_var_img

    @sky_img.setter
    def sky_img(self, value):
        self._sky_img = value

    @property
    def bkg_stddev_img(self):
        return 1 / np.sqrt(self.interim_var_img)

    @bkg_stddev_img.setter
    def bkg_stddev_img(self, value):
        self._bkg_stddev_img = value

    @property
    def snr_img(self):
        return self.sky_img / self.bkg_stddev_img

    @snr_img.setter
    def snr_img(self, value):
        self._snr_img = value
