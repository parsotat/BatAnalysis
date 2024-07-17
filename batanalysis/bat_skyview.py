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

from .bat_dpi import BatDPI
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
    """

    def __init__(
            self,
            skyimg_file=None,
            bat_dpi=None,
            attitude_file=None,
            input_dict=None,
            recalc=False,
            load_dir=None,
            create_pcode_img=True,
            create_snr_img=False,
            create_bkg_stddev_img=False,
            # sky_img=None,
            # bkg_stddev_img=None,
            # snr_img=None,
            interim_sky_img=None,
            interim_var_img=None,
            pcode_img=None,
            exposure_img=None
    ):
        """
        A sky view object contains various sky images that are associated with a view of the sky in a single timebin.
        These images can be the flux sky image, at a miminum, and can include other images such as a partial coding
        image, a background standard deviation image, and a SNR image showing the significance of each image pixel.

        This class is very flexible in being able to
            1) construct sky flux images from dpi's, including snr, background standard deviation, and partial coding
                images
            2) load in preconstructed sky flux images with the associated snr, background standard deviation, and
                partial coding images
            2) hold mosaic sky images, including their intermediate images that can easily be summed together

        For this class to hold information related to a sky image, the interm_sky_img, interim_var_img, pcode_img, and
        exposure_img all need to be passed in. In this case, any other parameters passed in will be ignored.

        To contain information related to a canonical flux sky image, including an associated background standard
        deviation sky image, a SNR sky image, and a partial coding image, all other parameters play a role. These allow
        the class to either create a new set of sky images (which collectively we call a sky view) and load in those
        images OR load in pre-calculated sky images that are of the same view of the sky.

        :param skyimg_file: None, or a Path object to the flux sky image file that will either be read in, if it exists,
            or created via batfftimage. If None is specified, a flux sky image file name will be set to be the same as
            the dpi_file (except the suffix will be .img instead of .dpi) and the directory that it will be assumed to
            be contained in will be the same as dpi_file. If the assumed flux sky image exists, it will be reloaded so
            long as recalc=False, otherwise the flux sky image will be created with batfftimage
        :param bat_dpi: None or a BatDPI object that contains the DPI file that will be used to produce a flux sky image in the case
            where skyimg_file is None or the passed in skyimg_file does not exist. The BatDPI must have the dpi_file and
            detector_quality_file attributes defined.
        :param attitude_file: None or a Path object to the attitude file associated with the DPI. If a new flux sky
            is being created, this file is needed.
        :param input_dict: None or dict of key/value pairs that will be passed to batfftimage. If this is set to None,
            the default batfftimage parameter values will be used. If a dictionary is passed in, it will overwrite the
            default values
        :param recalc: boolean to denote if the sky image and associated images should be loaded or completely
            recalculated
        :param load_dir: Not implemented yet
        :param create_pcode_img: bool to denote if the partial coding should be calculated with the flux sky image.
            If the sky view will be added to others, then this is necessary
        :param create_snr_img: bool to denote if the SNR image map should be calculated with the
            flux sky image
        :param create_bkg_stddev_img: bool to denote if the background standard deviation image should be calculated with the
            flux sky image.
            If the sky view will be added to others, then this is necessary
        :param sky_img: Placeholder, Not implemented yet
        :param bkg_stddev_img: Placeholder, Not implemented yet
        :param snr_img: Placeholder, Not implemented yet
        :param interim_sky_img: None or a BatSkyImage object that holds the intermediate mosaic flux sky image
            NOTE: for the BatSkyView to recognize that it contains mosaiced images this parameter needs to be supplied
        :param interim_var_img:None or a BatSkyImage object that holds the intermediate mosaic inverse variance sky
            image
            NOTE: for the BatSkyView to recognize that it contains mosaiced images this parameter needs to be supplied
        :param pcode_img: None or a BatSkyImage object that holds the intermediate mosaic partial coding exposure sky
            image
            NOTE: for the BatSkyView to recognize that it contains mosaiced images this parameter needs to be supplied
        :param exposure_img:None or a BatSkyImage object that holds the intermediate mosaic exposure sky image
            NOTE: for the BatSkyView to recognize that it contains mosaiced images this parameter needs to be supplied

        """

        # do some error checking
        # The user can either specify parameters related to the creation of a normal sky image (and associated images) or
        # they can specify all the intermediate images that are created in the image mosaicing process. If the class
        # is holding mosaicing data, only want to compute the sky image (and associated images) once, and only if the
        # user requests them. If the class is used to hold the normal sky images etc, then there should be no attributes
        # related to the intermiediate images.

        # if the user wants to use this to hold mosaic image data then they need to pass in all the necessary images
        # and we will skip over anything related to reading a sky image from a file, etc. If they didnt pass in all the
        # images, then raise an error.

        # if all necessary inputs for a mosaic image are None, then is_mosaic=False
        self.is_mosaic = not np.all([i is None for i in [interim_sky_img, interim_var_img, pcode_img, exposure_img]])
        if self.is_mosaic:
            # if is_mosac is True, this can mean that at least one of the parameters passed in is not None,
            # need to check if any are None now. If there are any that are None, throw an error b/c user needs to pass
            # all images in. also make sure that they are all BaSkyImage objects
            if np.any([i is None for i in [interim_sky_img, interim_var_img, pcode_img, exposure_img]]):
                raise ValueError(
                    "To properly create a BatSkyView from mosaics, the intermediate sky flux, background variance, partial coding vignette, and exposure images need to be passed in.")

            if np.any([not isinstance(i, BatSkyImage) for i in
                       [interim_sky_img, interim_var_img, pcode_img, exposure_img]]):
                raise ValueError(
                    "To properly create a BatSkyView from mosaics, the intermediate sky flux, background variance, partial coding vignette, and exposure images all need to be BatSkyImage objects.")

        # if self.is_mosaic is true, we dont care about the stuff related to creating a sky image from a DPI
        if not self.is_mosaic:

            # make sure we have the correct object for bat_dpi
            if bat_dpi is not None:
                if not isinstance(bat_dpi, BatDPI):
                    raise ValueError("The input to the bat_dpi parameter must be a BatDPI object. ")

            # if the user specified a sky image then use it, otherwise set the sky image to be the same name as the dpi
            # and same location
            self.skyimg_file = None
            if skyimg_file is not None:
                self.skyimg_file = Path(skyimg_file).expanduser().resolve()
            else:
                # make sure that we can define the output sky image filename
                if bat_dpi.dpi_file is not None:
                    self.skyimg_file = bat_dpi.dpi_file.parent.joinpath(f"{bat_dpi.dpi_file.stem}.img")
                else:
                    raise ValueError(
                        "The BatDPI object passed to bat_dpi must have the dpi_file attribute defined to create the sky image from.")

            if bat_dpi.dpi_file is not None:
                self.dpi_file = Path(bat_dpi.dpi_file).expanduser().resolve()
                if not self.dpi_file.exists():
                    raise ValueError(
                        f"The specified DPI file {self.dpi_file} does not seem "
                        f"to exist. Please double check that it does.")
            else:
                # the user could have passed in just a sky image that was previously created and then the dpi file doesnt
                # need to be passed in
                self.dpi_file = None
                if not self.skyimg_file.exists() or recalc:
                    raise ValueError(
                        "The BatDPI object passed to bat_dpi must have the dpi_file attribute defined to create the sky image from.")

            if bat_dpi.detector_quality_file is not None:
                self.detector_quality_file = Path(bat_dpi.detector_quality_file).expanduser().resolve()
                if not self.detector_quality_file.exists():
                    raise ValueError(
                        f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                        f"to exist. Please double check that it does.")
            else:
                self.detector_quality_file = None
                warnings.warn(
                    "No detector quality mask file has been specified. Sky images will be constructed assuming "
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
                if not self.skyimg_file.exists() or recalc:
                    raise ValueError("Please specify an attitude file associated with the DPI.")

            # get the default names of the parameters for batfftimage including its name (which should never change)
            test = hsp.HSPTask("batfftimage")
            default_params_dict = test.default_params.copy()

            if not self.skyimg_file.exists() or recalc:
                # fill in defaults, which can be overwritten if values are passed into the input_dict parameter
                self.skyimg_input_dict = default_params_dict
                self.skyimg_input_dict["infile"] = str(self.dpi_file)
                self.skyimg_input_dict["outfile"] = str(self.skyimg_file)
                self.skyimg_input_dict["attitude"] = str(self.attitude_file)

                if self.detector_quality_file is not None:
                    self.skyimg_input_dict["detmask"] = str(self.detector_quality_file)
                else:
                    self.skyimg_input_dict["detmask"] = "NONE"

                if create_bkg_stddev_img:
                    self.bkg_stddev_img_file = self.skyimg_file.parent.joinpath(
                        f"{self.dpi_file.stem}_bkg_stddev.img")
                    self.skyimg_input_dict["bkgvarmap"] = str(self.bkg_stddev_img_file)
                else:
                    self.bkg_stddev_img_file = None

                if create_snr_img:
                    self.snr_img_file = self.skyimg_file.parent.joinpath(
                        f"{self.dpi_file.stem}_snr.img")
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
                        f"{self.dpi_file.stem}.pcodeimg")
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

            # parse through all the images and get the previous input to batfftimage
            self._parse_skyimages()

        else:
            # just set the mosaic intermediate images
            self.interim_sky_img = interim_sky_img
            self.interim_var_img = interim_var_img
            self.pcode_img = pcode_img
            self.exposure_img = exposure_img

        # set the default attribute for projection where we can add the skyviews to be the healpix
        self.projection = "healpix"
        self.healpix_nside = 128
        self.healpix_coordsys = "galactic"

    def _call_batfftimage(self, input_dict):
        """
        Calls heasoftpy's batfftimage with an error wrapper, ensures that no runtime errors were encountered

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batfftimage
        :return: heasoftpy Result object from batfftimage

        :param input_dict: dictionary of key/value pairs that will be passed to batfftimage
        :return: heasoft result object
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
        if not self.skyimg_file.exists():
            raise ValueError(
                f'The sky image file {self.skyimg_file} does not seem to exist. An error must have occured '
                f'in the creation of this file.')

        # read in the skyimage file and create a SkyImage object. Note that the BatSkyImage.from_file() method
        # read in the first N hdus in the file where N is the number of energy bins that sky images were created for
        # by default, the partial coding map which is set to append_last is not read in
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
        """
        The file that contains the partial coding image associated with the BatSkyView

        :return: Path
        """
        return self._pcodeimg_file

    @pcodeimg_file.setter
    def pcodeimg_file(self, value):
        # if the user is setting this value, want to make sure that it exists and that we are not containing a mosaic
        # pcode file. Then we reparse all the images. This is inefficient, can just read in the pcode file but that is a
        # future TODO.
        if value is not None:
            temp_value = Path(value).expanduser().resolve()
            if temp_value.exists() and not self.is_mosaic:
                self._pcodeimg_file = temp_value
                self._parse_skyimages()
            else:
                raise ValueError(f"The file {temp_value} does not exist")
        else:
            self._pcodeimg_file = value

    @property
    def projection(self):
        """
        The type of projection that should be used when added to another BatSkyImage. Can be None of "healpix"

        :return: str
        """
        return self._projection

    @projection.setter
    def projection(self, value):
        # this helps set the addition of the sky images and creation of the mosaiced images. None means that
        # we will be using the "traditional" sky facets, while "healpix" means that we will be projecting everything
        # onto the healpix map
        if value is not None and "healpix" not in value:
            raise ValueError("The projection attribute can only be set to None or healpix.")

        self._projection = value

    @property
    def healpix_nside(self):
        """
        When projection is set to "healpix" this defines the healpix map resolution for the addition of the sky views
        once they are projected onto the healpix map.

        :return: int
        """
        return self._healpix_nside

    @healpix_nside.setter
    def healpix_nside(self, value):
        # need to make sure that the projection is actually set to healpix for this attribute to mean anything
        # when it is set, it defines the healpix map resolution that the sky images will be projected to and then added
        if self.projection is not None and "healpix" not in self.projection:
            raise ValueError("The projection attribute needs to be set to healpix before healpix_nside can be set.")

        self._healpix_nside = value

    @property
    def healpix_coordsys(self):
        """
        When projection is set to "healpix" this defines the healpix map's coordinate system that will be used to add
        the BatSkyView objects. This can be set to "galactic" or "icrs"

        :return: str
        """
        return self._healpix_coordsys

    @healpix_coordsys.setter
    def healpix_coordsys(self, value):
        # need to make sure that the projection is actually set to healpix for this attribute to mean anything
        # when this is set, it will define the coordinate system of the healpix map
        if self.projection is not None and "healpix" not in self.projection:
            raise ValueError("The projection attribute needs to be set to healpix before healpix_coordsys can be set.")

        self._healpix_coordsys = value

    @classmethod
    def from_file(cls, skyimg_file, pcodeimg_file=None):
        """
        This class method allows a user to load in a sky flux map and an associated partial coding image.

        :param skyimg_file: Path to the flux sky image that will be loaded in
        :param pcodeimg_file: None or a Path to the partial coding image associated with the flux sky image file
        :return: BatSkyView object
        """

        skyview = cls(skyimg_file=skyimg_file)
        skyview.pcodeimg_file = pcodeimg_file

        return skyview

    def detect_sources(self, catalog_file=None, input_dict=None):
        """
        This method allows for the detection of any source in the BatSkyView, using batcelldetect.

        There must be a partial coding image file specified via the pcodeimg_file attribute in order to detect sources.
        An output catalog of all the sources and their information including counts, SNR, etc are created.

        TODO: add support for healpix/mosaic projections of sky images

        :param catalog_file: None or a Path pointing to a input catalog file that will be passed to batcelldetect to
            identify any known or unknown sources. None will lead to the method using the survey catalog file that is
            included in the BatAnalysis python package.
        :param input_dict: None or a dictionary or key/value pairs that will be passed to batcelldetect. If a value of
            None is passed in the following dictionary will be passed to batcelldetect:
                self.src_detect_input_dict["outfile"] = self.skyimg_file.parent.joinpath(f"{self.skyimg_file.stem}.cat")
                self.src_detect_input_dict["regionfile"] = self.skyimg_file.parent.joinpath(f"{self.skyimg_file.stem}.reg")
                self.src_detect_input_dict["infile"] = str(self.skyimg_file)
                self.src_detect_input_dict["incatalog"] = str(catalog_file)
                self.src_detect_input_dict["pcodefile"] = str(self.pcodeimg_file)
                self.src_detect_input_dict["snrthresh"] = 6
                self.src_detect_input_dict["pcodethresh"] = 0.05
                self.src_detect_input_dict["vectorflux"] = "YES"

        :return: None
        """

        if self.is_mosaic and "HPX" in self.sky_img.axes.labels:
            raise ValueError("Cannot run batcelldetect on healpix sky images.")

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

    @property
    def sky_img(self):
        """
        The flux sky image. If the BatSkyView contains a mosaic sky view, this value will be calculated on the fly.

        TODO: calculate this value once, if is_mosaic==True so if the user accesses this property many times we dont
            waste time/memory recalculating.

        :return: BatSkyImage
        """
        if self.is_mosaic:
            return BatSkyImage(image_data=self.interim_sky_img / self.interim_var_img, image_type="flux")
        else:
            return self._sky_img

    @sky_img.setter
    def sky_img(self, value):
        self._sky_img = value

    @property
    def bkg_stddev_img(self):
        """
        The background standard deviation sky image. If the BatSkyView contains a mosaic sky view, this value will be calculated on the fly.

        TODO: calculate this value once, if is_mosaic==True so if the user accesses this property many times we dont
            waste time/memory recalculating.

        :return: BatSkyImage
        """

        if self.is_mosaic:
            return BatSkyImage(
                image_data=Histogram(self.interim_var_img.axes, contents=1 / np.sqrt(self.interim_var_img),
                                     unit=np.sqrt(1 / self.interim_var_img.unit).unit), image_type="stddev")
        else:
            return self._bkg_stddev_img

    @bkg_stddev_img.setter
    def bkg_stddev_img(self, value):
        self._bkg_stddev_img = value

    @property
    def snr_img(self):
        """
        The SNR sky image. If the BatSkyView contains a mosaic sky view, this value will be calculated on the fly.

        TODO: calculate this value once, if is_mosaic==True so if the user accesses this property many times we dont
            waste time/memory recalculating.

        :return: BatSkyImage
        """

        if self.is_mosaic:
            return BatSkyImage(image_data=self.sky_img / self.bkg_stddev_img, image_type="snr")
        else:
            return self._snr_img

    @snr_img.setter
    def snr_img(self, value):
        self._snr_img = value

    def __add__(self, other):
        """
        If we are adding 2 skyviews we can either do
            1) a "simple" add if we want the healpix projection. Here we take the
        partial coding, and variance weighting into account.
            2) a reprojection onto the skyfacets taking the partial coding, the variance weighting, the off-axis
                corrections into account

        We verified the mosaicing by doing the mosaicing over the preslew time period and comparing the healpix SNR to
        that of the normal DPI and sky image for the preconstructed preslew image projected into a healpix map. We found
        approximately the same value. When we increase the mosaiced healpix_nsides to 512, we get the SNR of the mosaic to be
        about the same as the sky image from the dpi, without it being projected to healpix.

        :param other: BatSkyView
        :return: BaSkyView
        """
        # make sure that we are adding 2 skyviews
        if not isinstance(other, type(self)):
            return NotImplemented

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
                nsides = int(np.nanmax(np.array(nsides, dtype=np.float64)))

            # try to choose between a healpix projection's coord sys  incase there are 2 different one that are
            # specified. By default use the one that the potential mosaic image uses
            is_mosaic = [i.is_mosaic for i in [self, other]]
            if np.any(is_mosaic):
                # id the mosaic's healpix coord sys
                if is_mosaic[0]:
                    coordsys = self.healpix_coordsys
                else:
                    coordsys = other.healpix_coordsys
            else:
                # choose the first one, seof
                coordsys = self.healpix_coordsys

            # now create the histograms that we need to do the calculations, need flux and std dev to be in units of cts/s
            exposure = []
            tstart = []
            tstop = []
            for i, count in zip([self, other], range(2)):
                exposure.append(i.sky_img.exposure)
                tstart.append(i.sky_img.tbins["TIME_START"])
                tstop.append(i.sky_img.tbins["TIME_STOP"])

                # do the healpix projection calculation and get rid of time axis since it is irrelevant now
                # also the time axis prevents us from directly adding 2 histograms together if they dont have the
                # same timebins.

                # if we have a BatSkyView object that is a mosaic image already, dont need to do the sky_img flux and
                # all the associated inverse variance weighting calculations all over again. Can just get those images
                # via the appropriate attributes.
                if not i.is_mosaic:
                    flux_hist = i.sky_img.healpix_projection(coordsys=coordsys,
                                                             nside=nsides).project("HPX", "ENERGY")
                    pcode_hist = i.pcode_img.healpix_projection(coordsys=coordsys,
                                                                nside=nsides).project("HPX", "ENERGY")
                    bkg_stddev_hist = i.bkg_stddev_img.healpix_projection(coordsys=coordsys,
                                                                          nside=nsides).project("HPX",
                                                                                                "ENERGY")

                    exposure_hist = Histogram(flux_hist.axes,
                                              contents=flux_hist.contents.value * 0 + i.sky_img.exposure.value,
                                              unit=i.sky_img.exposure.unit)

                    # correct the units
                    if flux_hist.unit != u.count / u.s:
                        flux_hist /= i.sky_img.exposure
                    if bkg_stddev_hist.unit != u.count / u.s:
                        bkg_stddev_hist /= i.sky_img.exposure

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

                # testing for the energy integrated mosaicing
                # total_e_energybin_ax = Axis(
                #    u.Quantity([self.sky_img.axes["ENERGY"].edges[0], self.sky_img.axes["ENERGY"].edges[-1]]),
                #    label="ENERGY")
                # ax = [self.sky_img.axes["TIME"], bkg_stddev_hist.axes["HPX"], total_e_energybin_ax]
                # total_e_flux = Histogram(edges=ax,
                #                         contents=flux_hist.project("TIME", "HPX").contents[:, :, np.newaxis].value,
                #                         unit=flux_hist.unit)
                # total_e_bkg_stddev = Histogram(edges=ax, contents=bkg_stddev_hist.project("TIME", "HPX").contents[:, :,
                #                                                  np.newaxis].value, unit=bkg_stddev_hist.unit)

                # total_e_energy_quality_mask = np.zeros_like(
                #    total_e_flux.contents.value)
                # total_e_good_idx = np.where(
                #    (pcode_hist.contents[:, :, 0:1] > _pcodethresh
                #     )
                #    & (total_e_bkg_stddev.contents > 0)
                #    & np.isfinite(total_e_flux.contents)
                #    & np.isfinite(total_e_bkg_stddev.contents)
                # )
                # total_e_energy_quality_mask[total_e_good_idx] = 1

                # make the intermediate images to do operations with or just get the appropriate images if we have them
                if count == 0:
                    if not i.is_mosaic:
                        tot_exposure_hist = deepcopy(exposure_hist) * energy_quality_mask
                        interim_pcode_hist = deepcopy(pcode_hist) * tot_exposure_hist.contents.value
                        interm_inv_var_hist = (1 / (bkg_stddev_hist * bkg_stddev_hist)) * energy_quality_mask
                        interm_flux_hist = flux_hist * interm_inv_var_hist
                        # testing for the energy integrated mosaicing
                        # total_e_interim_inv_var_hist = (1 / (
                        #        total_e_bkg_stddev * total_e_bkg_stddev)) * total_e_energy_quality_mask
                        # total_e_interim_flux_hist = total_e_flux * total_e_interim_inv_var_hist
                    else:
                        tot_exposure_hist = i.exposure_img.project("HPX", "ENERGY")
                        interim_pcode_hist = i.pcode_img.project("HPX", "ENERGY")
                        interm_inv_var_hist = i.interim_var_img.project("HPX", "ENERGY")
                        interm_flux_hist = i.interim_sky_img.project("HPX", "ENERGY")

                else:
                    # if we dont have a mosaic sky view object where we calculated exposure_hist, etc above then execute
                    # this code block to do mosaicing. otherwise just call the relevant attributes
                    if not i.is_mosaic:
                        tot_exposure_hist += deepcopy(exposure_hist) * energy_quality_mask
                        interim_pcode_hist += deepcopy(pcode_hist) * tot_exposure_hist.contents.value
                        temp_interm_inv_var_hist = (1 / (bkg_stddev_hist * bkg_stddev_hist)) * energy_quality_mask
                        interm_inv_var_hist += temp_interm_inv_var_hist
                        interm_flux_hist += flux_hist * temp_interm_inv_var_hist

                        # testing for the energy integrated mosaicing
                        # temp_total_e_interim_inv_var_hist = (1 / (
                        #        total_e_bkg_stddev * total_e_bkg_stddev)) * total_e_energy_quality_mask
                        # total_e_interim_flux_hist += total_e_flux * temp_total_e_interim_inv_var_hist
                        # total_e_interim_inv_var_hist += temp_total_e_interim_inv_var_hist
                    else:
                        tot_exposure_hist += i.exposure_img.project("HPX", "ENERGY")
                        interim_pcode_hist += i.pcode_img.project("HPX", "ENERGY")
                        interm_inv_var_hist += i.interim_var_img.project("HPX", "ENERGY")
                        interm_flux_hist += i.interim_sky_img.project("HPX", "ENERGY")

            tmin = u.Quantity(tstart).min()
            tmax = u.Quantity(tstop).max()
            energybin_ax = self.sky_img.axes["ENERGY"]
            hp_ax = HealpixAxis(nside=nsides, coordsys=coordsys, label="HPX")
            t_ax = Axis(u.Quantity([tmin, tmax]), label="TIME")

            # create the SkyImages for each quantity
            tot_exposure = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=tot_exposure_hist.contents[np.newaxis],
                                     unit=tot_exposure_hist.unit), image_type="exposure")
            pcode = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=interim_pcode_hist.contents[np.newaxis],
                                     unit=interim_pcode_hist.unit), image_type="pcode")

            interm_flux = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=interm_flux_hist.contents[np.newaxis],
                                     unit=interm_flux_hist.unit), is_mosaic_intermediate=True, image_type="flux")

            # set the image type to None as we can still do projections normally without any additional considerations
            interm_inv_var = BatSkyImage(
                image_data=Histogram([t_ax, hp_ax, energybin_ax], contents=interm_inv_var_hist.contents[np.newaxis],
                                     unit=interm_inv_var_hist.unit), is_mosaic_intermediate=True, image_type=None)

            test_mosaic = BatSkyView(interim_sky_img=interm_flux, interim_var_img=interm_inv_var, pcode_img=pcode,
                                     exposure_img=tot_exposure)

            # make sure that these attributes are set correctly for the mosaic skyview
            test_mosaic.healpix_nside = nsides
            test_mosaic.healpix_coordsys = coordsys

            # testing for the energy integrated mosaicing
            # total_e_interm_flux = BatSkyImage(
            #    image_data=Histogram([t_ax, hp_ax, total_e_energybin_ax], contents=total_e_interim_flux_hist.contents,
            #                         unit=total_e_interim_flux_hist.unit), is_mosaic_intermediate=True,
            #    image_type="flux")
            # total_e_interm_inv_var = BatSkyImage(
            #    image_data=Histogram([t_ax, hp_ax, total_e_energybin_ax],
            #                         contents=total_e_interim_inv_var_hist.contents,
            #                         unit=total_e_interim_inv_var_hist.unit), is_mosaic_intermediate=True,
            #    image_type=None)

            # test_total_e_mosaic = BatSkyView(interim_sky_img=total_e_interm_flux,
            #                                 interim_var_img=total_e_interm_inv_var, pcode_img=pcode,
            #                                 exposure_img=tot_exposure)

            return test_mosaic  # , test_total_e_mosaic
        else:
            raise NotImplementedError("Adding Sky Images with the template sky facets is not yet implemented.")

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        return self.__add__(other)
