"""
This file holds the BatSkyView object which contains all the necessary sky map information that can be generated from
batfftimage (flux sky image, background variation map, partial coding map).

Tyler Parsotan May 15 2024
"""
import warnings
from pathlib import Path

from astropy.io import fits

from .bat_skyimage import BatSkyImage

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
            dpi_file=None,
            detector_quality_file=None,
            attitude_file=None,
            dpi_data=None,
            input_dict=None,
            recalc=False,
            load_dir=None,
            create_pcode_img=True,
            create_snr_img=False,
            create_bkg_stddev_img=False
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
            self.skyimg_file = dpi_file.parent.joinpath(f"test_{dpi_file.stem}.img")

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
            if not self.skyimg_file.exists() or recalc:
                raise ValueError("Please specify a DPI file to create the sky image from.")

        if detector_quality_file is not None:
            self.detector_quality_file = Path(detector_quality_file).expanduser().resolve()
            if not self.detector_quality_file.exists():
                raise ValueError(
                    f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                    f"to exist. Please double check that it does.")
        else:
            self.detector_quality_file = "NONE"
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
                self.pcodeimg_file = self.skyimg_file.parent.joinpath(
                    f"{dpi_file.stem}.pcodeimg")
                pcodeimg_input_dict = self.skyimg_input_dict.copy()
                pcodeimg_input_dict["pcodemap"] = "YES"
                pcodeimg_input_dict["outfile"] = str(self.pcodeimg_file)

                batfftimage_pcode_result = self._call_batfftimage(pcodeimg_input_dict)

                # make sure that this calculation ran successfully
                if batfftimage_pcode_result.returncode != 0:
                    raise RuntimeError(
                        f"The creation of the associated partial coding map failed with message: {batfftimage_pcode_result.output}"
                    )
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

    def _parse_skyimages(self):
        """
        This method goes through the sky image file that was produced by batfftimage and reads in all the sky images'
        fits files and saves them as BatSkyImage objects to the appropriate attributes

        TODO: batgrbproducts doesnt append the partial coding map to the output, how can users load this file in separately
            if it doesnt show up in the history of the main skymap?
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
        if self.pcodeimg_file is not None and self.pcodeimg_file.exists():
            self.pcode_img = BatSkyImage.from_file(self.pcodeimg_file)
        else:
            self.pcode_img = None

        # see if there are background/snr images for us to read in
        # this can be defined in the history of the batfftimage call or in the constructor method
        # we prioritize anything that was set in the constructor
        if self.snr_img_file is None and self.skyimg_input_dict["signifmap"] != "NONE":
            self.snr_img_file = Path(self.skyimg_input_dict["signifmap"])

        # now read in the file
        if self.snr_img_file is not None:
            self.snr_img = BatSkyImage.from_file(self.snr_img_file)
        else:
            self.snr_img = None

        if self.bkg_stddev_img_file is None and self.skyimg_input_dict["bkgvarmap"] != "NONE":
            self.bkg_stddev_img_file = Path(self.skyimg_input_dict["bkgvarmap"])

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
                raise ValueError("The file {temp_value} does not exist")
        else:
            self._pcodeimg_file = value
