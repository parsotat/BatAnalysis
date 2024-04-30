"""
This file holds the BatSkyImage class which contains binned data from a skymap generated

Tyler Parsotan March 11 2024
"""

from histpy import Histogram

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
            bkg_dpi_file=None
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
                raise ValueError(f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                                 f"to exist. Please double check that it does.")
        else:
            self.detector_quality_file = None
            # warnings.warn("No detector quality mask file has been specified. The resulting DPI object "
            #              "will not be able to be modified either by rebinning in energy or time.", stacklevel=2)
