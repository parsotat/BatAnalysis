import types
import unittest
import warnings
from unittest.mock import patch

import astropy.units as u
import numpy as np
from histpy import Axis, HealpixAxis, Histogram

from batanalysis.mosaic import _pcodethresh
from batanalysis.bat_skyimage import BatSkyImage
from batanalysis.bat_skyview import BatSkyView


class MosaicLowPcodeDetectionTests(unittest.TestCase):
    def test_mosaic_partial_coding_floor_is_one_percent(self):
        self.assertAlmostEqual(_pcodethresh, 0.01)

    def test_healpix_image_without_wcs_does_not_emit_detector_plane_warning(self):
        t_ax = Axis([0, 1] * u.s, label="TIME")
        hp_ax = HealpixAxis(nside=1, coordsys="galactic", label="HPX")
        e_ax = Axis([15, 350] * u.keV, label="ENERGY")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BatSkyImage(
                Histogram([t_ax, hp_ax, e_ax], contents=np.zeros((1, hp_ax.nbins, 1)), unit=u.count),
                image_type="flux",
            )

        warning_messages = [str(item.message) for item in caught]
        self.assertFalse(
            any("No astropy World Coordinate System has been specified" in message for message in warning_messages)
        )

    def test_mosaic_detection_uses_normalized_partial_coding(self):
        t_ax = Axis([0, 1] * u.s, label="TIME")
        hp_ax = HealpixAxis(nside=1, coordsys="galactic", label="HPX")
        e_ax = Axis([15, 350] * u.keV, label="ENERGY")
        shape = (1, hp_ax.nbins, 1)

        interim_flux = np.zeros(shape)
        interim_var = np.ones(shape)
        weighted_pcode = np.zeros(shape)
        exposure = np.ones(shape)

        weighted_pcode[0, 0, 0] = 0.02 * 0.2
        exposure[0, 0, 0] = 0.2
        interim_flux[0, 0, 0] = 6.0

        flux_img = BatSkyImage(
            Histogram([t_ax, hp_ax, e_ax], contents=interim_flux, unit=u.count),
            is_mosaic_intermediate=True,
            image_type="flux",
        )
        var_img = BatSkyImage(
            Histogram([t_ax, hp_ax, e_ax], contents=interim_var, unit=1 / (u.count ** 2)),
            is_mosaic_intermediate=True,
            image_type=None,
        )
        pcode_img = BatSkyImage(
            Histogram([t_ax, hp_ax, e_ax], contents=weighted_pcode, unit=u.s),
            image_type="pcode",
        )
        exposure_img = BatSkyImage(
            Histogram([t_ax, hp_ax, e_ax], contents=exposure, unit=u.s),
            image_type="exposure",
        )

        skyview = BatSkyView(
            interim_sky_img=flux_img,
            interim_var_img=var_img,
            pcode_img=pcode_img,
            exposure_img=exposure_img,
        )

        fake_hsp_core = types.SimpleNamespace(
            HSPTask=lambda name: types.SimpleNamespace(default_params={}, taskname=name)
        )

        with patch("batanalysis.bat_skyview.hsp_core", fake_hsp_core, create=True):
            normalized_pcode = skyview._normalized_detection_pcode_img().contents[0, 0, 0]
            result = skyview.detect_sources(input_dict=dict(snrthresh=5.0, pcodethresh=_pcodethresh))

        self.assertAlmostEqual(normalized_pcode, 0.02)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()