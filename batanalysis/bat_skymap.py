"""
This file holds the BatSkyImage class which contains binned data from a skymap generated

Tyler Parsotan March 11 2024
"""

import gzip
import shutil
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from .bat_dpi import BatDPI
from .batlib import create_gti_file
from .detectorplanehist import DetectorPlaneHistogram

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)

class BatSkyImage(DetectorPlaneHistogram):
