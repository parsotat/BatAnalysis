import os

__all__ = [
    "batobservation",
    "batproducts",
    "bat_survey",
    "bat_tte",
    "batlib",
    "plotting",
    "mosaic",
]

# first want to check if caldb is initalized/installed
# checks for heasoft is contained in each .py file of the package
try:
    caldb = os.environ["CALDB"]
except KeyError as e:
    raise EnvironmentError('CALDB does not seem to be initialized/installed. '
                           'BatAnalysis cannot be imported without this.')

# can also get None for caldb which is not good
if caldb is None:
    raise EnvironmentError('CALDB does not seem to be initialized/installed. '
                           'BatAnalysis cannot be imported without this.')


from ._version import __version__

from .batobservation import *
from .batproducts import *
from .attitude import *
from .bat_survey import *
from .bat_tte import *
from .bat_drm import *
from .bat_dph import *
from .bat_dpi import *
from .bat_skyimage import *
from .bat_skyview import *
from .batlib import *
from .plotting import *
from .mosaic import *
from . import parallel
