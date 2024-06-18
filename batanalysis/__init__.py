import os

__all__ = ["batobservation", "bat_survey", "bat_tte", "batlib", "plotting", "mosaic"]

# first want to check if caldb is initalized/installed
# checks for heasoft is contained in each .py file of the package
if os.environ["CALDB"] is None:
    raise EnvironmentError('CALDB does not seem to be initialized/installed. '
                           'BatAnalysis cannot be imported without this.')


from ._version import __version__

from .batobservation import *
from .bat_survey import *
from .bat_tte import *
from .batlib import *
from .plotting import *
from .mosaic import *
from . import parallel
