"""
This file contains the batobservation class which contains information pertaining to a given bat observation.

Tyler Parsotan Jan 24 2022
"""

from .batlib import datadir
from pathlib import Path
from astropy.time import Time
from datetime import datetime, timedelta
import re
import warnings

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)

#try:
    #import xspec as xsp
#except ModuleNotFoundError as err:
    # Error handling
    #print(err)



class BatObservation(object):
    """
    A general Bat Observation object that holds information about the observation ID and the directory of the
    observation ID. This class ensures that the observation ID directory exists and throws an error if it does not.
    """
    def __init__(self, obs_id, obs_dir=None):
        """
        Constructor for the BatObservation object.

        :param obs_id: string of the observation id number
        :param obs_dir: string of the directory that the observation id folder resides within
        """

        self.obs_id = str(obs_id)
        if obs_dir is not None:
            obs_dir = Path(obs_dir).expanduser().resolve()
            # the use has provided a directory to where the bat observation id folder is kept
            # test to see if the folder exists there
            if  obs_dir.joinpath(self.obs_id).is_dir():
                self.obs_dir = obs_dir.joinpath(self.obs_id) # os.path.join(obs_dir , self.obs_id)
            else:
                raise FileNotFoundError(
                    'The directory %s does not contain the observation data corresponding to ID: %s' % (obs_dir, self.obs_id))
        else:
            obs_dir = datadir()  #Path.cwd()

            if obs_dir.joinpath(self.obs_id).is_dir():
                #os.path.isdir(os.path.join(obs_dir , self.obs_id)):
                self.obs_dir = obs_dir.joinpath(self.obs_id) #self.obs_dir = os.path.join(obs_dir , self.obs_id)
            else:
                raise FileNotFoundError('The directory %s does not contain the observation data correponding to ID: %s' % (obs_dir, self.obs_id))

    def _set_local_pfile_dir(self, dir):
        """
        make the local pfile dir if it doesnt exist and set this value

        :return: None
        """
        #make sure that it is a Path object
        self._local_pfile_dir=Path(dir)

        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp.utils.local_pfiles(par_dir=str(self._local_pfile_dir))

    def _get_local_pfile_dir(self):
        """
        Return the _local_pfile_dir attribute

        :return: Returns the _local_pfile_dir Path object
        """

        return self._local_pfile_dir


    def _call_bathotpix(self, input_dict):
        """
        Calls heasoftpy's bathotpix with an error wrapper

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's bathotpix
        :return: heasoftpy Result object from bathotpix
        """

        # directly calls bathotpix
        try:
            return hsp.bathotpix(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft bathotpix failed with inputs: {input_dict}.")

    def _call_batbinevent(self, input_dict):
        """
        Calls heasoftpy's batbinevt with an error wrapper

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batbinevt
        :return: heasoftpy Result object from batbinevt
        """
        # directly calls bathotpix
        try:
            return hsp.batbinevt(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batbinevt failed with inputs {input_dict}.")

    def _call_batmaskwtevt(self, input_dict):
        """
        Calls heasoftpy's batmaskwtevt with an error wrapper

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batmaskwtevt
        :return: heasoftpy Result object from batmaskwtevt
        """
        # directly calls bathotpix
        try:
            return hsp.batmaskwtevt(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batmaskwtevt failed with inputs {input_dict}.")

    def _call_bateconvert(self, input_dict):
        """
        Calls heasoftpy's bateconvert with an error wrapper

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's bateconvert
        :return: heasoftpy Result object from bateconvert
        """
        # directly calls bateconvert
        try:
            return hsp.bateconvert(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batmaskwtevt failed with inputs {input_dict}.")



class Lightcurve(object):
    """
    This is a general light curve class that contains typical information that a user may want from their lightcurve.
    This object is a wrapper around a light curve created from BAT event data.
    """

    def __init__(self, eventfile,  ):
        """
        This constructor reads in a fits file that contains light curve data for a given BAT event dataset. The fits file
        should have been created by a call to

        :param lightcurve_file:
        """


    def set_timebins(self):
        """
        This method allows for the dynamic rebinning of a light curve in time.

        :return:
        """

    def set_energy_bins(self):
        """
        This method allows for the dynamic rebinning of a light curve in energy

        :return:
        """

    def plot(self):
        """
        This convenience funciton plots the light curve for the specified timebinning and/or energy binning

        :return:
        """


