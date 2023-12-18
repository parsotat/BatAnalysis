"""
This file contains the batobservation class which contains information pertaining to a given bat observation.

Tyler Parsotan Jan 24 2022
"""

from .batlib import datadir
from pathlib import Path


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
            if obs_dir.joinpath(self.obs_id).is_dir():
                self.obs_dir = obs_dir.joinpath(
                    self.obs_id
                )
            else:
                raise FileNotFoundError(
                    "The directory %s does not contain the observation data corresponding to ID: %s"
                    % (obs_dir, self.obs_id)
                )
        else:
            obs_dir = datadir()  # Path.cwd()

            if obs_dir.joinpath(self.obs_id).is_dir():
                self.obs_dir = obs_dir.joinpath(
                    self.obs_id
                )
            else:
                raise FileNotFoundError(
                    "The directory %s does not contain the observation data correponding to ID: %s"
                    % (obs_dir, self.obs_id)
                )
