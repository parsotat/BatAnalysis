"""
This file hods the BatAttitude class that reads in a *.sat or .mkf file to access
the attitude data.

Tyler Parsotan July 2024
"""
from pathlib import Path

import astropy.units as u
from astropy.io import fits


class Attitude(object):
    """
    This class encapsulates the Swift attitude data contained in a *.sat or *.mkf file that is obtained by Swift

    TODO: add methods to add/concatenate event data, plot event data, etc
    """

    def __init__(self):
        """
        Itialize something
        """
        return None

    @classmethod
    def from_file(cls, attitude_file):
        attitude_file = Path(attitude_file).expanduser().resolve()

        if not attitude_file.exists():
            raise ValueError(f"The attitude file passed in to be read {attitude_file} does not seem to exist.")

        # iteratively read in the data with units
        all_data = {}
        with fits.open(attitude_file) as file:
            data = file[1].data
            for i in data.columns:
                all_data[i.name] = u.Quantity(data[i.name], i.unit)
