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

    TODO: add methods to add/concatenate attitude data, plot attitude data, etc
    """

    def __init__(self, time, ra, dec, roll, acs_flags):
        """
        Itialize something
        """

        self.time = time
        self.ra = ra
        self.dec = dec
        self.roll = roll
        self.acs_flags = acs_flags

    @classmethod
    def from_file(cls, attitude_file):
        attitude_file = Path(attitude_file).expanduser().resolve()

        if not attitude_file.exists():
            raise ValueError(f"The attitude file passed in to be read {attitude_file} does not seem to exist.")

        # iteratively read in the data with units for the *sat and mkf files
        all_data = {}
        with fits.open(attitude_file) as file:
            # read in extension 1 with general data
            data = file[1].data
            for i in data.columns:
                all_data[i.name] = u.Quantity(data[i.name], i.unit)

            if "sat" in str(attitude_file):
                # also read in the 2nd extension to get the ACS flags in teh sar file
                data = file["ACS_DATA"].data
                all_data["FLAGS"] = data["FLAGS"]

        # with the data read in, process it and access the info that we need to create the class
        if "sat" in str(attitude_file):
            time = all_data["TIME"]
            ra = all_data["POINTING"][:, 0]
            dec = all_data["POINTING"][:, 1]
            roll = all_data["POINTING"][:, 2]
            flags = all_data["FLAGS"]

        elif "mkf" in str(attitude_file):

            time = all_data["TIME"]
            ra = all_data["RA"]
            dec = all_data["DEC"]
            roll = all_data["ROLL"]
        else:
            raise ValueError("This attitude file is not recognized. Please pass in a *.sat or *.mkf file.")

        return cls(time=time, ra=ra, dec=dec,
                   roll=roll, acs_flags=flags)
