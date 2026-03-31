"""
This file hods the BatAttitude class that reads in a *.sat or .mkf file to access
the attitude data.

Tyler Parsotan July 2024
"""
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits


class Attitude(object):
    """
    This class encapsulates the Swift attitude data contained in a *.sat or *.mkf file that is obtained by Swift

    TODO: add methods to add/concatenate attitude data etc
    """

    def __init__(self, time, ra, dec, roll, quarternion=None, is_10arcmin_settled=None, is_settled=None,
                 in_saa=None,
                 in_safehold=None):
        """
        Itialize something
        """

        self.time = time
        self.ra = ra
        self.dec = dec
        self.roll = roll

        self.quarternions = quarternion

        self.is_10arcmin_settled = is_10arcmin_settled
        self.is_settled = is_settled
        self.in_saa = in_saa
        self.in_safehold = in_safehold

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

            # quarternions have scalar last, we may need to multiply by -1 to get it to fix the scipy/gdt convention
            quarternions = all_data["QPARAM"]

            is_10arcmin_settled = all_data["FLAGS"][:, 0]
            is_settled = all_data["FLAGS"][:, 1]
            in_saa = all_data["FLAGS"][:, 2]
            in_safehold = all_data["FLAGS"][:, 3]

        elif "mkf" in str(attitude_file):

            time = all_data["TIME"]
            ra = all_data["RA"]
            dec = all_data["DEC"]
            roll = all_data["ROLL"]

            quarternions = None

            is_10arcmin_settled = all_data["TEN_ARCMIN"]
            is_settled = all_data["SETTLED"]
            in_saa = all_data["ACS_SAA"]
            in_safehold = all_data["SAFEHOLD"]

        else:
            raise ValueError("This attitude file is not recognized. Please pass in a *.sat or *.mkf file.")

        return cls(time=time, ra=ra, dec=dec,
                   roll=roll, is_10arcmin_settled=is_10arcmin_settled, is_settled=is_settled, in_saa=in_saa,
                   in_safehold=in_safehold)

    def plot(self, T0=None):
        """
        Plot the ra/dec/roll of the attitude.

        :return:
        """

        if T0 is None:
            t_rel = self.time.min()
        else:
            if type(T0) is not u.Quantity:
                t_rel = T0 * u.s

        fig, ax = plt.subplots(1, sharex=True)

        plt.plot(self.time - t_rel, self.ra, label="RA")
        plt.plot(self.time - t_rel, self.dec, label="DEC")
        plt.plot(self.time - t_rel, self.roll, label="ROLL")

        if T0 is not None:
            plt.axvline(0, ls='--')

        plt.legend()
        plt.xlabel(f"MET-{t_rel}")
        plt.ylabel(f"Pointing ({self.ra.unit})")

#TODO: implement merging attitude files and loading the merged file into a Attitude object
"""
from astropy.table import Table, vstack
import os

# Create dummy FITS files for demonstration purposes if needed, 
# or assume 'table1.fits' and 'table2.fits' exist.

# 1. Read the FITS tables into astropy Table objects
table1 = Table.read('table1.fits', format='fits')
table2 = Table.read('table2.fits', format='fits')

# 2. Vertically stack the tables
# This appends rows of table2 to table1.
combined_table = vstack([table1, table2])

# 3. Remove duplicate rows based on a specific column 
unique_table = combined_table.unique(keys=['TIME'])

# 4. Write the unique table to a new FITS file
output_filename = 'merged_unique_catalog.fits'
unique_table.write(output_filename, format='fits', overwrite=True)

print(f"Merged tables and saved unique rows to {output_filename}")

trigtime=Time("2021-06-05T14:55:58.000")

guano=GUANO(triggertime=trigtime.datetime)

afst_obs=ObsQuery(begin=trigtime-TimeDelta(0,format="sec"), end=trigtime+TimeDelta(60*60,format="sec"))

if guano.entries[0].obsid != afst_obs[0].obsid:
    #then download
    Data(afst_obs[0].obsid, auxil=True)

    #then merge the attitudes using the pseudo code above, making sure to save the new file in the guano.entries[0].obsid auxil directory
"""