"""
This file holds the BatDRM class which hold the detector response matrix at a given time.

Tyler Parsotan Sept 3 2024

"""
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from histpy import Histogram

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


class BatDRM(Histogram):

    @u.quantity_input(
        timebins=["time"],
        tmin=["time"],
        tmax=["time"],
        input_energybins=["energy"],
        input_emin=["energy"],
        input_emax=["energy"],
        output_energybins=["energy"],
        output_emin=["energy"],
        output_emax=["energy"],
    )
    def __init__(self,
                 drm_data,
                 timebins=None,
                 tmin=None,
                 tmax=None,
                 input_energybins=None,
                 input_emin=None,
                 input_emax=None,
                 output_energybins=None,
                 output_emin=None,
                 output_emax=None,
                 ):

        """"
        This constructor can either:
            3) create a BatDRM object based on an input DRM numpy array or a Histogram object

        """

    @staticmethod
    def calc_drm(pha_file):
        """
        This calls heasoftpy's batdrmgen which produces the associated drm for fitting the PHA file.

        :param pha_file: a list of PHA path objects or a Path object to the PHA file that the DRM will be constructed for
        :return: Path object to the created DRM file or a list of Path objects to all the DRM files created
        """

        if type(pha_file) is not list:
            pha_file = [pha_file]

        # when passing in tht whole filename, the paths mess up the connection between the response file and the pha file
        # since there seems to be some character limit to this header value. Therefore, we need to cd to the directory
        # that the PHA file lives in and create the .rsp file and then cd back to the original location.

        # make sure that all elements are paths
        phafilename = [Path(i) for i in pha_file]

        # we are passing in a whole filepath or
        # we are already located in the PHA directory and are mabe calculating the upperlimit bkg spectrum

        # Check if the phafilename is a string and if it has an extension .pha. If NOT then exit
        # drm_file=[]
        for filename in phafilename:
            if ".pha" not in filename.name:
                raise ValueError(
                    f"The file name {filename} needs to be a string and must have an extension of .pha ."
                )

            # get the cwd
            current_dir = Path.cwd()

            # get the directory that we have to cd to and the name of the file
            pha_dir = filename.parent
            pha_file = filename.name

            # cd to that dir
            if str(pha_dir) != str(current_dir):
                os.chdir(pha_dir)

            # Split the filename by extension, so as to remove the .pha and replace it with .rsp
            # this is necessary since sources can have '.' in name
            out = filename.stem + ".rsp"

            # create drm
            output = hsp.batdrmgen(
                infile=pha_file, outfile=out, chatter=2, clobber="YES", hkfile="NONE"
            )

            if output.returncode != 0:
                raise RuntimeError(f"The call to Heasoft batdrmgen failed with output {output.stdout}.")

            # cd back
            if str(pha_dir) != str(current_dir):
                os.chdir(current_dir)

        drm_file = [i.parent.joinpath(f"{pha_file.stem}.rsp") for i in pha_file]

        if len(drm_file) > 1:
            return drm_file
        else:
            return drm_file[0]

    @classmethod
    def from_file(cls, pha_file=None, drm_file=None):
        """
        This class method takes either a pha file of a drm file and either:
            1) create a drm file based on an input PHA file and load that into a BatDRM object
            2) create a BatDRM object from a preconstructed BatDRM file

        :param pha_file:
        :param drm_file:
        :return:
        """

        # make sure something is specified
        if pha_file is None and drm_file is None:
            raise ValueError("Either a Path object specifying a PHA file or a Path object specifying a DRM file needs"
                             " to be passed in.")

        # maks sure only 1 file is specified and do error checking
        if pha_file is not None and drm_file is not None:
            raise ValueError("Please only specify either a pha file or a drm file.")

        if pha_file is not None:
            if not isinstance(pha_file, Path):
                raise ValueError("The pha_file that has been passed in needs to be a pathlib Path object.")

            pha_file = Path(pha_file).expanduser().resolve()
            if not pha_file.exists():
                raise ValueError(f"The specified file {pha_file} does not seem to exist. "
                                 f"Please double check that it does.")

        if drm_file is not None:
            if not isinstance(drm_file, Path):
                raise ValueError("The drm_file that has been passed in needs to be a pathlib Path object.")

            drm_file = Path(drm_file).expanduser().resolve()
            if not drm_file.exists():
                raise ValueError(f"The specified file {drm_file} does not seem to exist. "
                                 f"Please double check that it does.")

        # first see if we have a pha file to create a drm for
        if pha_file is not None:
            drm_file = cls.calc_drm(pha_file)

        # if we had to first create a pha file, we still set drm_file so we enter this if statement and parse the file
        # otherwise the user passes in drm_file and we still execute this.
        if drm_file is not None:
            with fits.open(drm_file) as f:
                # get the size of things that we need to save
                n_out_ebins = f["EBOUNDS"].header["NAXIS2"]
                n_in_ebins = f[1].header["NAXIS2"]

                energy_unit = u.Quantity(f'1{f["EBOUNDS"].header["TUNIT2"]}')
                time_unit = u.Quantity(f'1{f[1].header["TIMEUNIT"]}')

                timebin = [f[1].header["TSTART"], f[1].header["TSTOP"]] * time_unit

                # get the output energy bin edges, therefore need the +1
                out_ebins = np.zeros(n_out_ebins + 1)
                in_ebins = np.zeros(n_in_ebins + 1)

                # create the arrays
                out_ebins[:-1] = f["EBOUNDS"].data["E_MIN"]
                out_ebins[-1] = f["EBOUNDS"].data["E_MAX"][-1]

                in_ebins[:-1] = f[1].data["ENERG_LO"]
                in_ebins[-1] = f[1].data["ENERG_HI"][-1]

                # get the full response matrix
                rsp = np.zeros((n_in_ebins, n_out_ebins))

                for count, mat in enumerate(f[1].data["MATRIX"]):
                    rsp[count, :] = mat

        return cls(drm_data=mat, input_energybins=in_ebins * energy_unit, output_energybins=out_ebins * energy_unit,
                   timebins=timebin)
