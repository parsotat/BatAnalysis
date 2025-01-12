"""
This file contains the various product-related classes associated with Bat observations

Tyler Parsotan Jan 28 2024
"""
import gzip
import os
import shutil
import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from .bat_drm import BatDRM
from .batlib import met2mjd, met2utc, create_gti_file, calc_response
from .batobservation import BatObservation

# for python>3.6
try:
    import heasoftpy.swift as hsp
    import heasoftpy as hsp_core
except ModuleNotFoundError as err:
    # Error handling
    print(err)

# try:
# import xspec as xsp
# except ModuleNotFoundError as err:
# Error handling
# print(err)

_warn_skips = (os.path.dirname(__file__),)


class Lightcurve(BatObservation):
    """
    This is a general light curve class that contains typical information that a user may want from their lightcurve.
    This object is a wrapper around a light curve created from BAT event data.

    """

    @u.quantity_input(ra=u.deg, dec=u.deg)
    def __init__(self, lightcurve_file, event_file, detector_quality_file, ra=None, dec=None, lc_input_dict=None,
                 recalc=False, mask_weighting=True):
        """
        This constructor either creates a lightcurve fits file based off of a passed in event file where mask weighting
        has been applied and the detector quality mask has been constructed. Alternatively, this method can read in a
        previously calculated lightcurve. If recalc=True, then the lightcurve can be recalculated using the passed in
        lc_input_dict or a default input_dict defined as:

        dict(infile=str(event_file), outfile=str(lightcurve_file), outtype="LC",
                              energybins="15-350", weighted="YES", timedel=0.064,
                              detmask=str(detector_quality_file),
                              tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")

        The ra/dec of the source that this lightcurve was constructed for (and for which the weighting was applied to
        the event file), can be specified or it can be dynamically read from the event file.

        :param lightcurve_file: path object of the lightcurve file that will be read in, if previously calculated,
            or the location/name of the new lightcurve file that will contain the newly calculated lightcurve.
        :param event_file: Path object for the event file with mask weighting already applied, from which we will construct
            the lightcurve or read the previously ocnstructed lightcurve file
        :param detector_quality_file: Path object for the detector quality mask that was constructed for the associated
            event file
        :param ra: None or astropy Quantity representing the decimal degree RA value of the source for which the mask weighting was
            applied to the passed in event file. A value of None indicates that the RA of the source will be obtained
            from the event file which is then saved to lightcurve_file
        :param dec: None or astropy Quantity representing the decimal degree DEC value of the source for which the mask weighting was
            applied to the passed in event file. A value of None indicates that the DEC of the source will be obtained
            from the event file which is then saved to lightcurve_file
        :param lc_input_dict: None or a dict of values that will be passed to batbinevt in the creation of the lightcurve.
            If a lightcurve is being read in from one that was previously created, the prior parameters that were used to
            calculate the lightcurve will be used.
            If lc_input_dict is None, this will be set to:
                dict(infile=str(event_file), outfile=str(lightcurve_file), outtype="LC",
                                  energybins="15-350", weighted="YES", timedel=0.064,
                                  detmask=str(detector_quality_file),
                                  tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")
            See HEASoft docmentation on batbinevt to see what these parameters mean. Alternatively, these defaults can
            be used in the inital call and time/energy rebinning can be done using the set_timebins and set_energybins
            methods associated with the Lightcurve object.
        :param recalc: Boolean to denote if the lightcurve specified by lightcurve_file should be recalculated with the
            lc_input_dict values (either those passed in or those that are defined by default)
        :param mask_weighting: Boolean to denote if mask weighting should be applied. By default this is set to True,
            however if a source is out of the BAT field of view the mask weighting will produce a lightcurve of 0 counts.
            Setting mask_weighting=False in this case ignores the position of the source and allows the pure rates/counts
            to be calculated.
        """
        # NOTE: Lots of similarities here as with the spectrum since we are using batbinevt as the base. If there are any
        # issues with the lightcurve object, then we should make sure that these same problems do not occur in the
        # spectrum object and vice versa

        # save these variables
        self.lightcurve_file = Path(lightcurve_file).expanduser().resolve()

        # if any of these below are None, produce a warning that we wont be able to modify the spectrum. Also do
        # error checking for the files existing, etc
        if event_file is not None:
            self.event_file = Path(event_file).expanduser().resolve()
            if not self.event_file.exists():
                raise ValueError(f"The specified event file {self.event_file} does not seem to exist. "
                                 f"Please double check that it does.")
        else:
            self.event_file = None
            warnings.warn("No event file has been specified. The resulting lightcurve object will not be able "
                          "to be modified either by rebinning in energy or time.", stacklevel=2)

        if detector_quality_file is not None:
            self.detector_quality_file = Path(detector_quality_file).expanduser().resolve()
            if not self.detector_quality_file.exists():
                raise ValueError(f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                                 f"to exist. Please double check that it does.")
        else:
            self.detector_quality_file = None
            warnings.warn("No detector quality mask file has been specified. The resulting lightcurve object "
                          "will not be able to be modified either by rebinning in energy or time.", stacklevel=2)

        # error checking for weighting
        if type(mask_weighting) is not bool:
            raise ValueError("The mask_weighting parameter should be a boolean value.")

        # need to see if we have to construct the lightcurve if the file doesnt exist
        if not self.lightcurve_file.exists() or recalc:
            # see if the input dict is None so we can set these defaults, otherwise save the requested inputs for use later
            if lc_input_dict is None:
                self.lc_input_dict = dict(infile=str(self.event_file), outfile=str(self.lightcurve_file), outtype="LC",
                                          energybins="15-350", weighted="YES", timedel=0.064,
                                          detmask=str(self.detector_quality_file),
                                          tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")

                # specify if we want mask weighting
                if mask_weighting:
                    self.lc_input_dict["weighted"] = "YES"
                else:
                    self.lc_input_dict["weighted"] = "NO"

            else:
                self.lc_input_dict = lc_input_dict

            # create the lightcurve
            self.bat_lc_result = self._call_batbinevt(self.lc_input_dict)

            # make sure that this calculation ran successfully
            if self.bat_lc_result.returncode != 0:
                raise RuntimeError(f'The creation of the lightcurve failed with message: {self.bat_lc_result.output}')

        else:
            # set the self.lc_input_dict = None so the parsing of the lightcurve tries to also load in the
            # parameters passed into batbinevt to create the lightcurve
            # try to parse the existing lightcurve file to see what parameters were passed to batbinevt to construct the file
            self.lc_input_dict = None

        # set default RA/DEC coordinates correcpondent to the LC file which will be filled in later if it is set to None
        self.ra = ra
        self.dec = dec

        # read info from the lightcurve file assume that the lightcurve file is not a rate file. The parsing will
        # determine if it is or not
        self._is_rate_lc = False
        self._parse_lightcurve_file()

        # read in the information about the weights
        self._get_event_weights()

        # set the duration info to None for now until the user calls battblocks
        self.tdurs = None

        # were done getting all the info that we need. From here, the user can rebin the timebins and the energy bins

    @property
    def ra(self):
        """The right ascension of the source and the associated weighting assigned to the event file to produce the lightcurve"""
        return self._ra

    @ra.setter
    @u.quantity_input
    def ra(self, value: u.Quantity[u.deg] | None):
        self._ra = value

    @property
    def dec(self):
        """The declination of the source and the associated weighting assigned to the event file to produce the lightcurve"""
        return self._dec

    @dec.setter
    @u.quantity_input
    def dec(self, value: u.Quantity[u.deg] | None):
        self._dec = value

    @u.quantity_input(timebins=['time'], tmin=['time'], tmax=['time'])
    def set_timebins(self, timebinalg="uniform", timebins=None, tmin=None, tmax=None, T0=None, is_relative=False,
                     timedelta=np.timedelta64(64, 'ms'), snrthresh=None, save_durations=True):
        """
        This method allows for the rebinning of the lightcurve in time. The timebins can be uniform, snr-based,
        custom defined, or based on bayesian blocks (using battblocks). The time binning is done dymaically and the
        information for the rebinned lightcurve is automatically updated in the data attribute (which holds the light
        curve information itself including rates/counts, errors, fracitonal exposure, total counts, etc),
        and the tbins attibute with the time bin edges and the time bin centers.

        :param timebinalg: a string that can be set to "uniform", "snr", "highsnr", or "bayesian"
            "uniform" will do a uniform time binning from the specified tmin to tmax with the size of the bin set by
                the timedelta parameter.
            "snr" will bin the lightcurve until a maximum snr threshold is achieved, as is specified by the snrthresh parameter,
                or the width of the timebin becomes the size of timedelta
            "highsnr" will bin the lightcurve with a minimum bin size specified by the timedelta parameter. Longer
                timebin widths will be used if the source is not deteted at the snr level specified by the snrthresh parameter
            "bayesian" will use the battblocks bayesian algorithm to calculate the timebins based off of the energy
                energy integrated lightcurve with 64 ms time binning. Then the lightcurve will be binned in time to the
                tiembins determined by the battblocks algorithm. Using this option also allows for the calculation of
                T90, T50, background time periods, etc if the save_durations parameter =True (more information can
                be found from the battblocks HEASoft documentation).
            NOTE: more information can be found by looking at the HEASoft documentation for batbinevt and battblocks
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param tmin: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the lightcurve to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.

            If the length of tmin is 1 then this denotes the time when the binned lightcurve should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param tmax: astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the lightcurve to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.

            If the length of tmin is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param T0: float or an astropy.units.Quantity object with some tiem of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tmin/tmax that were passed in.
        :param timedelta: numpy.timedelta64 object denoting the size of the timebinning. This value is used when
            timebinalg is used in the binning algorithm
        :param snrthresh: float representing the snr threshold associated with the timebinalg="snr" or timebinalg="highsnr"
            parameter values. See above description of the timebinalg parameter to see how this snrthresh parameter is used.
        :param save_durations: Boolean switch denoting if the T90, T50, and other durations calculated by the battblocks
            algorithm should be saved. If they are, this information will be located in the tdurs attribute. This calculation
            is only possible if timebinalg="bayesian". (More information can be found from the battblocks HEASoft documentation)
        :return: None
        """

        # make sure the user isnt trying to rebin a rate file
        if self._is_rate_lc:
            raise RuntimeError("The rate light curves cannot be rebinned in energy")

        # make sure we have all the info to do the rebinning
        if self.event_file is None or self.detector_quality_file is None:
            raise RuntimeError("The lightcurve cannot be rebinned in time since one of the following files was not "
                               "initalized with this Lightcurve object: the event file "
                               "or the detector quality mask file.")

        # create a temp copy incase the time rebinning doesnt complete successfully
        tmp_lc_input_dict = self.lc_input_dict.copy()

        # create a copy of the timebins if it is not None to prevent modifying the original array
        if timebins is not None:
            timebins = timebins.copy()

        # return an error if there is no event file set
        if self.event_file is None:
            raise RuntimeError("There was no event file specified ")

        # error checking for calc_energy_integrated
        if type(is_relative) is not bool:
            raise ValueError("The is_relative parameter should be a boolean value.")

        # see if the timebinalg is properly set approporiately
        # if "uniform" not in timebinalg or "snr" not in timebinalg or "bayesian" not in timebinalg:
        if timebinalg not in ["uniform", "snr", "highsnr", "bayesian"]:
            raise ValueError(
                'The timebinalg only accepts the following values: uniform, snr, highsnr, and bayesian (for bayesian blocks).')

        # if timebinalg == uniform/snr/highsnr, make sure that we have a timedelta that is a np.timedelta object
        if "uniform" in timebinalg or "snr" in timebinalg:
            if type(timedelta) is not np.timedelta64:
                raise ValueError('The timedelta variable needs to be a numpy timedelta64 object.')

        # need to make sure that snrthresh is set for "snr" timebinalg
        if "snr" in timebinalg and snrthresh is None:
            raise ValueError(f'The snrthresh value should be set since timebinalg is set to be {snrthresh}.')

        # test if is_relative is false and make sure that T0 is defined
        if is_relative and T0 is None:
            raise ValueError('The is_relative value is set to True however there is no T0 that is defined ' +
                             '(ie the time from which the time bins are defined relative to is not specified).')

        # see if we need to calculate energy integrated light curve at end of rebinning
        if len(self.ebins["INDEX"]) > 1:
            # see if we have the enrgy integrated bin included in the arrays:
            if (self.ebins["E_MIN"][0] == self.ebins["E_MIN"][-1]) and (
                    self.ebins["E_MAX"][-2] == self.ebins["E_MAX"][-1]):
                calc_energy_integrated = True  # this is for recalculating the lightcurve later in the method
            else:
                calc_energy_integrated = False
        else:
            calc_energy_integrated = False

        # if timebins, or tmin and tmax are defined then we ignore the timebinalg parameter
        # if tmin and tmax are specified while timebins is also specified then ignore timebins

        # do error checking on tmin/tmax
        if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
            raise ValueError('Both tmin and tmax must be defined.')

        if tmin is not None and tmax is not None:
            if tmin.size != tmax.size:
                raise ValueError('Both tmin and tmax must have the same length.')

            # now try to construct single array of all timebin edges in seconds
            timebins = np.zeros(tmin.size + 1) * u.s
            timebins[:-1] = tmin
            if tmin.size > 1:
                timebins[-1] = tmax[-1]
            else:
                timebins[-1] = tmax

        # See if we need to add T0 to everything
        if is_relative:
            # see if T0 is Quantity class
            if type(T0) is u.Quantity:
                timebins += T0
            else:
                timebins += T0 * u.s

        # if we are doing battblocks or the user has passed in timebins/tmin/tmax then we have to create a good time interval file
        # otherwise proceed with normal rebinning
        if "bayesian" in timebinalg:
            # we need to call battblocks to get the good time interval file
            self.timebins_file, battblocks_return = self._call_battblocks(save_durations=save_durations)
            tmp_lc_input_dict['timebinalg'] = "gti"
            tmp_lc_input_dict['gtifile'] = str(self.timebins_file)

            # make sure there are no predefined tstart/tstop
            tmp_lc_input_dict['tstart'] = "INDEF"
            tmp_lc_input_dict['tstop'] = "INDEF"

        elif (timebins is not None and timebins.size > 2):
            # tmin is not None and tmax.size > 1 and
            # already checked that tmin && tmax are not 1 and have the same size
            # if they are defined and they are more than 1 element then we have a series of timebins otherwise we just have the

            tmp_lc_input_dict['tstart'] = "INDEF"
            tmp_lc_input_dict['tstop'] = "INDEF"

            # start/stop times of the lightcurve
            self.timebins_file = self._create_custom_timebins(timebins)
            tmp_lc_input_dict['timebinalg'] = "gti"
            tmp_lc_input_dict['gtifile'] = str(self.timebins_file)
        else:
            tmp_lc_input_dict['gtifile'] = "NONE"

            # should have everything that we need to do the rebinning for a uniform/snr related rebinning
            # first need to update the tmp_lc_input_dict
            if "uniform" in timebinalg or "snr" in timebinalg:
                tmp_lc_input_dict['timebinalg'] = timebinalg

                # if we have snr we also need to modify the snrthreshold
                if "snr" in timebinalg:
                    tmp_lc_input_dict['snrthresh'] = snrthresh

            tmp_lc_input_dict['timedel'] = timedelta / np.timedelta64(1, 's')  # convert to seconds

            # see if we have the min/max times defined
            if (tmin is not None and tmax.size == 1):
                tmp_lc_input_dict['tstart'] = timebins[0].value
                tmp_lc_input_dict['tstop'] = timebins[1].value

        # before doing the recalculation, make sure that the proper weights are in the event file
        self._set_event_weights()

        # the LC _call_batbinevt method ensures that  outtype = LC and that clobber=YES
        lc_return = self._call_batbinevt(tmp_lc_input_dict)

        # make sure that the lc_return was successful
        if lc_return.returncode != 0:
            raise RuntimeError(f'The creation of the lightcurve failed with message: {lc_return.output}')
        else:
            self.bat_lc_result = lc_return
            self.lc_input_dict = tmp_lc_input_dict

            # reparse the lightcurve file to get the info
            self._parse_lightcurve_file(calc_energy_integrated=calc_energy_integrated)

    @u.quantity_input(energybins=["energy"], emin=['energy'], emax=['energy'])
    def set_energybins(self, energybins=[15, 25, 50, 100, 350] * u.keV, emin=None, emax=None,
                       calc_energy_integrated=True):
        """
        This method allows the energy binning to be set for the lightcurve. The energy rebinning is done automatically
        and the information for the rebinned lightcurve is automatically updated in the data attribute (which holds the
        light curve information itself including rates/counts, errors, fractional exposure, total counts, etc) and the
        ebins attribute which holds the energybins associated with the lightcurve.

        :param energybins: None or an astropy.unit.Quantity object of 2 or more elements with the energy bin edges in
            keV that the lightcurve should be binned into. None of the energy ranges should overlap.
        :param emin: an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the user would like. NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param emax: an astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            energy bins that the user would like. It should have the same number of elements as emin.
            NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param calc_energy_integrated: Boolean to denote wether the energy integrated light curve should be calculated
            based off the min and max energies that were passed in. If a single energy bin is requested for the rebinning
            then this argument does nothing.
        :return: None.
        """
        # make sure the user isnt trying to rebin a rate file
        if self._is_rate_lc:
            raise RuntimeError("The rate light curves cannot be rebinned in energy")

        # make sure we have all the info to do the rebinning
        if self.event_file is None or self.detector_quality_file is None:
            raise RuntimeError("The lightcurve cannot be rebinned in energy since one of the following files was not "
                               "initalized with this Lightcurve object: the event file "
                               "or the detector quality mask file.")

        # error checking for calc_energy_integrated
        if type(calc_energy_integrated) is not bool:
            raise ValueError("The calc_energy_integrated parameter should be a boolean value.")

        # see if the user specified either the energy bins directly or emin/emax separately
        if emin is None and emax is None:
            # make sure the energybins is not None
            if energybins is None:
                raise ValueError("energybins cannot be None if both emin and emax are set to None.")

            if energybins.size < 2:
                raise ValueError("The size of the energybins array must be >1.")

            emin = energybins[:-1].to(u.keV)
            emax = energybins[1:].to(u.keV)

        else:
            # make sure that both emin and emax are defined and have the same number of elements
            if (emin is None and emax is not None) or (emax is None and emin is not None):
                raise ValueError('Both emin and emax must be defined.')

            if emin.size != emax.size:
                raise ValueError('Both emin and emax must have the same length.')

        # create our energybins input to batbinevt
        if emin.size > 1:
            str_energybins = []
            for min, max in zip(emin.to(u.keV), emax.to(u.keV)):
                str_energybins.append(f"{min.value}-{max.value}")
        else:
            str_energybins = [f"{emin.to(u.keV).value}-{emax.to(u.keV).value}"]

        # create the full string
        ebins = ','.join(str_energybins)

        # create a temp dict to hold the energy rebinning parameters to pass to heasoftpy. If things dont run
        # successfully then the updated parameter list will not be saved
        tmp_lc_input_dict = self.lc_input_dict.copy()

        # need to see if the energybins are different (and even need to be calculated), if so do the recalculation
        if not np.array_equal(emin, self.ebins['E_MIN']) or not np.array_equal(emax, self.ebins['E_MAX']):
            # the tmp_lc_input_dict wil need to be modified with new Energybins
            tmp_lc_input_dict["energybins"] = ebins

            # before doing the recalculation, make sure that the proper weights are in the event file
            self._set_event_weights()

            # the LC _call_batbinevt method ensures that  outtype = LC and that clobber=YES
            lc_return = self._call_batbinevt(tmp_lc_input_dict)

            # make sure that the lc_return was successful
            if lc_return.returncode != 0:
                raise RuntimeError(f'The creation of the lightcurve failed with message: {lc_return.output}')
            else:
                self.bat_lc_result = lc_return
                self.lc_input_dict = tmp_lc_input_dict

                # reparse the lightcurve file to get the info
                self._parse_lightcurve_file(calc_energy_integrated=calc_energy_integrated)

    def _parse_lightcurve_file(self, calc_energy_integrated=True):
        """
        This method parses through a light curve file that has been created by batbinevent. The information included in
        the lightcurve file is read into the RA/DEC attributes (and checked to make sure that this is the lightcurve that
        the user wants to load in), the data attribute (which holds the light curve information itself including rates/counts,
        errors, fracitonal exposure, total counts, etc), the ebins attribute which holds the energybins associated with
        the lightcurve, the tbins attibute with the time bin edges and the time bin centers

        NOTE: A special value of timepixr=-1 (the default used when constructing light curves) specifies that
              timepixr=0.0 for uniformly binned light curves and
              timepixr=0.5 for non-unformly binned light curves. We recommend using the unambiguous TIME_CENT in the
              tbins attribute to prevent any confusion instead of the data["TIME"] values

        :param calc_energy_integrated: Boolean to denote if the energy integrated lightcurve should be calculated or not.
            By default, it is calculated unless the user has created a lightcurve of only 1 energy bin.
        :return: None
        """

        # determine if we have a rate lightcurve file, produced by the SDC, or a normally produced one calcualted with
        # batbinevt

        # error checking for calc_energy_integrated
        if type(calc_energy_integrated) is not bool:
            raise ValueError("The calc_energy_integrated parameter should be a boolean value.")

        with fits.open(self.lightcurve_file) as f:
            header = f[1].header
            data = f[1].data

            # if we are reading in a mask weighted event lightcurve then we need to read in the energy bins, otherwise
            # the energy bins are already known
            if "event" in header["DATAMODE"].lower():
                energies = f["EBOUNDS"].data
                energies_header = f["EBOUNDS"].header
                self._is_rate_lc = False
            else:
                self._is_rate_lc = True

        if self.ra is None and self.dec is None:
            if "deg" in header.comments["RA_OBJ"]:
                self.ra = header["RA_OBJ"] * u.deg
                self.dec = header["DEC_OBJ"] * u.deg
            else:
                raise ValueError(
                    "The lightcurve file RA/DEC_OBJ does not seem to be in the units of decimal degrees which is not supported.")
        else:
            # test if the passed in coordinates are what they should be for the light curve file
            # TODO: see if we are ~? arcmin close to one another
            assert (np.isclose(self.ra.to(u.deg).value, header["RA_OBJ"]) and np.isclose(self.dec.to(u.deg).value,
                                                                                         header["DEC_OBJ"])), \
                f"The passed in RA/DEC values ({self.ra},{self.dec}) do not match the values used to produce the lightcurve which are ({header['RA_OBJ']},{header['DEC_OBJ']})"

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy arrays as values
        self.data = {}
        for i in data.columns:
            self.data[i.name] = u.Quantity(data[i.name], i.unit)

        # fill in the energy bin info
        if "event" in header["DATAMODE"].lower():
            self.ebins = {}
            for i in energies.columns:
                if "CHANNEL" in i.name:
                    self.ebins["INDEX"] = energies[i.name]
                elif "E" in i.name:
                    self.ebins[i.name] = u.Quantity(energies[i.name], i.unit)
        else:
            if "1s" in header["DATAMODE"].lower():
                idx = np.array([0], dtype=np.int16)
                emin = [0] * u.keV
                emax = [np.inf] * u.keV
            else:
                idx = np.array([0, 1, 2, 3], dtype=np.int16)
                emin = [15, 25, 50, 100] * u.keV
                emax = [25, 50, 100, 350] * u.keV
            self.ebins = {"INDEX": idx, "E_MIN": emin, "E_MAX": emax}

        # fill in the time info separately
        timepixr = header["TIMEPIXR"]
        # see if there is a time delta column exists for variable time bin widths
        if "TIMEDEL" not in self.data.keys():
            dt = header["TIMEDEL"] * u.s
        else:
            dt = self.data["TIMEDEL"]

        self.tbins = {}
        # see https://heasarc.gsfc.nasa.gov/ftools/caldb/help/batbinevt.html
        self.tbins["TIME_CENT"] = self.data["TIME"] + (0.5 - timepixr) * dt
        self.tbins["TIME_START"] = self.data["TIME"] - timepixr * dt
        self.tbins["TIME_STOP"] = self.data["TIME"] + (1 - timepixr) * dt

        # if self.lc_input_dict ==None, then we will need to try to read in the hisotry of parameters passed into batbinevt
        # to create the lightcurve file. thsi usually is needed when we first parse a file so we know what things are if we need to
        # do some sort of rebinning.

        # were looking for something like:
        # START PARAMETER list for batbinevt_1.48 at 2023-11-01T20:38:05
        #
        # P1 infile = /Users/tparsota/Documents/01116441000_eventresult/events/sw0
        # P1 1116441000bevshsp_uf.evt
        # P2 outfile = 01116441000_eventresult/lc/lightcurve_0.lc
        # P3 outtype = LC
        # P4 timedel = 0.064
        # P5 timebinalg = uniform
        # P6 energybins = 15-350
        # P7 gtifile = NONE
        # P8 ecol = ENERGY
        # P9 weighted = YES
        # P10 outunits = INDEF
        # P11 timepixr = -1.0
        # P12 maskwt = NONE
        # P13 tstart = INDEF
        # P14 tstop = INDEF
        # P15 snrthresh = 6.0
        # P16 detmask = /Users/tparsota/Documents/01116441000_eventresult/auxil/sw
        # P16 01116441000bdqcb.hk.gz
        # P17 tcol = TIME
        # P18 countscol = DPH_COUNTS
        # P19 xcol = DETX
        # P20 ycol = DETY
        # P21 maskwtcol = MASK_WEIGHT
        # P22 ebinquant = 0.1
        # P23 delzeroes = no
        # P24 minfracexp = 0.1
        # P25 min_dph_frac_overlap = 0.999
        # P26 min_dph_time_overlap = 0.0
        # P27 max_dph_time_nonoverlap = 0.5
        # P28 buffersize = 16384
        # P29 clobber = yes
        # P30 chatter = 2
        # P31 history = yes
        # P32 mode = ql
        # END PARAMETER list for batbinevt_1.48

        if "event" in header["DATAMODE"].lower() and self.lc_input_dict is None:
            # get the default names of the parameters for batbinevt including its name 9which should never change)
            test = hsp_core.HSPTask('batbinevt')
            default_params_dict = test.default_params.copy()
            taskname = test.taskname
            start_processing = None

            for i in header["HISTORY"]:
                if taskname in i and start_processing is None:
                    # then set a switch for us to start looking at things
                    start_processing = True
                elif taskname in i and start_processing is True:
                    # we want to stop processing things
                    start_processing = False

                if start_processing and "START" not in i and len(i) > 0:
                    values = i.split(" ")
                    # print(i, values, "=" in values)

                    parameter_num = values[0]
                    parameter = values[1]
                    if "=" not in values:
                        # this belongs with the previous parameter and is a line continuation
                        default_params_dict[old_parameter] = default_params_dict[old_parameter] + values[-1]
                    else:
                        default_params_dict[parameter] = values[-1]

                    old_parameter = parameter

            self.lc_input_dict = default_params_dict.copy()

        if calc_energy_integrated:
            self._calc_energy_integrated()

    def _get_count_related_keys(self):
        """
        This method, returns only the names of the self.data keys that contain count information. This method
        explicitly exlcudes the following keys:
            - "TIME" for both rate lightcurves and normal batbinevt lightcurves
            - for normal lightcurves:
                - 'ERROR', 'TOTCOUNTS', 'FRACEXP', 'TIMEDEL'
        This is necessary for eg calculating the energy integrated rates/parsing the lightcurve files/plotting the lightcurves

        :return: list of keys
        """
        exclude_keys = ["TIME", "ERROR", 'TOTCOUNTS', 'FRACEXP', 'TIMEDEL']

        return [i for i in self.data.keys() if i not in exclude_keys]

    def _get_event_weights(self):
        """
        This method reads in the appropriate weights for event data once it has been applied to a event file, for a
        given RA/DEC position. This should only need to be done once, when the user has applied mask weighting
        in the BatEvent object and is creating a lightcurve.

        :return: None
        """

        # read in all the info for the weights and save it such that we can use these weights in the future for
        # redoing lightcurve calculation
        if self.event_file is not None:
            with fits.open(self.event_file) as file:
                self._event_weights = file[1].data["MASK_WEIGHT"]
        else:
            # no event file was specified when the object was created
            self._event_weights = None

    def _set_event_weights(self):
        """
        This method sets the appropriate weights for event data, for a
        given RA/DEC position. The weights are rewritten to the event file in the "MASK_WEIGHT" column.
        This may be necessary if a user is analyzing multiple sources for which event data has been
        obtained.

        Note: event weightings need to be set if the RA/DEC of the light curve doesnt match what is in the event file
        Note: if we have a mask weight image, then this can be used in batbinevt and supersedes the MASK_WEIGHT column
            of the event file
        :return: None
        """

        if self.event_file is not None and not self._same_event_lc_coords():
            # read in the event file and replace the values in the MASK_WEIGHT with the appropriate values in self._event_weights
            with fits.open(self.event_file, mode="update") as file:
                file[1].data["MASK_WEIGHT"] = self._event_weights
                # also make sure to modify the RA/DEC in header so we know what points in the sky the weights are
                # calculated for
                # update the event file RA/DEC_OBJ values everywhere
                for i in file:
                    i.header["RA_OBJ"] = self.ra.to(u.deg).value
                    i.header["DEC_OBJ"] = self.dec.to(u.deg).value

                    # the BAT_RA/BAT_DEC keys have to updated too since this is something
                    # that the software manual points out should be updated
                    i.header["BAT_RA"] = self.ra.to(u.deg).value
                    i.header["BAT_DEC"] = self.dec.to(u.deg).value

                file.flush()

    def _same_event_lc_coords(self):
        """
        This method reads in the event data coordinates and compares it to what is obtained from the lightcurve
        file that has been loaded in.

        :return: Boolean
        """

        with fits.open(self.event_file) as file:
            event_ra = file[0].header["RA_OBJ"]
            event_dec = file[0].header["DEC_OBJ"]
            coord_match = (event_ra == self.ra.to(u.deg).value) and (event_dec == self.dec.to(u.deg).value)

        return coord_match

    def _calc_energy_integrated(self):
        """
        This method just goes though the count rates in each energy bin that has been precalulated and adds them up.
        It also calcualtes the errors for lightcurved generated from event data. These arrays are added to self.data
        appropriately and the total energy min/max is added to self.ebins. For rate lightcurves in particular there can
        be multiple fits file extensions which is why this iterates over the data keys list that is returned from the
        _get_count_related_keys method.

        NOTE: This method can be expanded in the future to be an independent way of rebinning the lightcurves in
            energy without any calls to heasoftpy

        :return: None
        """

        data_keys = self._get_count_related_keys()

        # if we have more than 1 energy bin then we can calculate an energy integrated count rate, etc
        # otherwise we dont have to do anything since theres only one energy bin.
        # for rate lightcurves exclude error calculations since this is the hardware rates with no measurement error
        for data_key in data_keys:
            if self.data[data_key].ndim > 1:
                # calculate the total count rate and error
                integrated_count_rate = self.data[data_key].sum(axis=1)
                if not self._is_rate_lc:
                    integrated_count_rate_err = np.sqrt(np.sum(self.data["ERROR"] ** 2, axis=1))

                # get the energy info
                min_e = self.ebins["E_MIN"].min()
                max_e = self.ebins["E_MAX"].max()
                max_energy_index = self.ebins["INDEX"].max()

                # append energy integrated count rate, the error, and the additional energy bin to the respective dicts
                # only calculate the energy index/energy ranges for the first iteration of the outer loop since these
                # arrays should stay the same among all data dictionary keys
                if data_key == data_keys[0]:
                    new_energy_bin_size = self.ebins["INDEX"].size + 1
                    new_e_index = np.arange(new_energy_bin_size, dtype=self.ebins["INDEX"].dtype)

                    new_emin = np.zeros(new_energy_bin_size) * self.ebins["E_MIN"].unit
                    new_emin[:-1] = self.ebins["E_MIN"]
                    new_emin[-1] = min_e

                    new_emax = np.zeros_like(new_emin)  # the zeros_like gets the units from the array that is passed in
                    new_emax[:-1] = self.ebins["E_MAX"]
                    new_emax[-1] = max_e

                new_rate = np.zeros((self.data[data_key].shape[0], new_energy_bin_size)) * self.data[data_key].unit
                new_rate[:, :-1] = self.data[data_key]
                new_rate[:, -1] = integrated_count_rate

                # save the updated arrays
                self.ebins["INDEX"] = new_e_index
                self.ebins["E_MIN"] = new_emin
                self.ebins["E_MAX"] = new_emax

                self.data[data_key] = new_rate
                if not self._is_rate_lc:
                    new_rate_err = np.zeros_like(new_rate)
                    new_rate_err[:, :-1] = self.data["ERROR"]
                    new_rate_err[:, -1] = integrated_count_rate_err

                    self.data["ERROR"] = new_rate_err

    def plot(self, energybins=None, plot_counts=False, plot_exposure_fraction=False, time_unit="MET", T0=None,
             plot_relative=False):
        """
        This convenience method allows the user to plot the lightcurve generated from an event file or the rate
        lightcurve typically generated from hardware counts.

        :param energybins: None or a list or an astropy.units.Quantity object of 2 elements denoting the min and maximum
            of the energy bin that the user wants to plot. None defaults to plotting all the energy bins that have been
            created already. If a list is passed in the units of the min and max values are assumed to be keV.
        :param plot_counts: Boolean to denote if the total counts should be plotted alongside the rate/count lightcurve
        :param plot_exposure_fraction: Boolean to denote if the exposure fraction should be plotted alongside the rate/count lightcurve
        :param time_unit: string denoting the timeunit of the x-axis of the plot. This can be "MET", "MJD", or "UTC"
        :param T0: float or an astropy.units.Quantity with some tiem of interest (eg trigger time)
        :param plot_relative: Boolean switch denoting if the T0 that is passed in should be subtracted from the times.
            Thsi option is only applicable to time_unit="MET".
        :return: matplotlib figure, matplotlib axis for the plot
        """

        # make sure that energybins is a u.Quantity object
        if energybins is not None and type(energybins) is not u.Quantity:
            if type(energybins) is list:
                energybins = u.Quantity(energybins, u.keV)

        # have error checking
        if "MET" not in time_unit and "UTC" not in time_unit and "MJD" not in time_unit:
            raise ValueError("This method plots event data only using MET, UTC, or MJD time")

        if plot_relative and "MET" not in time_unit:
            raise ValueError("The plot_relative switch can only be set to True with time_unit=MET.")

        if "MET" in time_unit:
            start_times = self.tbins["TIME_START"]
            end_times = self.tbins["TIME_STOP"]
            mid_times = self.tbins["TIME_CENT"]
            xlabel = "MET (s)"

            if plot_relative:
                if T0 is None:
                    raise ValueError('The plot_relative value is set to True however there is no T0 that is defined ' +
                                     '(ie the time from which the time bins are defined relative to is not specified).')
                else:
                    # see if T0 is Quantity class
                    if type(T0) is not u.Quantity:
                        T0 *= u.s

                    start_times = start_times - T0
                    end_times = end_times - T0
                    mid_times = mid_times - T0
                    xlabel = f"MET - T0 (T0= {T0})"

        elif "MJD" in time_unit:
            start_times = met2mjd(self.tbins["TIME_START"].value)
            end_times = met2mjd(self.tbins["TIME_STOP"].value)
            mid_times = met2mjd(self.tbins["TIME_CENT"].value)
            xlabel = "MJD"
        else:
            start_times = met2utc(self.tbins["TIME_START"])
            end_times = met2utc(self.tbins["TIME_STOP"])
            mid_times = met2utc(self.tbins["TIME_CENT"].value)
            xlabel = "UTC"

        # get the number of axes we may need
        plot_data_key = self._get_count_related_keys()
        if not self._is_rate_lc:
            num_plots = 1
            num_plots += (plot_counts + plot_exposure_fraction)
            fig, ax = plt.subplots(num_plots, sharex=True)
        else:
            # if len(plot_data_key) >= 1:
            #    plot_data_key = plot_data_key[0]

            # want to have separate plots for the potential quad rate plots for the rtqd/rtmc rate lightcurves or just
            # the 1 plot for the 1s or ms rate lightcurves
            num_plots = len(plot_data_key)
            fig, ax = plt.subplots(num_plots, sharex=True)

        # assign the axes for each type of plot we may want
        axes_queue = [i for i in range(num_plots)]

        if num_plots > 1:
            if not self._is_rate_lc:
                rate_axes = [ax[axes_queue[0]]]
                axes_queue.pop(0)

                if plot_counts:
                    ax_count = ax[axes_queue[0]]
                    axes_queue.pop(0)

                if plot_exposure_fraction:
                    ax_exposure = ax[axes_queue[0]]
                    axes_queue.pop(0)
            else:
                rate_axes = ax[:len(plot_data_key)]
        else:
            rate_axes = [ax]

        all_lines = []
        all_labels = []
        # plot everything for the rates by default
        for ax_idx, data_key in enumerate(plot_data_key):
            ax_rate = rate_axes[ax_idx]
            for e_idx, emin, emax in zip(self.ebins["INDEX"], self.ebins["E_MIN"], self.ebins["E_MAX"]):
                plotting = True
                if energybins is not None:
                    # need to see if the energy range is what the user wants
                    if emin == energybins.min() and emax == energybins.max():
                        plotting = True
                    else:
                        plotting = False

                if plotting:
                    # use the proper indexing for the array
                    if len(self.ebins["INDEX"]) > 1:
                        rate = self.data[data_key][:, e_idx]
                        if not self._is_rate_lc:
                            rate_error = self.data["ERROR"][:, e_idx]
                        l = f'{self.ebins["E_MIN"][e_idx].value}-{self.ebins["E_MAX"][e_idx].value} ' + f'{self.ebins["E_MAX"][e_idx].unit}'
                    else:
                        rate = self.data[data_key]
                        if not self._is_rate_lc:
                            rate_error = self.data["ERROR"]
                        l = f'{self.ebins["E_MIN"][0].value}-{self.ebins["E_MAX"][0].value} ' + f'{self.ebins["E_MAX"].unit}'

                        # for the 1 second rate lightcurve, we have that the energy goes from 0 to +infinity. this is the
                        # only rate lightcurve that has a single energy bin
                        if self._is_rate_lc:
                            l = r"$0 - \infty$ " + f'{self.ebins["E_MAX"].unit}'

                    if self._is_rate_lc:
                        # if we are looking at a rate lightcurve, there may be gaps in the data. so filter these out for
                        # plotting. Min dt can be 1 sec, 1.6 sec, or up to 64 ms. To remove these gaps for all these
                        #  different time binnings just ID gaps that are larger than the mean dt
                        idx = np.where(np.diff(self.data["TIME"]) > np.diff(self.data["TIME"]).mean())[0]
                        rate[idx] = np.nan
                        start_times[idx] = np.nan
                        end_times[idx] = np.nan

                    line = ax_rate.plot(start_times, rate, ds='steps-post')
                    line_handle, = ax_rate.plot(end_times, rate, ds='steps-pre', color=line[-1].get_color(), label=l)
                    all_lines.append(line_handle)
                    all_labels.append(l)
                    if not self._is_rate_lc:
                        ax_rate.errorbar(mid_times, rate, yerr=rate_error, ls='None', color=line[-1].get_color())

                    # add the axis labels
                    data_label = data_key.replace('_', " ")

                    # for the quad counts just have Q0 for example
                    if "QUAD" in data_label:
                        data_label = data_label.replace("UAD", '')

                    # for the combined quad counts, to have Q_0_1 -> Q0+1
                    if len(data_label.split()) > 2:
                        str_list = ["".join(data_label.split(" ", 2)[:2]), data_label.split(" ", 2)[-1]]
                        data_label = "+".join(str_list)

                    ax_rate.set_ylabel(data_label + f" ({rate.unit})")

        # if we have multiple count related plots put legend at the top
        if len(plot_data_key) > 1 or not self._is_rate_lc:
            # calc the number of energy bins that we need to have labels for
            num_e = int(len(all_lines) / len(plot_data_key))
            if num_plots == 1:
                legend_ax = ax
            else:
                legend_ax = ax[0]
            legend_ax.legend(handles=all_lines[:num_e], labels=all_labels[:num_e],
                             bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=3)

        if plot_counts:
            line = ax_count.plot(start_times, self.data["TOTCOUNTS"], ds='steps-post', c='k')
            ax_count.plot(end_times, self.data["TOTCOUNTS"], ds='steps-pre', color=line[-1].get_color())
            ax_count.set_ylabel('Total counts (ct)')

        if plot_exposure_fraction:
            line = ax_exposure.plot(start_times, self.data["FRACEXP"], ds='steps-post', c='k')
            ax_exposure.plot(end_times, self.data["FRACEXP"], ds='steps-pre', color=line[-1].get_color())
            ax_exposure.set_ylabel('Fractional Exposure')

        if T0 is not None and not plot_relative:
            # plot the trigger time for all panels if we dont want the plotted times to be relative
            if num_plots > 1:
                for axis in ax:
                    line_handle = axis.axvline(T0, 0, 1, ls='--', label=f"T0={T0:.2f}", color='k')
            else:
                line_handle = ax_rate.axvline(T0, 0, 1, ls='--', label=f"T0={T0:.2f}", color='k')

        if num_plots > 1:
            if T0 is not None and not plot_relative:
                ax[1].legend([line_handle], [line_handle.get_label()])
            ax[-1].set_xlabel(xlabel)
        else:
            ax_rate.legend()
            ax_rate.set_xlabel(xlabel)

        plt.gca().ticklabel_format(useMathText=True)

        return fig, ax

    def _create_custom_timebins(self, timebins, output_file=None):
        """
        This method creates custom time bins from a user defined set of time bin edges. The created fits file with the
        timebins of interest will by default have the same name as the lightcurve file, however it will have a "gti" suffix instead of
        a "lc" suffix and it will be stored in the gti subdirectory of the event results directory.

        Note: This method is here so the call to create a gti file with custom timebins can be phased out eventually.

        :param timebins: a astropy.unit.Quantity object with the edges of the timebins that the user would like
        :param output_file: None or a Path object to where the output *.gti file will be saved to. A value of None
            defaults to the above description
        :return: Path object of the created good time intervals file
        """

        if output_file is None:
            # use the same filename as for the lightcurve file but replace suffix with gti and put it in gti subdir instead of lc
            new_path = self.lightcurve_file.parts
            new_name = self.lightcurve_file.name.replace("lc", "gti")

            output_file = Path(*new_path[:self.lightcurve_file.parts.index('lc')]).joinpath("gti").joinpath(new_name)

        return create_gti_file(timebins, output_file, T0=None, is_relative=False, overwrite=True)

    def _call_battblocks(self, output_file=None, save_durations=False):
        """
        This method calls battblocks for bayesian blocks binning of a lightcurve. This rebins the lightcurve into a 64 ms
        energy integrated energy bin (based on current ebins) to calculate the bayesian block time bins and then
        restores the lightcurve back to the prior energy binning.

        :param output_file: Path object of the file that will be created by battblocks with the good time intervals of
            interest
        :param save_durations: Boolean determining if battblocks should calculate durations such as T90, T50, background
            durations, etc and save those to a file. This file, if created, will be read in and saved to the tdurs attribute.
            This file will be saved to the gti subdirectory of the event results directory.
        :return: output_file Path object, HEASoftpy results object from battblocks call
        """

        if len(self.ebins["INDEX"]) > 1:
            recalc_energy = True
            # need to rebin to a single energy and save current energy bins
            old_ebins = self.ebins.copy()
            # see if we have the enrgy integrated bin included in the arrays:
            if (self.ebins["E_MIN"][0] == self.ebins["E_MIN"][-1]) and (
                    self.ebins["E_MAX"][-2] == self.ebins["E_MAX"][-1]):
                self.set_energybins(emin=self.ebins["E_MIN"][-1], emax=self.ebins["E_MAX"][-1])
                calc_energy_integrated = True  # this is for recalculating the lightcurve later in the method
            else:
                self.set_energybins(emin=self.ebins["E_MIN"][0], emax=self.ebins["E_MAX"][-1])
                calc_energy_integrated = False
        else:
            recalc_energy = False

        # set the time binning to be 64 ms. This time binning will be over written anyways so dont need to restore anything
        self.set_timebins()

        # get the set of default values which we will modify
        # stop
        test = hsp_core.HSPTask('battblocks')
        input_dict = test.default_params.copy()

        if output_file is None:
            # use the same filename as for the lightcurve file but replace suffix with gti and put it in gti subdir instead of lc
            new_path = self.lightcurve_file.parts
            new_name = self.lightcurve_file.name.replace("lc", "gti")

            output_file = Path(*new_path[:self.lightcurve_file.parts.index('lc')]).joinpath("gti").joinpath(new_name)

        # modify some of the inputs here
        input_dict["infile"] = str(
            self.lightcurve_file)  # this should ideally be a 64 ms lightcurve of a single energy bin
        input_dict["outfile"] = str(output_file)

        # these are used by batgrbproducts:
        input_dict["bkgsub"] = "YES"
        input_dict["clobber"] = "YES"
        input_dict["tlookback"] = 10

        if save_durations:
            dur_output_file = output_file.parent / output_file.name.replace("gti", "dur")
            input_dict["durfile"] = str(dur_output_file)

        try:
            battblocks_return = hsp.battblocks(**input_dict)

        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft battblocks failed with inputs {input_dict}.")

        # reset the energy bins to what they were before
        if recalc_energy:
            if calc_energy_integrated:
                min_ebins = old_ebins["E_MIN"][:-1]
                max_ebins = old_ebins["E_MAX"][:-1]
            else:
                min_ebins = old_ebins["E_MIN"]
                max_ebins = old_ebins["E_MAX"]

            self.set_energybins(emin=min_ebins, emax=max_ebins,
                                calc_energy_integrated=calc_energy_integrated)

        if battblocks_return.returncode != 0:
            raise RuntimeError(f'The call to Heasoft battblocks failed with message: {battblocks_return.output}')

        if save_durations:
            self._parse_durations(dur_output_file)

        return output_file, battblocks_return

    def _parse_durations(self, duration_file):
        """
        This method reads in a duration file that is produced by battblocks.

        :param duration_file: a Path object to the battblocks produced durations file that will be read in
        :return: None
        """

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy arrays as values
        with fits.open(duration_file) as file:
            # skip the primary since it contains no info, the other extensions have the T90, backgrounds, etc
            for f in file[1:]:
                header = f.header
                data = f.data
                dur_quantity = f.name.split("_")[-1]

                # there is only 1 time in each data column so index data[0] and convert START/STOP to TSTART/TSTOP
                self.set_duration(dur_quantity, u.Quantity(data[data.columns[0].name][0], data.columns[0].unit),
                                  u.Quantity(data[data.columns[1].name][0], data.columns[1].unit))

    def _call_batbinevt(self, input_dict):
        """
        Calls heasoftpy's batbinevt with an error wrapper, ensures that this bins the event data to produce a lightcurve

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batbinevt
        :return: heasoftpy Result object from batbinevt
        """

        input_dict["clobber"] = "YES"
        input_dict["outtype"] = "LC"

        try:
            return hsp.batbinevt(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batbinevt failed with inputs {input_dict}.")

    def get_duration(self, duration_str):
        """
        This method allows a user to get the start MET time, stop MET time, and difference between the two for the
        specified duration of interest. This string should be a dictionary key that has been saved previously otherwise
        a ValueError will be raised. If there is no tdurs attibute that has been filled with data, then a RuntimeError will be
        raised.

        :param duration_str: string of a duration quantity that has been saved for the lightcurve. This should be an existing
            key in the tdurs attrubute of the lightcurve.
        :return: start time in MET, stop time in MET, (stop time in MET) - (start time in MET)
        """

        if self.tdurs is not None:
            try:
                data = self.tdurs[duration_str]
            except KeyError as e:
                print(e)
                raise ValueError(f"The duration {duration_str} has not been calculated by battblocks.")
        else:
            raise RuntimeError(
                "There are not durations (T90, T50, etc) associated with this lightcurve. Please rerun set_timebins with timebinalg = bayesian.")

        return data["TSTART"], data["TSTOP"], data["TSTOP"] - data["TSTART"]

    @u.quantity_input(tstart=['time'], tstop=['time'])
    def set_duration(self, duration_str, tstart, tstop):
        """
        This method allows users to set durations that were calculated.

        :param duration_str: a string to denote the duration value that will be saved
        :param tstart: a astropy.unit.Quantity object that denotes the start time in MET of the duration that will be saved
        :param tstop: a astropy.unit.Quantity object that denotes the end time in MET of the duration that will be saved
        :return: None
        """

        # if we havent created a dict , do so now
        if self.tdurs is None:
            self.tdurs = {}

        # see if we can access the key of interest, otherwise we need to create it
        try:
            data = self.tdurs[duration_str]
        except KeyError as e:
            self.tdurs[duration_str] = {}

        # now save values appropriately
        self.tdurs[duration_str]["TSTART"] = tstart
        self.tdurs[duration_str]["TSTOP"] = tstop

    @classmethod
    def from_file(cls, lightcurve_file, event_file=None, detector_quality_file=None):
        """
        This class method takes an existing lightcurve file and returns a Lightcurve class object with the data
        contained in the lightcurve file. The user will be able to plot the lightcurve. If the event file, or the
        detector quality mask files are not specified, then the user will not be able to dynamically change the
        lightcurve energy bins or time bins

        :param lightcurve_file: path object of the lightcurve file that will be read in, if previously calculated,
            or the location/name of the new lightcurve file that will contain the newly calculated lightcurve.
        :param event_file: None or Path object for the event file with mask weighting already applied, from which we
            will construct the lightcurve or read the previously constructed lightcurve file
        :param detector_quality_file: None or Path object for the detector quality mask that was constructed for
            the associated event file
        :return: Lightcurve class object with the loaded light curve file data
        """
        lightcurve_file = Path(lightcurve_file).expanduser().resolve()

        if not lightcurve_file.exists():
            raise ValueError(f"The specified lightcurve file {lightcurve_file} does not seem to exist. "
                             f"Please double check that it does.")

        return cls(lightcurve_file, event_file, detector_quality_file)


class Spectrum(BatObservation):
    @u.quantity_input(ra=u.deg, dec=u.deg)
    def __init__(self, pha_file, event_file, detector_quality_file, auxil_raytracing_file, ra=None, dec=None,
                 pha_input_dict=None, mask_weighting=True, recalc=False):
        """
        This initalizes a pha fits file based off of a passed in event file where mask weighting
        has been applied and the detector quality mask has been constructed. Alternatively, this method can read in a
        previously calculated pha file. If recalc=True then the passed in pha file will be recalculated using the passed
        in pha_input_dict or a default pha_input_dict defined as:

        dict(infile=str(event_file), outfile=str(pha_file), outtype="PHA",
                            energybins="CALDB", weighted="YES", timedel=0.0,
                            detmask=str(detector_quality_file),
                            tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")

        The ra/dec of the source that this lightcurve was constructed for (and for which the weighting was applied to
        the event file), can be specified or it can be dynamically read from the event file.

        :param pha_file: Path object of the pha file that will be read in, if previously calculated,
            or the full path and name of the new lightcurve file that will contain the newly calculated pha file.
        :param event_file: Path object for the event file with mask weighting already applied, from which we will construct
            the pha file or read the previously constructed pha file
        :param detector_quality_file: Path object for the detector quality mask that was constructed for the associated
            event file
        :param auxil_raytracing_file: Path object pointing to the auxiliary ray tracing file that is created by applying
            the mask weighting to the event file that is passed in.
        :param ra: None or astropy Quantity representing the decimal degree RA value of the source for which the mask weighting was
            applied to the passed in event file. A value of None indicates that the RA of the source will be obtained
            from the event file which is then saved to pha_file
        :param dec: None or astropy Quantity representing the decimal degree DEC value of the source for which the mask weighting was
            applied to the passed in event file. A value of None indicates that the DEC of the source will be obtained
            from the event file which is then saved to pha_file
        :param pha_input_dict: None or a dict of values that will be passed to batbinevt in the creation of the pha file.
            If a pha file is being read in that was previously created, the prior parameters that were used to
            calculate the lightcurve will be used.
            If pha_input_dict is None, this will be set to:
                dict(infile=str(event_file), outfile=str(pha_file), outtype="PHA",
                            energybins="CALDB", weighted="YES", timedel=0.0,
                            detmask=str(detector_quality_file),
                            tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")
            See HEASoft documentation on batbinevt to see what these parameters mean. Alternatively, these defaults can
            be used in the initial call and time/energy rebinning can be done using the set_timebins and set_energybins
            methods associated with the Spectrum object.
        :param mask_weighting: Boolean to denote if mask weighting should be applied. By default this is set to True,
            however if a source is out of the BAT field of view the mask weighting will produce a pha with of 0 counts.
            Setting mask_weighting=False in this case ignores the position of the source and allows the pure rates/counts
            to be calculated.
        :param recalc: Boolean to denote if the pha specified by pha_file should be recalculated with the
            pha_input_dict values (either those passed in or those that are defined by default)

        """
        # NOTE: Lots of similarities here as with the lightcurve since we are using batbinevt as the base. If there
        # are any issues with the lightcurve object, then we should make sure that these same problems do not occur
        # in the spectrum object and vice versa

        # save these variables
        self.pha_file = Path(pha_file).expanduser().resolve()

        # if any of these below are None, produce a warning that we wont be able to modify the spectrum. Also do
        # error checking for the files existing, etc
        if event_file is not None:
            self.event_file = Path(event_file).expanduser().resolve()
            if not self.event_file.exists():
                raise ValueError(f"The specified event file {self.event_file} does not seem to exist. Please double "
                                 f"check that it does.")
        else:
            self.event_file = None
            warnings.warn("No event file has been specified. The resulting spectrum object will not be able to be"
                          "modified either by rebinning in energy or time.", stacklevel=2)

        if detector_quality_file is not None:
            self.detector_quality_file = Path(detector_quality_file).expanduser().resolve()
            if not self.detector_quality_file.exists():
                raise ValueError(f"The specified detector quality mask file {self.detector_quality_file} does not seem "
                                 f"to exist. Please double check that it does.")

        else:
            self.detector_quality_file = None
            warnings.warn("No detector quality mask file has been specified. The resulting spectrum object will not "
                          "be able to be modified either by rebinning in energy or time.", stacklevel=2)

        if auxil_raytracing_file is not None:
            self.auxil_raytracing_file = Path(auxil_raytracing_file).expanduser().resolve()
            if not self.auxil_raytracing_file.exists():
                raise ValueError(f"The specified auxiliary raytracing file {self.auxil_raytracing_file} does not seem "
                                 f"to exist. Please double check that it does.")

        else:
            self.auxil_raytracing_file = None
            warnings.warn("No auxiliary ray tracing file has been specified. The resulting spectrum object will not "
                          "be able to be modified either by rebinning in energy or time.", stacklevel=2)

        # need to see if we have to construct the lightcurve if the file doesnt exist
        if not self.pha_file.exists() or recalc:
            # see if the input dict is None so we can set these defaults, otherwise save the requested inputs for use
            # later
            if pha_input_dict is None:
                # energybins = "CALDB" gives us the 80 channel spectrum
                self.pha_input_dict = dict(infile=str(self.event_file), outfile=str(self.pha_file), outtype="PHA",
                                           energybins="CALDB", weighted="YES", timedel=0.0,
                                           detmask=str(self.detector_quality_file),
                                           tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")

                # specify if we want mask weighting
                if mask_weighting:
                    self.pha_input_dict["weighted"] = "YES"
                else:
                    self.pha_input_dict["weighted"] = "NO"


            else:
                self.pha_input_dict = pha_input_dict

            # create the pha
            self._create_pha(self.pha_input_dict)
        else:
            # set the self.lc_input_dict = None so the parsing of the pha file tries to also load in the
            # parameters passed into batbinevt to create the pha file
            # try to parse the existing pha file to see what parameters were passed to batbinevt to construct the file
            self.pha_input_dict = None

        # set default DRM info
        self.drm_file = None
        self.drm = None
        self.spectral_model = None

        # set default RA/DEC coordinates correcpondent to the pha file which will be filled in later if it is set to None
        self.ra = ra
        self.dec = dec

        # read info from the lightcurve file including if there is a drm file associated through the RESP header key
        self._parse_pha_file()

        # read in the information about the weights
        self._get_event_weights()

        # (re)calculate the drm file if it hasnt been set in the _parse_pha_file method
        # or if we are directed to
        if self.drm_file is None or recalc:
            self._call_batdrmgen()

        # were done getting all the info that we need. From here, the user can rebin the timebins and the energy bins

    def _create_pha(self, batbinevt_input_dict):
        """
        This method calls all necessary bits of creating a PHA file:
        _call_batbinevt, _call_batphasyserr, & _call_batupdatephakw.

        These create the PHA file, applies the systematic error, and updates the keywords based on the auxiliary
        ray tracing file.

        :param batbinevt_input_dict:  dict of values that will be passed to batbinevt in the creation of the pha file
        :return: None
        """

        # create the pha
        tmp_bat_pha_result = self._call_batbinevt(batbinevt_input_dict)

        # make sure that this calculation ran successfully
        if tmp_bat_pha_result.returncode != 0:
            raise RuntimeError(f'The creation of the PHA file failed with message: {tmp_bat_pha_result.output}')
        else:
            self.bat_pha_result = tmp_bat_pha_result
            self._call_batphasyserr()
            self._call_batupdatephakw()

    @u.quantity_input(timebins=['time'], tmin=['time'], tmax=['time'])
    def set_timebins(self, timebinalg="uniform", timebins=None, tmin=None, tmax=None, T0=None, is_relative=False,
                     timedelta=np.timedelta64(0, 's'), snrthresh=None):
        """
        This method allows for the rebinning of the pha in time. The time binning is done dymaically and the
        information for the rebinned pha file is automatically updated in the data attribute (which holds the pha
        information itself including rates/counts, errors, etc),
        and the tbins attribute with the time bin edges and the time bin centers.

        :param timebinalg: a string that can be set to "uniform" or "snr"
            "uniform" will do a uniform time binning from the specified tmin to tmax with the size of the bin set by
                the timedelta parameter.
            "snr" will bin the pha until a maximum snr threshold is achieved, as is specified by the snrthresh parameter,

            NOTE: more information can be found by looking at the HEASoft documentation for batbinevt and battblocks
        :param timebins: astropy.units.Quantity denoting the array of time bin edges. Units will usually be in seconds
            for this. The values can be relative to the specified T0. If so, then the T0 needs to be specified and
            the is_relative parameter should be True.
        :param tmin: astropy.units.Quantity denoting the minimum values of the timebin edges that the user would like
            the lightcurve to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.

            If the length of tmin is 1 then this denotes the time when the binned pha file should start. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param tmax:astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
            the pha to be binned into. Units will usually be in seconds for this. The values can be relative to
            the specified T0. If so, then the T0 needs to be specified andthe is_relative parameter should be True.
            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.

            If the length of tmin is 1 then this denotes the time when the binned lightcurve should end. For this single
            value, it can also be defined relative to T0. If so, then the T0 needs to be specified and the is_relative parameter
            should be True.

            NOTE: if tmin/tmax are specified then anything passed to the timebins parameter is ignored.
        :param T0: float or an astropy.units.Quantity object with some time of interest (eg trigger time)
        :param is_relative: Boolean switch denoting if the T0 that is passed in should be added to the
            timebins/tmin/tmax that were passed in.
        :param timedelta: numpy.timedelta64 object denoting the size of the time binning. This value is used when
            timebinalg is set. When timebin=np.timedelta64(0, "s") the whole event dataset gets
            accumulated into a single spectrum.
        :param snrthresh: float representing the snr threshold associated with the timebinalg="snr" or timebinalg="highsnr"
            parameter values. See above description of the timebinalg parameter to see how this snrthresh parameter is used.
        :return: None
        """

        # create a temp copy incase the time rebinning doesnt complete successfully
        tmp_pha_input_dict = self.pha_input_dict.copy()

        # create a copy of the timebins if it is not None to prevent modifying the original array
        if timebins is not None:
            timebins = timebins.copy()

        # make sure we have all the info to do the rebinning
        if self.event_file is None or self.auxil_raytracing_file is None or self.detector_quality_file is None:
            raise RuntimeError("The spectrum cannot be rebinned in time since one of the following files was not "
                               "initalized with this Spectrum object: the event file, the auxiliary raytracing file, "
                               "or the detector quality mask file.")

        # error checking for calc_energy_integrated
        if type(is_relative) is not bool:
            raise ValueError("The is_relative parameter should be a boolean value.")

        # see if the timebinalg is properly set approporiately
        if timebinalg not in ["uniform", "snr", "highsnr"]:
            raise ValueError('The timebinalg only accepts the following values: uniform and snr.')

        # if timebinalg == uniform/snr/highsnr, make sure that we have a timedelta that is a np.timedelta object
        if "uniform" in timebinalg or "snr" in timebinalg:
            if type(timedelta) is not np.timedelta64:
                raise ValueError('The timedelta variable needs to be a numpy timedelta64 object.')

        # need to make sure that snrthresh is set for "snr" timebinalg
        if "snr" in timebinalg and snrthresh is None:
            raise ValueError(f'The snrthresh value should be set since timebinalg is set to be {snrthresh}.')

        # test if is_relative is false and make sure that T0 is defined
        if is_relative and T0 is None:
            raise ValueError('The is_relative value is set to True however there is no T0 that is defined ' +
                             '(ie the time from which the time bins are defined relative to is not specified).')

        # if timebins, or tmin and tmax are defined then we ignore the timebinalg parameter
        # if tmin and tmax are specified while timebins is also specified then ignore timebins

        # do error checking on tmin/tmax
        if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
            raise ValueError('Both emin and emax must be defined.')

        if tmin is not None and tmax is not None:
            if tmin.size != tmax.size:
                raise ValueError('Both tmin and tmax must have the same length.')

            # now try to construct single array of all timebin edges in seconds
            timebins = np.zeros(tmin.size + 1) * u.s
            timebins[:-1] = tmin
            if tmin.size > 1:
                timebins[-1] = tmax[-1]
            else:
                timebins[-1] = tmax

        # See if we need to add T0 to everything
        if is_relative:
            # see if T0 is Quantity class
            if type(T0) is u.Quantity:
                timebins += T0
            else:
                timebins += T0 * u.s

        # if the user has passed in timebins/tmin/tmax then we have to create a good time interval file
        # otherwise proceed with normal rebinning
        if (timebins is not None and timebins.size > 2):
            # tmin is not None and tmax.size > 1 and already checked that tmin && tmax are not 1 and have the same
            # size if they are defined and they are more than 1 element then we have a series of timebins otherwise
            # we just have the

            tmp_pha_input_dict['tstart'] = "INDEF"
            tmp_pha_input_dict['tstop'] = "INDEF"

            # start/stop times of the lightcurve
            self.timebins_file = self._create_custom_timebins(timebins)
            tmp_pha_input_dict['timebinalg'] = "gti"
            tmp_pha_input_dict['gtifile'] = str(self.timebins_file)
        else:
            tmp_pha_input_dict['gtifile'] = "NONE"

            # should have everything that we need to do the rebinning for a uniform/snr related rebinning
            # first need to update the tmp_lc_input_dict
            if "uniform" in timebinalg or "snr" in timebinalg:
                tmp_pha_input_dict['timebinalg'] = timebinalg

                # if we have snr we also need to modify the snrthreshold
                if "snr" in timebinalg:
                    tmp_pha_input_dict['snrthresh'] = snrthresh

            tmp_pha_input_dict['timedel'] = timedelta / np.timedelta64(1, 's')  # convert to seconds

            # see if we have the min/max times defined
            if (tmin is not None and tmax.size == 1):
                # were just defining the min/max times so want timedel=0 for there to just be a singel time bin
                tmp_pha_input_dict['timedel'] = 0
                tmp_pha_input_dict['tstart'] = timebins[0].value
                tmp_pha_input_dict['tstop'] = timebins[1].value

        # before doing the recalculation, make sure that the proper weights are in the event file
        self._set_event_weights()

        # create the pha
        self._create_pha(tmp_pha_input_dict)

        # all error handling is in _create_pha so we are all good if we get here to save the dict
        self.pha_input_dict = tmp_pha_input_dict

        # reparse the pha file to get the info
        self._parse_pha_file()

        # recalculate the drm file
        self._call_batdrmgen()

        # reset any spectral fits
        self.spectral_model = None

    @u.quantity_input(emin=['energy'], emax=['energy'])
    def set_energybins(self, energybins="CALDB", emin=None, emax=None):
        """
        This method allows the energy binning to be set for the pha file. The energy rebinning is done automatically
        and the information for the rebinned lightcurve is automatically updated in the data attribute (which holds the
        pha file information itself including rates/counts, errors,  etc) and the
        ebins attribute which holds the energybins associated with the pha file.

        :param energybins: single string "CALDB" denoting that the 80 channel default spectrum should be constructed or
          an astropy.unit.Quantity object of 2 or more elements with the energy bin edges in keV that the pha should be
          binned into. None of the energy ranges should overlap.
        :param emin: an astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the user would like. NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param emax: an astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            energy bins that the user would like. It shoudl have the same number of elements as emin.
            NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :return: None.
        """

        # make sure we have all the info to do the rebinning
        if self.event_file is None or self.auxil_raytracing_file is None or self.detector_quality_file is None:
            raise RuntimeError("The spectrum cannot be rebinned in energy since at least one of the following files was"
                               " not initalized with this Spectrum object: the event file, the auxiliary raytracing "
                               "file, or the detector quality mask file.")

        # see if the user specified either the energy bins directly or emin/emax separately
        if emin is None and emax is None:

            if "CALDB" not in energybins:
                # verify that we have a Quantity array with >1 elements
                if not isinstance(energybins, u.Quantity):
                    raise ValueError(
                        'The energybins only accepts an astropy Quantity with the energy bin edges for the pha file.')

                if energybins.size < 2:
                    raise ValueError("The size of the energybins array must be >1.")

                emin = energybins[:-1].to(u.keV)
                emax = energybins[1:].to(u.keV)

        else:
            # make sure that both emin and emax are defined and have the same number of elements
            if (emin is None and emax is not None) or (emax is None and emin is not None):
                raise ValueError('Both emin and emax must be defined.')

            # see if they are astropy quantity items with units
            if type(emin) is not u.Quantity:
                emin = u.Quantity(emin, u.keV)
            if type(emax) is not u.Quantity:
                emax = u.Quantity(emax, u.keV)

            if emin.size != emax.size:
                raise ValueError('Both emin and emax must have the same length.')

        # create our energybins input to batbinevt
        if emin.size > 1:
            str_energybins = []
            for min_e, max_e in zip(emin.to(u.keV), emax.to(u.keV)):
                str_energybins.append(f"{min_e.value}-{max_e.value}")
        else:
            str_energybins = [f"{emin.to(u.keV).value}-{emax.to(u.keV).value}"]

        # create the full string
        ebins = ','.join(str_energybins)

        # create a temp dict to hold the energy rebinning parameters to pass to heasoftpy. If things dont run
        # successfully then the updated parameter list will not be saved
        tmp_pha_input_dict = self.pha_input_dict.copy()

        # also need to check if ebins=="CALDB and that is different from the tmp_pha_input_dict
        if "CALDB" in ebins:
            recalc = not np.array_equal(ebins, self.tmp_pha_input_dict["energybins"])
        else:
            # need to see if the energybins are different (and even need to be calculated), if so do the recalculation
            recalc = not np.array_equal(emin, self.ebins['E_MIN']) or not np.array_equal(emax, self.ebins['E_MAX'])

        # do the rebinning if needed
        if recalc:
            # the tmp_lc_input_dict wil need to be modified with new Energybins
            tmp_pha_input_dict["energybins"] = ebins

            # before doing the recalculation, make sure that the proper weights are in the event file
            self._set_event_weights()

            # create the pha
            self._create_pha(tmp_pha_input_dict)

            # all error handling is in _create_pha so we are all good if we get here to save the dict
            self.pha_input_dict = tmp_pha_input_dict

            # reparse the pha file to get the info
            self._parse_pha_file()

            self._call_batdrmgen()

            # reset any spectral fit
            self.spectral_model = None

    def _call_batbinevt(self, input_dict):
        """
        Calls heasoftpy's batbinevt with an error wrapper, ensures that this bins the event data to produce a lightcurve

        :param input_dict: Dictionary of inputs that will be passed to heasoftpy's batbinevt
        :return: heasoftpy Result object from batbinevt
        """

        input_dict["clobber"] = "YES"
        input_dict["outtype"] = "PHA"

        try:
            return hsp.batbinevt(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batbinevt failed with inputs {input_dict}.")

    def _call_batphasyserr(self):
        """
        Calls heasoftpy's batphasyserr which applies systematic errors to the 80 channel PHA spectrum. The systematic
        errors live in CALDB at: https://heasarc.gsfc.nasa.gov/FTP/caldb/data/swift/bat/cpf/swbsyserr20030101v003.fits

        :return: heasoftpy Result object from batphasyserr
        """
        pha_file = self.pha_file
        input_dict = dict(infile=str(pha_file), syserrfile="CALDB")

        try:
            tmp_bat_pha_sys_result = hsp.batphasyserr(**input_dict)
            if tmp_bat_pha_sys_result.returncode != 0:
                raise RuntimeError(
                    f'The application of systematic errors to the PHA file failed with message: {tmp_bat_pha_sys_result.output}')
            else:
                return tmp_bat_pha_sys_result
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batphasyserr failed with inputs {input_dict}.")

    def _call_batupdatephakw(self):
        """
        Calls heasoftpy's batupdatephakw which applies geometrical corrections to the PHA spectrum which is especially
        important is BAT is slewing during an observation and the source position is changing.

        :return: heasoftpy Result object from batupdatephakw
        """
        pha_file = self.pha_file
        input_dict = dict(infile=str(pha_file), auxfile=str(self.auxil_raytracing_file))

        try:
            tmp_bat_pha_kw_result = hsp.batupdatephakw(**input_dict)
            if tmp_bat_pha_kw_result.returncode != 0:
                raise RuntimeError(
                    f'The application of geometric corrections to the PHA file failed with message: {tmp_bat_pha_kw_result.output}')
            else:
                return tmp_bat_pha_kw_result

        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batupdatephakw failed with inputs {input_dict}.")

    def _call_batdrmgen(self):
        """
        This calls heasoftpy's batdrmgen which produces the associated drm for fitting the PHA file.

        :return: None
        """

        pha_file = self.pha_file
        output = calc_response(pha_file)

        if output.returncode != 0:
            raise RuntimeError(f"The call to Heasoft batdrmgen failed with output {output.stdout}.")

        self.drm_file = pha_file.parent.joinpath(f"{pha_file.stem}.rsp")
        # self.set_drm_filename(drm_file)

    def _get_event_weights(self):
        """
        This method reads in the appropriate weights for event data once it has been applied to an event file, for a
        given RA/DEC position. This should only need to be done once, when the user has applied mask weighting
        in the BatEvent object and is creating a lightcurve.

        :return: None
        """

        # read in all the info for the weights and save it such that we can use these weights in the future for
        # redoing lightcurve calculation
        if self.event_file is not None:
            with fits.open(self.event_file) as file:
                self._event_weights = file[1].data["MASK_WEIGHT"]
        else:
            # no event file was specified when the object was created
            self._event_weights = None

    def _set_event_weights(self):
        """
        This method sets the appropriate weights for event data, for a
        given RA/DEC position. The weights are rewritten to the event file in the "MASK_WEIGHT" column.
        This may be necessary if a user is analyzing multiple sources for which event data has been
        obtained.

        Note: event weightings need to be set if the RA/DEC of the light curve doesnt match what is in the event file
        Note: if we have a mask weight image, then this can be used in batbinevt and supersedes the MASK_WEIGHT column
            of the event file

        :return: None
        """

        if self.event_file is not None and not self._same_event_lc_coords():
            # read in the event file and replace the values in the MASK_WEIGHT with the appropriate values in
            # self._event_weights
            with fits.open(self.event_file, mode="update") as file:
                file[1].data["MASK_WEIGHT"] = self._event_weights
                # also make sure to modify the RA/DEC in header so we know what points in the sky the weights are
                # calculated for
                # update the event file RA/DEC_OBJ values everywhere
                for i in file:
                    i.header["RA_OBJ"] = self.ra.to(u.deg).value
                    i.header["DEC_OBJ"] = self.dec.to(u.deg).value

                    # the BAT_RA/BAT_DEC keys have to updated too since this is something
                    # that the software manual points out should be updated
                    i.header["BAT_RA"] = self.ra.to(u.deg).value
                    i.header["BAT_DEC"] = self.dec.to(u.deg).value

                file.flush()

    def _same_event_lc_coords(self):
        """
        This method reads in the event data coordinates and compares it to what is obtained from the lightcurve
        file that has been loaded in.

        :return: Boolean
        """

        with fits.open(self.event_file) as file:
            if "deg" in file[0].header.comments["RA_OBJ"]:
                event_ra = file[0].header["RA_OBJ"] * u.deg
                event_dec = file[0].header["DEC_OBJ"] * u.deg
            else:
                raise ValueError(
                    "The PHA file RA/DEC_OBJ does not seem to be in the units of decimal degrees which is not supported.")
            coord_match = (event_ra == self.ra) and (event_dec == self.dec)

        return coord_match

    def _parse_pha_file(self):
        """
        This method parses through a pha file that has been created by batbinevent. The information included in
        the pha file is read into the RA/DEC attributes (and checked to make sure that this is the pha that
        the user wants to load in), the data attribute (which holds the pha information itself including rates/counts,
        errors,  etc), the ebins attribute which holds the energybins associated with
        the pha file, the tbins attibute with the time bin edges and the time bin centers

        NOTE: A special value of timepixr=-1 (the default used when constructing light curves) specifies that
              timepixr=0.0 for uniformly binned light curves and
              timepixr=0.5 for non-unformly binned light curves. We recommend using the unambiguous TIME_CENT in the
              tbins attribute to prevent any confusion instead of the data["TIME"] values

        :return: None
        """
        pha_file = self.pha_file
        with fits.open(pha_file) as f:
            header = f[1].header
            data = f[1].data
            energies = f["EBOUNDS"].data
            energies_header = f["EBOUNDS"].header
            times = f["STDGTI"].data

        if self.ra is None and self.dec is None:
            if "deg" in header.comments["RA_OBJ"]:
                self.ra = header["RA_OBJ"] * u.deg
                self.dec = header["DEC_OBJ"] * u.deg
            else:
                raise ValueError(
                    "The PHA file RA/DEC_OBJ does not seem to be in the units of decimal degrees which is not supported.")

        else:
            # test if the passed in coordinates are what they should be for the light curve file
            # TODO: see if we are ~? arcmin close to one another
            assert (np.isclose(self.ra.to(u.deg).value, header["RA_OBJ"]) and np.isclose(self.dec.to(u.deg).value,
                                                                                         header["DEC_OBJ"])), \
                (f"The passed in RA/DEC values ({self.ra},{self.dec}) "
                 f"do not match the values used to produce the lightcurve which are "
                 f"({header['RA_OBJ']},{header['DEC_OBJ']})")

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy
        # arrays as values
        self.data = {}
        for i in data.columns:
            self.data[i.name] = u.Quantity(data[i.name], i.unit)

        # recalculate the systematic error based on the counts/rates
        if 'RATE' in self.data.keys():
            key = "RATE"
        else:
            key = "COUNTS"

        self.data["SYS_ERR"] = self.data["SYS_ERR"] * self.data[key]

        # fill in the energy bin info
        self.ebins = {}
        for i in energies.columns:
            if "CHANNEL" in i.name:
                self.ebins["INDEX"] = energies[i.name]
            elif "E" in i.name:
                self.ebins[i.name] = u.Quantity(energies[i.name], i.unit)

        # fill in the time info separately
        self.tbins = {}
        if "HDUCLAS4" in header.keys():
            # we have a PHA2 spectrum with multiple spectra for different time bins
            # for PHA files, "TIME" is always start and TIME_STOP/TSTOP is the end of time bin
            self.tbins["TIME_START"] = self.data["TIME"]
            try:
                self.tbins["TIME_STOP"] = self.data["TIME_STOP"]
            except KeyError as e:
                self.tbins["TIME_STOP"] = self.data["TSTOP"]
            self.tbins["TIME_CENT"] = 0.5 * (self.tbins["TIME_START"] + self.tbins["TIME_STOP"])
        else:
            # if there is only 1 time bin (and one spectrum) this is sufficient
            # see https://heasarc.gsfc.nasa.gov/ftools/caldb/help/batbinevt.html
            # also convert START/STOP to TIME_START/TIME_STOP for consistency between classes
            for i in times.columns:
                self.tbins[f"TIME_{i.name}"] = u.Quantity(times[i.name], i.unit)
            self.tbins["TIME_CENT"] = 0.5 * (self.tbins[f"TIME_START"] + self.tbins[f"TIME_STOP"])

        # see if there is a response file associated with this and that it exists
        if "RESPFILE" in header.keys():
            drm_file = header["RESPFILE"]
            if drm_file.lower() == "none":
                self.drm_file = None
            else:
                drm_file = pha_file.parent.joinpath(header["RESPFILE"])
                if not drm_file.exists():
                    self.drm_file = None
                    warnings.warn(
                        f"The drm file {drm_file} does not seem to exist. Setting to None and continuing to parse the pha file.",
                        stacklevel=2,
                    )
                else:
                    self.drm_file = drm_file

        # if self.pha_input_dict ==None, then we will need to try to read in the hisotry of parameters passed into
        # batbinevt to create the pha file. thsi usually is needed when we first parse a file so we know what things
        # are if we need to do some sort of rebinning.

        # were looking for something like:
        #   START PARAMETER list for batbinevt_1.48 at 2023-11-16T18:47:52
        #
        #   P1 infile = /Users/tparsota/Documents/01116441000_eventresult/events/sw0
        #   P1 1116441000bevshsp_uf.evt
        #   P2 outfile = /Users/tparsota/Documents/01116441000_eventresult/pha/spect
        #   P2 rum_0.pha
        #   P3 outtype = PHA
        #   P4 timedel = 0.0
        #   P5 timebinalg = uniform
        #   P6 energybins = CALDB
        #   P7 gtifile = NONE
        #   P8 ecol = ENERGY
        #   P9 weighted = YES
        #   P10 outunits = INDEF
        #   P11 timepixr = -1.0
        #   P12 maskwt = NONE
        #   P13 tstart = INDEF
        #   P14 tstop = INDEF
        #   P15 snrthresh = 6.0
        #   P16 detmask = /Users/tparsota/Documents/01116441000_eventresult/auxil/sw
        #   P16 01116441000bdqcb.hk.gz
        #   P17 tcol = TIME
        #   P18 countscol = DPH_COUNTS
        #   P19 xcol = DETX
        #   P20 ycol = DETY
        #   P21 maskwtcol = MASK_WEIGHT
        #   P22 ebinquant = 0.1
        #   P23 delzeroes = no
        #   P24 minfracexp = 0.1
        #   P25 min_dph_frac_overlap = 0.999
        #   P26 min_dph_time_overlap = 0.0
        #   P27 max_dph_time_nonoverlap = 0.5
        #   P28 buffersize = 16384
        #   P29 clobber = yes
        #   P30 chatter = 2
        #   P31 history = yes
        #   P32 mode = ql
        #   END PARAMETER list for batbinevt_1.48

        if self.pha_input_dict is None:
            # get the default names of the parameters for batbinevt including its name 9which should never change)
            test = hsp_core.HSPTask('batbinevt')
            default_params_dict = test.default_params.copy()
            taskname = test.taskname
            start_processing = None

            for i in header["HISTORY"]:
                if taskname in i and start_processing is None:
                    # then set a switch for us to start looking at things
                    start_processing = True
                elif taskname in i and start_processing is True:
                    # we want to stop processing things
                    start_processing = False

                if start_processing and "START" not in i and len(i) > 0:
                    values = i.split(" ")
                    # print(i, values, "=" in values)

                    parameter_num = values[0]
                    parameter = values[1]
                    if "=" not in values:
                        # this belongs with the previous parameter and is a line continuation
                        default_params_dict[old_parameter] = default_params_dict[old_parameter] + values[-1]
                        # assume that we need to keep appending to the previous parameter
                    else:
                        default_params_dict[parameter] = values[-1]

                        old_parameter = parameter

            self.pha_input_dict = default_params_dict.copy()

    def _create_custom_timebins(self, timebins, output_file=None):
        """
        This method creates custom time bins from a user defined set of time bin edges. The created fits file with the
        timebins of interest will by default have the same name as the pha file, however it will have a "gti"
        suffix instead of a "pha" suffix and it will be stored in the gti subdirectory of the event results directory.

        Note: This method is here so the call to create a gti file with custom timebins can be phased out eventually.

        :param timebins: a astropy.unit.Quantity object with the edges of the timebins that the user would like
        :param output_file: None or a Path object to where the output *.gti file will be saved to. A value of None
            defaults to the above description
        :return: Path object of the created good time intervals file
        """

        if output_file is None:
            # use the same filename as for the lightcurve file but replace suffix with gti and put it in gti subdir
            # instead of lc
            pha_file = self.pha_file
            new_path = pha_file.parts
            new_name = pha_file.name.replace("pha", "gti")

            output_file = Path(*new_path[:pha_file.parts.index('pha')]).joinpath("gti").joinpath(new_name)

        return create_gti_file(timebins, output_file, T0=None, is_relative=False, overwrite=True)

    @u.quantity_input(emin=['energy'], emax=['energy'])
    def plot(self, emin=15 * u.keV, emax=150 * u.keV, plot_model=True):
        """
        This method allows the user to conveniently plot the spectrum that has been created. If it has been fitted
        with a model, then the model can also be plotted as well.

        :param emin: an astropy.unit.Quantity denoting the min energy that should be plotted
        :param emax: an astropy.unit.Quantity denoting the max energy that should be plotted
        :param plot_model: Boolean to denote if the model that has been fit to the pha file should also be plotted, if
            it exists
        :return: matplotlib figure, matplotlib axis
        """

        # calculate the center of the energy bin
        ecen = 0.5 * (self.ebins["E_MIN"] + self.ebins["E_MAX"])

        # get where the energy is >15 keV and <195 keV
        if emin is not None and emax is not None:
            energy_idx = np.where((self.ebins["E_MIN"] >= emin) & (self.ebins["E_MAX"] < emax))
        else:
            energy_idx = np.where((self.ebins["E_MIN"] > -1 * np.inf) & (self.ebins["E_MAX"] < np.inf))

        # calculate error including both systematic error and statistical error, note that systematic error has
        # been multiplied by the rates/counts in the _parse_pha method
        tot_error = np.sqrt(self.data["STAT_ERR"].value ** 2 + self.data["SYS_ERR"].value ** 2)

        # get the quantity to be plotted
        if "RATE" in self.data.keys():
            plot_data = self.data["RATE"]
        else:
            plot_data = self.data["COUNTS"]

        fig, ax = plt.subplots(1)
        ax.loglog(self.ebins["E_MIN"][energy_idx], plot_data[energy_idx], color="k", drawstyle="steps-post")
        ax.loglog(self.ebins["E_MAX"][energy_idx], plot_data[energy_idx], color="k", drawstyle="steps-pre")
        ax.errorbar(
            ecen[energy_idx],
            plot_data[energy_idx],
            yerr=tot_error[energy_idx] * plot_data.unit,
            color="k",
            marker="None",
            ls="None",
            label=f"Event Data Spectrum\nt={self.tbins['TIME_START'].value[0]}-{self.tbins['TIME_STOP'][0]}",
        )

        if "RATE" in self.data.keys():
            ax.set_ylabel("Count Rate (ct/s)", fontsize=14)
        else:
            ax.set_ylabel("Counts (ct)", fontsize=14)
        ax.set_xlabel("E (keV)", fontsize=14)

        ax.tick_params(axis="both", which="major", labelsize=14)

        # if there is a fitted model need to get that and plot it
        if self.spectral_model is not None and plot_model:
            model_emin = self.spectral_model["ebins"]["E_MIN"]
            model_emax = self.spectral_model["ebins"]["E_MAX"]

            # get where the energy is >15 keV and <195 keV
            if emin is not None and emax is not None:
                energy_idx = np.where((model_emin >= emin) & (model_emax < emax))
            else:
                energy_idx = np.where((model_emin > -1 * np.inf) & (model_emax < np.inf))

            model = self.spectral_model["data"]["model_spectrum"][energy_idx]

            ax.loglog(model_emin[energy_idx], model, color="r", drawstyle="steps-post")
            ax.loglog(model_emax[energy_idx], model, color="r", drawstyle="steps-pre", label="Folded Model")

        ax.legend(loc="best")

        return fig, ax

    def calculate_drm(self):
        """
        This function calculates the detector response matrix for the created PHA file.

        This is formatted this way so in the future, the _call_batdrmgen() method which relies on heasoft can be
        modified to use python native code

        :return: heasoftpy result object for the batdrmgen heasoft call
        """

        # return self._call_batdrmgen()
        return BatDRM.calc_drm(self.pha_file)

    @property
    def ra(self):
        """The right ascension of the source and the associated weighting assigned to the event file to produce the spectrum"""
        return self._ra

    @ra.setter
    @u.quantity_input
    def ra(self, value: u.Quantity[u.deg] | None):
        self._ra = value

    @property
    def dec(self):
        """The declination of the source and the associated weighting assigned to the event file to produce the spectrum"""
        return self._dec

    @dec.setter
    @u.quantity_input
    def dec(self, value: u.Quantity[u.deg] | None):
        self._dec = value

    @property
    def drm_file(self):
        """
        The detector response function file.

        :return: Path object of the DRM file
        """
        if self._drm_file is None:
            self._drm_file = self.calculate_drm()
            self.drm = BatDRM.from_file(drm_file=self._drm_file)

        return self._drm_file

    @drm_file.setter
    def drm_file(self, value):
        if not isinstance(value, Path) and value is not None:
            raise ValueError("drm_file can only be set to None or a path object")

        if value is not None and not value.exists():
            raise ValueError(f"The file {value} does not seem to exist")

        # if a drm_file is set and it exists, will also need to potentially modify the header value for "RESPFILE"
        # need to ensure that it is in the same directory as the pha file, put a check here
        if value is not None and not self.pha_file.is_relative_to(value.parent):
            raise ValueError(
                f"The response file {value} needs to be in the same directory as the pha file {self.pha_file}")

        # not setting the _drm_file to None, also dont want to modify the header keyword if they are the same file
        with fits.open(self.pha_file) as pha_hdulist:
            rsp_file = pha_hdulist[1].header["RESPFILE"]

        if value is not None and rsp_file.lower() != value.name.lower():
            # if it is in the same directory, then modify the header value
            with fits.open(self.pha_file, mode="update") as pha_hdulist:
                header = pha_hdulist[1].header

                header["RESPFILE"] = value.name
                pha_hdulist.flush()

        self._drm_file = value
        if value is not None:
            self.drm = BatDRM.from_file(drm_file=value)

    @property
    def drm(self):
        """
        The BAT detector response object associated with the drm_file that is specified
        """
        return self._drm

    @drm.setter
    def drm(self, value):
        if not isinstance(value, BatDRM) and value is not None:
            raise ValueError("The drm attribute needs to be set to a BatDRM object or None")
        else:
            self._drm = value

    @property
    def pha_file(self):
        """
        The pulse height amplitude file.

        :return: Path object of the DRM file
        """

        return self._pha_file

    @pha_file.setter
    def pha_file(self, value):
        if not isinstance(value, Path):
            raise ValueError("pha_file can only be set to a path object")

        # if not value.exists():
        #    raise ValueError(f"The file {value} does not seem to exist")

        self._pha_file = value

    def calc_upper_limit(self, bkg_nsigma=5):
        """
        This method creates the N sigma upper limits for the spectrum

        NEED TO DOUBLE CHECK THIS
        :param bkg_nsigma: Float for the significance of the background scaling to obtain an upper limit at that limit
            (eg PHA count = bkg_nsigma*bkg_var), here

        :return: a Spectrum object with the upperlimit calculated pha file
        """
        try:
            pha_file = self.pha_file
        except ValueError as e:
            print(e)
            raise ValueError("There is no PHA file from which upper limits can be calculated.")

        # calculate error including both systematic error and statistical error, note that systematic error has
        # been multiplied by the rates/counts in the _parse_pha method
        tot_error = np.sqrt(self.data["STAT_ERR"].value ** 2)  # + self.data["SYS_ERR"].value ** 2)

        # modify the filename
        upperlimit_pha_file = pha_file.parent.joinpath(
            f"{pha_file.stem}_bkgnsigma_{int(bkg_nsigma)}_upperlim{pha_file.suffix}")

        # copy the pha file to the new filename and then modify the values
        shutil.copy(pha_file, upperlimit_pha_file)

        # modify the upper limits file with the appropriate "rate" values
        with fits.open(upperlimit_pha_file, mode="update") as pha_hdulist:
            spectrum_cols = [i.name for i in pha_hdulist["SPECTRUM"].data.columns]
            if "RATE" in spectrum_cols:
                val = "RATE"
            else:
                val = "COUNTS"
            pha_hdulist["SPECTRUM"].data[val] = bkg_nsigma * tot_error
            pha_hdulist["SPECTRUM"].data["STAT_ERR"] = np.zeros_like(tot_error)

            pha_hdulist.flush()

        return self.from_file(upperlimit_pha_file, self.event_file, self.detector_quality_file,
                              self.auxil_raytracing_file)

    @classmethod
    def from_file(cls, pha_file, event_file=None, detector_quality_file=None, auxil_raytracing_file=None):
        """
        This class method takes an existing PHA file and returns a Spectrum class object with the data contained in the
        PHA file. The user will be able to plot the spectrum, calculate the detector response matrix for the file (if it
        doesnt exist already), and fit the spectrum. If the event file, the detector quality mask, or the auxiliary ray
        tracing files are not specified, then the user will not be able to dynamically change the spectrum energy bins
        or time bin

        :param pha_file: Path object of the pha file that will be read in.
        :param event_file: Path object for the event file with mask weighting already applied so we can load the
            appropriate mask weights
        :param detector_quality_file: Path object for the detector quality mask that was constructed for the associated
            event file
        :param auxil_raytracing_file: Path object pointing to the auxiliary ray tracing file that is created by applying
            the mask weighting to the event file that is passed in.
        :return: a Spectrum object with the loaded pha file data
        """
        pha_file = Path(pha_file).expanduser().resolve()

        if not pha_file.exists():
            raise ValueError(f"The specified PHA file {pha_file} does not seem to exist. "
                             f"Please double check that it does.")

        # also make sure that the file is gunzipped which is necessary the user wants to do spectral fitting
        if ".gz" in pha_file.suffix:
            with gzip.open(pha_file, 'rb') as f_in:
                with open(pha_file.parent.joinpath(pha_file.stem), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            pha_file = pha_file.parent.joinpath(pha_file.stem)

        return cls(pha_file, event_file, detector_quality_file, auxil_raytracing_file)
