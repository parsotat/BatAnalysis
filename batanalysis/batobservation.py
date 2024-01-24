"""
This file contains the batobservation class which contains information pertaining to a given bat observation.

Tyler Parsotan Jan 24 2022
"""
import os
import shutil
import sys
from .batlib import datadir, dirtest, met2mjd, met2utc, create_gti_file, calc_response
import glob
from astropy.io import fits
import numpy as np
import subprocess
import pickle
import sys
import re
from pathlib import Path
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta
import re
import warnings
import matplotlib.pyplot as plt

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


# try:
# import xspec as xsp
# except ModuleNotFoundError as err:
# Error handling
# print(err)


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
                self.obs_dir = obs_dir.joinpath(self.obs_id)  # os.path.join(obs_dir , self.obs_id)
            else:
                raise FileNotFoundError(
                    'The directory %s does not contain the observation data corresponding to ID: %s' % (
                    obs_dir, self.obs_id))
        else:
            obs_dir = datadir()  # Path.cwd()

            if obs_dir.joinpath(self.obs_id).is_dir():
                # os.path.isdir(os.path.join(obs_dir , self.obs_id)):
                self.obs_dir = obs_dir.joinpath(self.obs_id)  # self.obs_dir = os.path.join(obs_dir , self.obs_id)
            else:
                raise FileNotFoundError(
                    'The directory %s does not contain the observation data correponding to ID: %s' % (
                    obs_dir, self.obs_id))

    def _set_local_pfile_dir(self, dir):
        """
        make the local pfile dir if it doesnt exist and set this value

        :return: None
        """
        # make sure that it is a Path object
        self._local_pfile_dir = Path(dir)

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

    def _call_batbinevt(self, input_dict):
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
        Calls heasoftpy's batmaskwtevt with an error wrapper,
        TODO: apply keyword correction for spectrum file (using the auxfile) via batupdatephakw

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


class Lightcurve(BatObservation):
    """
    This is a general light curve class that contains typical information that a user may want from their lightcurve.
    This object is a wrapper around a light curve created from BAT event data.

    TODO: make this flexible enough to read in the raw rates lightcurves if necessary
    """

    def __init__(self, lightcurve_file, event_file, detector_quality_mask, ra=None, dec=None, lc_input_dict=None,
                 recalc=False, mask_weighting=True):
        """
        This constructor either creates a lightcurve fits file based off of a passed in event file where mask weighting
        has been applied and the detector quality mask has been constructed. Alternatively, this method can read in a
        previously calculated lightcurve. If recalc=True, then the lightcuve can be recalculated using the passed in
        lc_input_dict or a default input_dict defined as:

        dict(infile=str(event_file), outfile=str(lightcurve_file), outtype="LC",
                              energybins="15-350", weighted="YES", timedel=0.064,
                              detmask=str(detector_quality_mask),
                              tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")

        The ra/dec of the source that this lightcurve was constructed for (and for which the weighting was applied to
        the event file), can be specified or it can be dynamically read from the lightcurve file.

        :param event_file: Path object for the event file with mask weighting already applied, from which we will construct
            the lightcurve or read the previously ocnstructed lightcurve file
        :param lightcurve_file: path object of the lightcurve file that will be read in, if previously calculated,
            or the location/name of the new lightcurve file that will contain the newly calculated lightcurve.
        :param detector_quality_mask: Path object for the detector quality mask that was constructed for the associated
            event file
        :param ra: None or float representing the decimal degree RA value of the source for which the mask weighting was
            applied to the passed in event file. A value of None indicates that the RA of the source will be obtained
            from the calculated lightcurve which is then saved to lightcurve_file
        :param dec: None or float representing the decimal degree DEC value of the source for which the mask weighting was
            applied to the passed in event file. A value of None indicates that the DEC of the source will be obtained
            from the calculated lightcurve which is then saved to lightcurve_file
        :param lc_input_dict: None or a dict of values that will be passed to batbinevt in the creation of the lightcurve.
            If a lightcurve is being read in from one that was previously created, the prior parameters that were used to
            calculate the lightcurve will be used.
            If lc_input_dict is None, this will be set to:
                dict(infile=str(event_file), outfile=str(lightcurve_file), outtype="LC",
                                  energybins="15-350", weighted="YES", timedel=0.064,
                                  detmask=str(detector_quality_mask),
                                  tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")
            See HEASoft docmentation on batbinevt to see what these parameters mean. Alternatively, these defaults can
            be used in the inital call and time/energy rebinning can be done using the set_timebins and set_energybins
            methods associated with the Lightcurve object.
        :param recalc: Boolean to denote if the lightcurve specified by lightcurve_file should be recalculated with the
            lc_input_dict values (either those passed in or those that are defined by default)
        :param mask_weighting: Boolean to denote if mask weighting should be applied. By default this is set to True,
            however if a source if out of the BAT field of view the mask weighting will produce a lightcurve of 0 counts.
            Setting mask_weighting=False in this case ignores the position of the source and allows the pure rates/counts
            to be calculated.
        """
        # NOTE: Lots of similarities here as with the spectrum since we are using batbinevt as the base. If there are any
        # issues with the lightcurve object, then we should make sure that these same problems do not occur in the
        # spectrum object and vice versa

        # save these variables
        self.event_file = Path(event_file).expanduser().resolve()
        self.lightcurve_file = Path(lightcurve_file).expanduser().resolve()
        self.detector_quality_mask = Path(detector_quality_mask).expanduser().resolve()

        # error checking for weighting
        if type(mask_weighting) is not bool:
            raise ValueError("The mask_weighting parameter should be a boolean value.")

        # need to see if we have to construct the lightcurve if the file doesnt exist
        if not self.lightcurve_file.exists() or recalc:
            # see if the input dict is None so we can set these defaults, otherwise save the requested inputs for use later
            if lc_input_dict is None:
                self.lc_input_dict = dict(infile=str(self.event_file), outfile=str(self.lightcurve_file), outtype="LC",
                                          energybins="15-350", weighted="YES", timedel=0.064,
                                          detmask=str(self.detector_quality_mask),
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
        self.lc_ra = ra
        self.lc_dec = dec

        # read info from the lightcurve file
        self._parse_lightcurve_file()

        # read in the information about the weights
        self._get_event_weights()

        # set the duration info to None for now until the user calls battblocks
        self.tdurs = None

        # were done getting all the info that we need. From here, the user can rebin the timebins and the energy bins

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
        :param tmax:astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
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
        :param snrthresh: float representing the snr threshold associated with the timebinalg="snr" or timebinalg="highsnr"
            parameter values. See above description of the timebinalg parameter to see how this snrthresh parameter is used.
        :param save_durations: Boolean switch denoting if the T90, T50, and other durations calculated by the battblocks
            algorithm should be saved. If they are, this information will be located in the tdurs attribute. This calculation
            is only possible if timebinalg="bayesian". (More information can be found from the battblocks HEASoft documentation)
        :return: None
        """

        # create a temp copy incase the time rebinning doesnt complete successfully
        tmp_lc_input_dict = self.lc_input_dict.copy()

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

    @u.quantity_input(emin=['energy'], emax=['energy'])
    def set_energybins(self, energybins=["15-25", "25-50", "50-100", "100-350"], emin=None, emax=None,
                       calc_energy_integrated=True):
        """
        This method allows the energy binning to be set for the lightcurve. The energy rebinning is done automatically
        and the information for the rebinned lightcurve is automatically updated in the data attribute (which holds the
        light curve information itself including rates/counts, errors, fracitonal exposure, total counts, etc) and the
        ebins attribute which holds the energybins associated with the lightcurve.

        :param energybins: a list or single string denoting the energy bins in keV that the lightcurve shoudl be binned into
            The string should be formatted as "15-25" where the dash is necessary. A list should be formatted as multiple
            elements of the strings, where none of the energy ranges overlap.
        :param emin: a list or a astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the user would like. NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param emax: a list or a astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            energy bins that the user would like. It shoudl have the same number of elements as emin.
            NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param calc_energy_integrated: Boolean to denote wether the energy integrated light curve should be calculated
            based off the min and max energies that were passed in. If a single energy bin is requested for the rebinning
            then this argument does nothing.
        :return: None.
        """

        # error checking for calc_energy_integrated
        if type(calc_energy_integrated) is not bool:
            raise ValueError("The calc_energy_integrated parameter should be a boolean value.")

        # see if the user specified either the energy bins directly or emin/emax separately
        if emin is None and emax is None:
            # make sure that energybins is a list
            if type(energybins) is not list:
                energybins = [energybins]

            # verify that all elements are strings
            for i in energybins:
                if type(i) is not str:
                    raise ValueError(
                        'All elements of the passed in energybins variable must be a string. Please make sure this condition is met.')

            # need to get emin and emax values, assume that these are in keV already when converting to astropy quantities
            emin = []
            emax = []
            for i in energybins:
                energies = i.split('-')
                emin.append(float(energies[0]))
                emax.append(float(energies[1]))
            emin = u.Quantity(emin, u.keV)
            emax = u.Quantity(emax, u.keV)

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
                energybins = []
                for min, max in zip(emin.to(u.keV), emax.to(u.keV)):
                    energybins.append(f"{min.value}-{max.value}")
            else:
                energybins = [f"{emin.to(u.keV).value}-{emax.to(u.keV).value}"]

        # create the full string
        ebins = ','.join(energybins)

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

        # error checking for calc_energy_integrated
        if type(calc_energy_integrated) is not bool:
            raise ValueError("The calc_energy_integrated parameter should be a boolean value.")

        with fits.open(self.lightcurve_file) as f:
            header = f[1].header
            data = f[1].data
            energies = f["EBOUNDS"].data
            energies_header = f["EBOUNDS"].header

        if self.lc_ra is None and self.lc_dec is None:
            self.lc_ra = header["RA_OBJ"]
            self.lc_dec = header["DEC_OBJ"]
        else:
            # test if the passed in coordinates are what they should be for the light curve file
            # TODO: see if we are ~? arcmin close to one another
            assert (np.isclose(self.lc_ra, header["RA_OBJ"]) and np.isclose(self.lc_dec, header["DEC_OBJ"])), \
                f"The passed in RA/DEC values ({self.lc_ra},{self.lc_dec}) do not match the values used to produce the lightcurve which are ({header['RA_OBJ']},{header['DEC_OBJ']})"

        # read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy arrays as values
        self.data = {}
        for i in data.columns:
            self.data[i.name] = u.Quantity(data[i.name], i.unit)

        # fill in the energy bin info
        self.ebins = {}
        for i in energies.columns:
            if "CHANNEL" in i.name:
                self.ebins["INDEX"] = energies[i.name]
            elif "E" in i.name:
                self.ebins[i.name] = u.Quantity(energies[i.name], i.unit)

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

        if self.lc_input_dict is None:
            # get the default names of the parameters for batbinevt including its name 9which should never change)
            test = hsp.HSPTask('batbinevt')
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

    def _get_event_weights(self):
        """
        This method reads in the appropriate weights for event data once it has been applied to a event file, for a
        given RA/DEC position. This should only need to be done once, when the user has applied mask weighting
        in the BatEvent object and is creating a lightcurve.

        :return: None
        """

        # read in all the info for the weights and save it such that we can use these weights in the future for
        # redoing lightcurve calculation
        with fits.open(self.event_file) as file:
            self._event_weights = file[1].data["MASK_WEIGHT"]

    def _set_event_weights(self):
        """
        This method sets the appropriate weights for event data, for a
        given RA/DEC position. The weights are rewritten to the event file in the "MASK_WEIGHT" column.
        This may be necessary if a user is analyzing multiple sources for which event data has been
        obtained.

        Note: event weightings need to be set if the RA/DEC of the light curve doesnt match what is in the event file

        :return: None
        """

        if not self._same_event_lc_coords():
            # read in the event file and replace the values in the MASK_WEIGHT with the appropriate values in self._event_weights
            with fits.open(self.event_file, mode="update") as file:
                file[1].data["MASK_WEIGHT"] = self._event_weights
                file.flush()

    def _same_event_lc_coords(self):
        """
        This method reads in the event data coordinates and compares it to what is obained from the lightcurve
        file that has been loaded in.

        :return: Boolean
        """

        with fits.open(self.event_file) as file:
            event_ra = file[0].header["RA_OBJ"]
            event_dec = file[0].header["DEC_OBJ"]
            coord_match = (event_ra == self.lc_ra) and (event_dec == self.lc_dec)

        return coord_match

    def _calc_energy_integrated(self):
        """
        This method just goes though the count rates in each energy bin that has been precalulated and adds them up.
        It also calcualtes the errors. These arrays are added to self.data appropriately and the total energy min/max is
        added to self.ebins

        NOTE: This method can be expanded in the future to be an independent way of rebinning the lightcurves in
            energy without any calls to heasoftpy

        :return: None
        """

        # if we have more than 1 energy bin then we can calculate an energy integrated count rate, etc
        # otherwise we dont have to do anything since theres only one energy bin
        if self.data["RATE"].ndim > 1:
            # calculate the total count rate and error
            integrated_count_rate = self.data["RATE"].sum(axis=1)
            integrated_count_rate_err = np.sqrt(np.sum(self.data["ERROR"] ** 2, axis=1))

            # get the energy info
            min_e = self.ebins["E_MIN"].min()
            max_e = self.ebins["E_MAX"].max()
            max_energy_index = self.ebins["INDEX"].max()

            # append energy integrated count rate, the error, and the additional energy bin to the respective dicts
            new_energy_bin_size = self.ebins["INDEX"].size + 1
            new_e_index = np.arange(new_energy_bin_size, dtype=self.ebins["INDEX"].dtype)

            new_emin = np.zeros(new_energy_bin_size) * self.ebins["E_MIN"].unit
            new_emin[:-1] = self.ebins["E_MIN"]
            new_emin[-1] = min_e

            new_emax = np.zeros_like(new_emin)  # the zeros_like gets the units from the array that is passed in
            new_emax[:-1] = self.ebins["E_MAX"]
            new_emax[-1] = max_e

            new_rate = np.zeros((self.data["RATE"].shape[0], new_energy_bin_size)) * self.data["RATE"].unit
            new_rate_err = np.zeros_like(new_rate)

            new_rate[:, :-1] = self.data["RATE"]
            new_rate[:, -1] = integrated_count_rate

            new_rate_err[:, :-1] = self.data["ERROR"]
            new_rate_err[:, -1] = integrated_count_rate_err

            # save the updated arrays
            self.ebins["INDEX"] = new_e_index
            self.ebins["E_MIN"] = new_emin
            self.ebins["E_MAX"] = new_emax

            self.data["RATE"] = new_rate
            self.data["ERROR"] = new_rate_err

    def plot(self, energybins=None, plot_counts=False, plot_exposure_fraction=False, time_unit="MET", T0=None,
             plot_relative=False):
        """
        This convenience method allows the user to plot the rate/count lightcurve.

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
        num_plots = 1
        num_plots += (plot_counts + plot_exposure_fraction)
        fig, ax = plt.subplots(num_plots, sharex=True)

        # assign the axes for each type of plot we may want
        axes_queue = [i for i in range(num_plots)]

        if num_plots > 1:
            ax_rate = ax[axes_queue[0]]
            axes_queue.pop(0)

            if plot_counts:
                ax_count = ax[axes_queue[0]]
                axes_queue.pop(0)

            if plot_exposure_fraction:
                ax_exposure = ax[axes_queue[0]]
                axes_queue.pop(0)
        else:
            ax_rate = ax

        # plot everything for the rates by default
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
                    rate = self.data["RATE"][:, e_idx]
                    rate_error = self.data["ERROR"][:, e_idx]
                else:
                    rate = self.data["RATE"]
                    rate_error = self.data["ERROR"]

                line = ax_rate.plot(start_times, rate, ds='steps-post')
                ax_rate.plot(end_times, rate, ds='steps-pre', color=line[-1].get_color())
                ax_rate.errorbar(mid_times, rate, yerr=rate_error, ls='None', color=line[-1].get_color())

        if num_plots > 1:
            ax_rate.legend()

        if plot_counts:
            line = ax_count.plot(start_times, self.data["TOTCOUNTS"], ds='steps-post')
            ax_count.plot(end_times, self.data["TOTCOUNTS"], ds='steps-pre', color=line[-1].get_color())
            ax_count.set_ylabel('Total counts (ct)')

        if plot_exposure_fraction:
            line = ax_exposure.plot(start_times, self.data["FRACEXP"], ds='steps-post')
            ax_exposure.plot(end_times, self.data["FRACEXP"], ds='steps-pre', color=line[-1].get_color())
            ax_exposure.set_ylabel('Fractional Exposure')

        if T0 is not None and not plot_relative:
            # plot the trigger time for all panels if we dont want the plotted times to be relative
            if num_plots > 1:
                for axis in ax:
                    axis.axvline(T0, 0, 1, ls='--', label=f"T0={T0:.2f}", color='k')
            else:
                ax_rate.axvline(T0, 0, 1, ls='--', label=f"T0={T0:.2f}", color='k')

        if num_plots > 1:
            ax[1].legend()
            ax[-1].set_xlabel(xlabel)
        else:
            ax_rate.legend()
            ax_rate.set_xlabel(xlabel)

        if "RATE" in self.data.keys():
            ax_rate.set_ylabel('Rate (ct/s)')
        else:
            ax_rate.set_ylabel('Counts (ct)')

        # fig.savefig("test.pdf")
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
        test = hsp.HSPTask('battblocks')
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


class Spectrum(BatObservation):
    def __init__(self, pha_file, event_file, detector_quality_mask, auxil_raytracing_file, ra=None, dec=None,
                 pha_input_dict=None, mask_weighting=True, recalc=False):
        """

        :param pha_file:
        :param event_file:
        :param detector_quality_mask:
        :param auxiliary_file:
        :param ra:
        :param dec:
        :param pha_input_dict:
        :param recalc:
        :param mask_weighting:
        """
        # NOTE: Lots of similarities here as with the lightcurve since we are using batbinevt as the base. If there are any
        # issues with the lightcurve object, then we should make sure that these same problems do not occur in the
        # spectrum object and vice versa

        # save these variables
        self.pha_file = Path(pha_file).expanduser().resolve()
        self.event_file = Path(event_file).expanduser().resolve()
        self.detector_quality_mask = Path(detector_quality_mask).expanduser().resolve()
        self.auxil_raytracing_file = Path(auxil_raytracing_file).expanduser().resolve()
        self.drm_file_list = None
        self.spectral_model = None

        # need to see if we have to construct the lightcurve if the file doesnt exist
        if not self.pha_file.exists() or recalc:
            # see if the input dict is None so we can set these defaults, otherwise save the requested inputs for use later
            if pha_input_dict is None:
                # energybins = "CALDB" gives us the 80 channel spectrum
                self.pha_input_dict = dict(infile=str(self.event_file), outfile=str(self.pha_file), outtype="PHA",
                                           energybins="CALDB", weighted="YES", timedel=0.0,
                                           detmask=str(self.detector_quality_mask),
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

        # set default RA/DEC coordinates correcpondent to the LC file which will be filled in later if it is set to None
        self.lc_ra = ra
        self.lc_dec = dec

        # read info from the lightcurve file
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

        :param batbinevt_input_dict:
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
                     timedelta=np.timedelta64(64, 'ms'), snrthresh=None):
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
        :param tmax:astropy.units.Quantity denoting the maximum values of the timebin edges that the user would like
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
        :param snrthresh: float representing the snr threshold associated with the timebinalg="snr" or timebinalg="highsnr"
            parameter values. See above description of the timebinalg parameter to see how this snrthresh parameter is used.
        :return: None
        """

        # create a temp copy incase the time rebinning doesnt complete successfully
        tmp_pha_input_dict = self.pha_input_dict.copy()

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

    @u.quantity_input(emin=['energy'], emax=['energy'])
    def set_energybins(self, energybins="CALDB", emin=None, emax=None):
        """
        This method allows the energy binning to be set for the lightcurve. The energy rebinning is done automatically
        and the information for the rebinned lightcurve is automatically updated in the data attribute (which holds the
        light curve information itself including rates/counts, errors, fracitonal exposure, total counts, etc) and the
        ebins attribute which holds the energybins associated with the lightcurve.

        :param energybins: a list or single string denoting the energy bins in keV that the lightcurve shoudl be binned into
            The string should be formatted as "15-25" where the dash is necessary. A list should be formatted as multiple
            elements of the strings, where none of the energy ranges overlap.
        :param emin: a list or a astropy.unit.Quantity object of 1 or more elements. These are the minimum edges of the
            energy bins that the user would like. NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :param emax: a list or a astropy.unit.Quantity object of 1 or more elements. These are the maximum edges of the
            energy bins that the user would like. It shoudl have the same number of elements as emin.
            NOTE: If emin/emax are specified, the energybins parameter is ignored.
        :return: None.
        """

        # see if the user specified either the energy bins directly or emin/emax separately
        if emin is None and emax is None:

            # make sure that energybins is a list
            if type(energybins) is not list:
                energybins = [energybins]
                energybins = [i.capitalize for i in
                              energybins]  # do thsi for potential "CALDB" input, doesnt affect numbers

            if "CALDB" not in energybins:
                # verify that all elements are strings
                for i in energybins:
                    if type(i) is not str:
                        raise ValueError(
                            'All elements of the passed in energybins variable must be a string. Please make sure this condition is met.')

                # need to get emin and emax values, assume that these are in keV already when converting to astropy quantities
                emin = []
                emax = []
                for i in energybins:
                    energies = i.split('-')
                    emin.append(float(energies[0]))
                    emax.append(float(energies[1]))
                emin = u.Quantity(emin, u.keV)
                emax = u.Quantity(emax, u.keV)

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
                energybins = []
                for min_e, max_e in zip(emin.to(u.keV), emax.to(u.keV)):
                    energybins.append(f"{min_e.value}-{max_e.value}")
            else:
                energybins = [f"{emin.to(u.keV).value}-{emax.to(u.keV).value}"]

        # create the full string
        ebins = ','.join(energybins)

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

        :return:
        """
        pha_file = self.get_pha_filename()
        input_dict = dict(infile=str(pha_file), syserrfile="CALDB")

        try:
            return hsp.batphasyserr(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batphasyserr failed with inputs {input_dict}.")

    def _call_batupdatephakw(self):
        """
        Calls heasoftpy's batupdatephakw which applies geometrical corrections to the PHA spectrum which is especially
        important is BAT is slewing during an observation and the source position is changing.

        :return:
        """
        pha_file = self.get_pha_filename()
        input_dict = dict(infile=str(pha_file), auxfile=str(self.auxil_raytracing_file))

        try:
            return hsp.batupdatephakw(**input_dict)
        except Exception as e:
            print(e)
            raise RuntimeError(f"The call to Heasoft batupdatephakw failed with inputs {input_dict}.")

    def _call_batdrmgen(self):
        """
        This calls heasoftpy's batdrmgen which produces the associated drm for fitting the PHA file.

        :return:
        """

        pha_file=self.get_pha_filename()
        output = calc_response(pha_file)

        if output.returncode != 0:
            raise RuntimeError(f"The call to Heasoft batdrmgen failed with output {output.stdout}.")

        drm_file = pha_file.parent.joinpath(f"{pha_file.stem}.rsp")
        self.set_drm_filename(drm_file)


    def _get_event_weights(self):
        """
        This method reads in the appropriate weights for event data once it has been applied to a event file, for a
        given RA/DEC position. This should only need to be done once, when the user has applied mask weighting
        in the BatEvent object and is creating a lightcurve.

        :return: None
        """

        # read in all the info for the weights and save it such that we can use these weights in the future for
        # redoing lightcurve calculation
        with fits.open(self.event_file) as file:
            self._event_weights = file[1].data["MASK_WEIGHT"]

    def _set_event_weights(self):
        """
        This method sets the appropriate weights for event data, for a
        given RA/DEC position. The weights are rewritten to the event file in the "MASK_WEIGHT" column.
        This may be necessary if a user is analyzing multiple sources for which event data has been
        obtained.

        Note: event weightings need to be set if the RA/DEC of the light curve doesnt match what is in the event file

        :return: None
        """

        if not self._same_event_lc_coords():
            # read in the event file and replace the values in the MASK_WEIGHT with the appropriate values in self._event_weights
            with fits.open(self.event_file, mode="update") as file:
                file[1].data["MASK_WEIGHT"] = self._event_weights
                file.flush()

    def _same_event_lc_coords(self):
        """
        This method reads in the event data coordinates and compares it to what is obained from the lightcurve
        file that has been loaded in.

        :return: Boolean
        """

        with fits.open(self.event_file) as file:
            event_ra = file[0].header["RA_OBJ"]
            event_dec = file[0].header["DEC_OBJ"]
            coord_match = (event_ra == self.lc_ra) and (event_dec == self.lc_dec)

        return coord_match

    def _parse_pha_file(self):
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
        pha_file = self.get_pha_filename()
        with fits.open(pha_file) as f:
            header = f[1].header
            data = f[1].data
            energies = f["EBOUNDS"].data
            energies_header = f["EBOUNDS"].header
            times = f["STDGTI"].data

        if self.lc_ra is None and self.lc_dec is None:
            self.lc_ra = header["RA_OBJ"]
            self.lc_dec = header["DEC_OBJ"]
        else:
            # test if the passed in coordinates are what they should be for the light curve file
            # TODO: see if we are ~? arcmin close to one another
            assert (np.isclose(self.lc_ra, header["RA_OBJ"]) and np.isclose(self.lc_dec, header["DEC_OBJ"])), \
                (f"The passed in RA/DEC values ({self.lc_ra},{self.lc_dec}) "
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

        self.data["SYS_ERR"] = self.data["SYS_ERR"]*self.data[key]

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
            self.drm_file = drm_file
            if drm_file == "NONE":
                self.drm_file = None
            else:
                drm_file = pha_file.parent.joinpath(header["RESPFILE"])
                self.drm_file = drm_file
                if not drm_file.exists():
                    self.drm_file = None

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
            test = hsp.HSPTask('batbinevt')
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

            self.pha_input_dict = default_params_dict.copy()



    def _create_custom_timebins(self, timebins, output_file=None):
        """
        This method creates custom time bins from a user defined set of time bin edges. The created fits file with the
        timebins of interest will by default have the same name as the lightcurve file, however it will have a "gti"
        suffix instead of a "lc" suffix and it will be stored in the gti subdirectory of the event results directory.

        Note: This method is here so the call to create a gti file with custom timebins can be phased out eventually.

        :param timebins: a astropy.unit.Quantity object with the edges of the timebins that the user would like
        :param output_file: None or a Path object to where the output *.gti file will be saved to. A value of None
            defaults to the above description
        :return: Path object of the created good time intervals file
        """

        if output_file is None:
            # use the same filename as for the lightcurve file but replace suffix with gti and put it in gti subdir
            # instead of lc
            pha_file = self.get_pha_filename()
            new_path = pha_file.parts
            new_name = pha_file.name.replace("pha", "gti")

            output_file = Path(*new_path[:pha_file.parts.index('pha')]).joinpath("gti").joinpath(new_name)

        return create_gti_file(timebins, output_file, T0=None, is_relative=False, overwrite=True)

    def plot(self, emin=15*u.keV, emax=195*u.keV, plot_model=True):
        """
        This method allows the user to conveniently plot the spectrum that has been created. If it has been fitted
        with a model, then the model can also be plotted as well.

        :return: matplotlib figure, matplotlib axis
        """

        # calculate the center of the energy bin
        ecen = 0.5*(self.ebins["E_MIN"]+self.ebins["E_MAX"])

        # get where the energy is >15 keV and <195 keV
        if emin is not None and emax is not None:
            energy_idx = np.where((self.ebins["E_MIN"] >= emin) & (self.ebins["E_MAX"] < emax))
        else:
            energy_idx = np.where((self.ebins["E_MIN"] > -1*np.inf) & (self.ebins["E_MAX"] < np.inf))

        # calculate error including both systematic error and statistical error, note that systematic error has
        # been multiplied by the rates/counts in the _parse_pha method
        tot_error = np.sqrt(self.data["STAT_ERR"].value**2+self.data["SYS_ERR"].value**2)

        # get the quantity to be plotted
        if "RATE" in self.data.keys():
            plot_data = self.data["RATE"]
        else:
            plot_data = self.data["COUNTS"]

        fig, ax = plt.subplots(1)
        ax.loglog(self.ebins["E_MIN"][energy_idx], plot_data[energy_idx], color="k", drawstyle="steps-post")
        ax.loglog(self.ebins["E_MAX"][energy_idx],  plot_data[energy_idx], color="k", drawstyle="steps-pre")
        ax.errorbar(
            ecen[energy_idx],
            plot_data[energy_idx],
            yerr=tot_error[energy_idx]*plot_data.unit,
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
            model_emin=self.spectral_model["ebins"]["E_MIN"]
            model_emax=self.spectral_model["ebins"]["E_MAX"]

            # get where the energy is >15 keV and <195 keV
            if emin is not None and emax is not None:
                energy_idx = np.where((model_emin >= emin) & (model_emax < emax))
            else:
                energy_idx = np.where((model_emin > -1 * np.inf) & (model_emax < np.inf))

            model=self.spectral_model["data"]["model_spectrum"][energy_idx]


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

        return self._call_batdrmgen()

    def get_drm_filename(self):
        """
        This method returns the detector response function file

        :return: Path object of the DRM file
        """

        return self.drm_file


    def set_drm_filename(self, drmfile):
        """
        This funciton allows the pha_file_list attribute to have pha file names to be saved to it.
        The upper limit pha file can be overwritten but the original pha file cannot be changed.

        :param phafile:
        :return:
        """

        self.drm_file=drmfile

    def get_pha_filename(self):
        """
        This method returns the pha filename

        :param getupperlim: Boolean to specify if the function should return just the upper limit PHA file. Default is
            False, meaning that just the normal PHA file will be returned
        :return: a path object of the specified pha filename
        """

        return self.pha_file

    def set_pha_files(self, phafile):
        """
        This funciton allows the pha_file_list attribute to have pha file names to be saved to it.
        The upper limit pha file can be overwritten but the original pha file cannot be changed.

        :param phafile:
        :return:
        """

        self.pha_file=phafile

    def calc_upper_limit(self, bkg_nsigma=5):
        """
        This method creates the N sigma upper limits for the spectrum

        NEED TO DOUBLE CHECK THIS
        :param bkg_nsigma: Float for the significance of the background scaling to obtain an upper limit at that limit
            (eg PHA count = bkg_nsigma*bkg_var), here


        :return:
        """
        try:
            pha_file = self.get_pha_filename()
        except ValueError as e:
            print(e)
            raise ValueError("There is no PHA file from which upper limits can be calculated.")

        # calculate error including both systematic error and statistical error, note that systematic error has
        # been multiplied by the rates/counts in the _parse_pha method
        tot_error = np.sqrt(self.data["STAT_ERR"].value ** 2 + self.data["SYS_ERR"].value ** 2)

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


        return self.from_file(upperlimit_pha_file, self.event_file, self.detector_quality_mask,
                              self.auxil_raytracing_file)

    @classmethod
    def from_file(cls, pha_file, event_file, detector_quality_mask, auxil_raytracing_file):
        return cls(pha_file, event_file, detector_quality_mask, auxil_raytracing_file)

