"""
This file contains the batobservation class which contains information pertaining to a given bat observation.

Tyler Parsotan Jan 24 2022
"""
import os
import shutil
import sys
from .batlib import datadir, dirtest, met2mjd, met2utc, create_gti_file
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

    def __init__(self, event_file,  lightcurve_file, detector_quality_mask, ra=None, dec=None, lc_input_dict=None, recalc=False, mask_weighting=True):
        """
        This constructor reads in a fits file that contains light curve data for a given BAT event dataset. The fits file
        should have been created by a call to

        :param lightcurve_file:
        """

        #save these variables
        #TODO: make sure that they are Path objects
        self.event_file = event_file
        self.lightcurve_file = lightcurve_file
        self.detector_quality_mask = detector_quality_mask

        #error checking for weighting
        if type(mask_weighting) is not bool:
            raise ValueError("The mask_weighting parameter should be a boolean value.")


        #need to see if we have to construct the lightcurve if the file doesnt exist
        if not self.lightcurve_file.exists() or recalc:
            #see if the input dict is None so we can set these defaults, otherwise save the requested inputs for use later
            if lc_input_dict is None:
                self.lc_input_dict = dict(infile=str(self.event_file), outfile=str(self.lightcurve_file), outtype="LC",
                              energybins="15-350", weighted="YES", timedel=0.064,
                              detmask=str(self.detector_quality_mask),
                              tstart="INDEF", tstop="INDEF", clobber="YES", timebinalg="uniform")

                #specify if we want mask weighting
                if mask_weighting:
                    self.lc_input_dict["weighted"] = "YES"
                else:
                    self.lc_input_dict["weighted"] = "NO"

            else:
                self.lc_input_dict = lc_input_dict

            #create the lightcurve
            self.bat_lc_result = self._call_batbinevt(self.lc_input_dict)

            #make sure that this calculation ran successfully
            if self.bat_lc_result.returncode != 0:
                raise RuntimeError(f'The creation of the lightcurve failed with message: {lc.bat_lc_result.output}')

        else:
            #set the self.lc_input_dict = None so the parsing of the lightcurve tries to also load in the
            #parameters passed into batbinevt to create the lightcurve
            #try to parse the existing lightcurve file to see what parameters were passed to batbinevt to construct the file
            self.lc_input_dict = None

        #set default RA/DEC coordinates correcpondent to the LC file which will be filled in later if it is set to None
        self.lc_ra = ra
        self.lc_dec = dec

        #read info from the lightcurve file
        self._parse_lightcurve_file()

        #read in the information about the weights
        self._get_event_weights()

        #were done getting all the info that we need. From here, the user can rebin the timebins and the energy bins

    @u.quantity_input(timebins=['time'], tmin=['time'], tmax=['time'])
    def set_timebins(self, timebinalg="uniform", timebins=None, tmin=None, tmax=None, T0=None, is_relative=False,
                       timedelta=np.timedelta64(64, 'ms'), snrthresh=None, calc_energy_integrated=True):
        """
        This method allows for the dynamic rebinning of a light curve in time.

        timebin_alg
        TODO: make tmin/tmax also be able to take single times to denote min/max times to calc LC for
        :return:
        """

        #create a temp copy incase the time rebinning doesnt complete successfully
        tmp_lc_input_dict=self.lc_input_dict.copy()

        #error checking for calc_energy_integrated
        if type(is_relative) is not bool:
            raise ValueError("The is_relative parameter should be a boolean value.")

        #error checking for calc_energy_integrated
        if type(calc_energy_integrated) is not bool:
            raise ValueError("The calc_energy_integrated parameter should be a boolean value.")



        #see if the timebinalg is properly set approporiately
        #if "uniform" not in timebinalg or "snr" not in timebinalg or "bayesian" not in timebinalg:
        if timebinalg not in ["uniform", "snr", "highsnr", "bayesian"]:
            raise ValueError('The timebinalg only accepts the following values: uniform, snr, highsnr, and bayesian (for bayesian blocks).')

        #if timebinalg == uniform/snr/highsnr, make sure that we have a timedelta that is a np.timedelta object
        if "uniform" in timebinalg or "snr" in timebinalg:
            if type(timedelta) is not np.timedelta64:
                raise ValueError('The timedelta variable needs to be a numpy timedelta64 object.')

        #need to make sure that snrthresh is set for "snr" timebinalg
        if "snr" in timebinalg and snrthresh is None:
            raise ValueError(f'The snrthresh value should be set since timebinalg is set to be {snrthresh}.')

        #test if is_relative is false and make sure that T0 is defined
        if is_relative and T0 is None:
            raise ValueError('The is_relative value is set to True however there is no T0 that is defined '+
                             '(ie the time from which the time bins are defined relative to is not specified).')

        #if timebins, or tmin and tmax are defined then we ignore the timebinalg parameter
        #if tmin and tmax are specified while timebins is also specified then ignore timebins

        #do error checking on tmin/tmax
        if (tmin is None and tmax is not None) or (tmax is None and tmin is not None):
            raise ValueError('Both emin and emax must be defined.')

        if tmin is not None and tmax is not None:
            if len(tmin) != len(tmax):
                raise ValueError('Both tmin and tmax must have the same length.')

            #now try to construct single array of all timebin edges in seconds
            timebins=np.zeros(len(tmin)+1)*u.s
            timebins[:-1] = tmin
            timebins[-1] = tmax[-1]

        #See if we need to add T0 to everything
        if is_relative:
            #see if T0 is Quantity class
            if type(T0) is u.Quantity:
                timebins += T0
            else:
                timebins += T0 * u.s


        #if we are doing battblocks or the user has passed in timebins/tmin/tmax then we have to create a good time interval file
        # otherwise proceed with normal rebinning
        if "bayesian" in timebinalg:
            #we need to call battblocks to get the good time interval file
            self.timebins_file, battblocks_return = self._call_battblocks()
            tmp_lc_input_dict['timebinalg'] = "gti"
            tmp_lc_input_dict['gtifile'] = str(self.timebins_file)

        elif ((tmin is not None and tmax is not None) or timebins is not None):
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

            #stop

        #before doing the recalculation, make sure that the proper weights are in the event file
        self._set_event_weights()

        #the LC _call_batbinevt method ensures that  outtype = LC and that clobber=YES
        lc_return = self._call_batbinevt(tmp_lc_input_dict)

        #make sure that the lc_return was successful
        if lc_return.returncode != 0:
            raise RuntimeError(f'The creation of the lightcurve failed with message: {lc_return.output}')
        else:
            self.bat_lc_result = lc_return
            self.lc_input_dict = tmp_lc_input_dict

            #reparse the lightcurve file to get the info
            self._parse_lightcurve_file(calc_energy_integrated=calc_energy_integrated)






    def set_energybins(self, energybins=["15-25", "25-50", "50-100", "100-350"], emin=None, emax=None, calc_energy_integrated=True):
        """
        This method allows for the dynamic rebinning of a light curve in energy

        energybins cannot have any overlapping energybins eg cant have energybins=["15-25", "25-50", "50-100", "100-350", "15-350"]
        since the last energy bins overlaps with the others. (this is very inconvenient but can be fixed by adding things up)

        :return:
        """

        #error checking for calc_energy_integrated
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

            #need to get emin and emax values, assume that these are in keV already when converting to astropy quantities
            emin=[]
            emax=[]
            for i in energybins:
                energies=i.split('-')
                emin.append(float(energies[0]))
                emax.append(float(energies[1]))
            emin = u.Quantity(emin, u.keV)
            emax = u.Quantity(emax, u.keV)

        else:
            # make sure that both emin and emax are defined and have the same number of elements
            if (emin is None and emax is not None) or (emax is None and emin is not None):
                raise ValueError('Both emin and emax must be defined.')

            if len(emin) != len(emax):
                raise ValueError('Both emin and emax must have the same length.')

            # see if they are astropy quantity items with units
            if type(emin) is not u.Quantity:
                emin = u.Quantity(emin, u.keV)
            if type(emax) is not u.Quantity:
                emax = u.Quantity(emax, u.keV)

            # create our energybins input to batbinevt
            energybins = []
            for min, max in zip(emin.to(u.keV), emax.to(u.keV)):
                energybins.append(f"{min.value}-{max.value}")

        # create the full string
        ebins = ','.join(energybins)

        #create a temp dict to hold the energy rebinning parameters to pass to heasoftpy. If things dont run
        # successfully then the updated parameter list will not be saved
        tmp_lc_input_dict = self.lc_input_dict.copy()

        # need to see if the energybins are different (and even need to be calculated), if so do the recalculation
        if not np.array_equal(emin, self.ebins['E_MIN']) or not np.array_equal(emax, self.ebins['E_MAX']):
            #the tmp_lc_input_dict wil need to be modified with new Energybins
            tmp_lc_input_dict["energybins"]=ebins

            # before doing the recalculation, make sure that the proper weights are in the event file
            self._set_event_weights()

            #the LC _call_batbinevt method ensures that  outtype = LC and that clobber=YES
            lc_return = self._call_batbinevt(tmp_lc_input_dict)

            #make sure that the lc_return was successful
            if lc_return.returncode != 0:
                raise RuntimeError(f'The creation of the lightcurve failed with message: {lc_return.output}')
            else:
                self.bat_lc_result = lc_return
                self.lc_input_dict = tmp_lc_input_dict

                #reparse the lightcurve file to get the info
                self._parse_lightcurve_file(calc_energy_integrated=calc_energy_integrated)


    def _parse_lightcurve_file(self, calc_energy_integrated=True):
        """
        This method parses through a light curve file that has been created by batbinevent

        NOTE: A special value of timepixr=-1 (the default used when constructing light curves) specifies that
              timepixr=0.0 for uniformly binned light curves and
              timepixr=0.5 for non-unformly binned light curves. We recommend using the unambiguous TIME_CENT in the
              tbins attribute to prevent any confusion instead of the data["TIME"] values

        :return:
        """

        #error checking for calc_energy_integrated
        if type(calc_energy_integrated) is not bool:
            raise ValueError("The calc_energy_integrated parameter should be a boolean value.")


        with fits.open(self.lightcurve_file) as f:
            header=f[1].header
            data=f[1].data
            energies=f["EBOUNDS"].data
            energies_header=f["EBOUNDS"].header

        if self.lc_ra is None and self.lc_dec is None:
            self.lc_ra = header["RA_OBJ"]
            self.lc_dec = header["DEC_OBJ"]
        else:
            #test if the passed in coordinates are what they should be for the light curve file
            #TODO: see if we are ~? arcmin close to one another
            assert (np.isclose(self.lc_ra, header["RA_OBJ"]) and np.isclose(self.lc_dec, header["DEC_OBJ"])), \
                   f"The passed in RA/DEC values ({self.lc_ra},{self.lc_dec}) do not match the values used to produce the lightcurve which are ({header['RA_OBJ']},{header['DEC_OBJ']})"

        #read in the data and save to data attribute which is a dictionary of the column names as keys and the numpy arrays as values
        self.data={}
        for i in data.columns:
            self.data[i.name] = u.Quantity(data[i.name], i.unit)

        #fill in the energy bin info
        self.ebins={}
        for i in energies.columns:
            if "CHANNEL" in i.name:
                self.ebins["INDEX"] = energies[i.name]
            elif "E" in i.name:
                self.ebins[i.name]=u.Quantity(energies[i.name], i.unit)

        #fill in the time info separately
        timepixr=header["TIMEPIXR"]
        #see if there is a time delta column exists for variable time bin widths
        if "TIMEDEL" not in self.data.keys():
            dt=header["TIMEDEL"]*u.s
        else:
            dt=self.data["TIMEDEL"]


        self.tbins = {}
        #see https://heasarc.gsfc.nasa.gov/ftools/caldb/help/batbinevt.html
        self.tbins["TIME_CENT"] = self.data["TIME"] + (0.5-timepixr)*dt
        self.tbins["TIME_START"] = self.data["TIME"] - timepixr*dt
        self.tbins["TIME_STOP"] = self.data["TIME"] + (1-timepixr)*dt

        #if self.lc_input_dict ==None, then we will need to try to read in the hisotry of parameters passed into batbinevt
        # to create the lightcurve file. thsi usually is needed when we first parse a file so we know what things are if we need to
        # do some sort of rebinning.

        #were looking for something like:
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
            #get the default names of the parameters for batbinevt including its name 9which should never change)
            test = hsp.HSPTask('batbinevt')
            default_params_dict=test.default_params.copy()
            taskname=test.taskname
            start_processing=None

            for i in header["HISTORY"]:
                if taskname in i and start_processing is None:
                    #then set a switch for us to start looking at things
                    start_processing = True
                elif taskname in i and start_processing is True:
                    #we want to stop processing things
                    start_processing = False

                if start_processing and "START" not in i and len(i)>0:
                    values=i.split(" ")
                    #print(i, values, "=" in values)

                    parameter_num=values[0]
                    parameter=values[1]
                    if "=" not in values:
                        #this belongs with the previous parameter and is a line continuation
                        default_params_dict[old_parameter] = default_params_dict[old_parameter] + values[-1]
                    else:
                        default_params_dict[parameter] = values[-1]

                    old_parameter=parameter

            self.lc_input_dict = default_params_dict.copy()

        if calc_energy_integrated:
            self._calc_energy_integrated()


    def _get_event_weights(self):
        """
        This method reads in the appropriate weights for event data once it has been applied to a event file, for a
        given RA/DEC position
        :return:
        """

        #read in all the info for the weights and save it such that we can use these weights in the future for
        #redoing lightcurve calculation
        with fits.open(self.event_file) as file:
            self._event_weights=file[1].data["MASK_WEIGHT"]




    def _set_event_weights(self):
        """
        This method sets the appropriate weights for event data, for a
        given RA/DEC position. This may be necessary if a user is analyzing multiple sources for which event data has been
        obtained.

        Note: event weightings need to be set if the RA/DEC of the light curve doesnt match what is in the event file

        :return:
        """

        if not self._same_event_lc_coords():
            #read in the event file and replace the values in the MASK_WEIGHT with the appropriate values in self._event_weights
            with fits.open(self.event_file, mode="update") as file:
                file[1].data["MASK_WEIGHT"]=self._event_weights
                file.flush()

    def _same_event_lc_coords(self):
        """
        This simple program reads in the event data coordinates and compares it to what is obained from the lightcurve
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

        :return:
        """

        #if we have more than 1 energy bin then we can calculate an energy integrated count rate, etc
        #otherwise we dont have to do anything since theres only one energy bin
        if self.data["RATE"].ndim > 1:
            #calculate the total count rate and error
            integrated_count_rate=self.data["RATE"].sum(axis=1)
            integrated_count_rate_err=np.sqrt(np.sum(self.data["ERROR"]**2, axis=1))

            #get the energy info
            min_e = self.ebins["E_MIN"].min()
            max_e = self.ebins["E_MAX"].max()
            max_energy_index = self.ebins["INDEX"].max()

            #append energy integrated count rate, the error, and the additional energy bin to the respective dicts
            new_energy_bin_size=self.ebins["INDEX"].size+1
            new_e_index=np.arange(new_energy_bin_size, dtype=self.ebins["INDEX"].dtype)

            new_emin=np.zeros(new_energy_bin_size)*self.ebins["E_MIN"].unit
            new_emin[:-1]=self.ebins["E_MIN"]
            new_emin[-1] = min_e

            new_emax=np.zeros_like(new_emin) #the zeros_like gets the units from the array that is passed in
            new_emax[:-1]=self.ebins["E_MAX"]
            new_emax[-1] = max_e

            new_rate=np.zeros((self.data["RATE"].shape[0], new_energy_bin_size))*self.data["RATE"].unit
            new_rate_err=np.zeros_like(new_rate)

            new_rate[:,:-1] = self.data["RATE"]
            new_rate[:,-1] = integrated_count_rate

            new_rate_err[:,:-1] = self.data["ERROR"]
            new_rate_err[:,-1] = integrated_count_rate_err

            #save the updated arrays
            self.ebins["INDEX"] = new_e_index
            self.ebins["E_MIN"] = new_emin
            self.ebins["E_MAX"] = new_emax

            self.data["RATE"] = new_rate
            self.data["ERROR"] = new_rate_err


    def plot(self, energybins=None, plot_counts=False, plot_exposure_fraction=False, time_unit="MET", T0=None):
        """
        This method automatically plots the lightcurve, the user can specify if a certain energy range should be plotted
        T0 has to be in same units as time_unit
        :return:
        """

        #make sure that energybins is a u.Quantity object
        if energybins is not None and type(energybins) is not u.Quantity:
            if type(energybins) is list:
                energybins = u.Quantity(energybins, u.keV)

        #have error checking
        if "MET" not in time_unit and "UTC" not in time_unit and "MJD" not in time_unit:
            raise ValueError("This method plots event data only using MET, UTC, or MJD time")

        if "MET" in time_unit:
            times = self.tbins["TIME_CENT"]
        elif "MJD" in time_unit:
            times = met2mjd(self.tbins["TIME_CENT"].value)
        else:
            times = met2utc(self.tbins["TIME_CENT"].value)

        #get the number of axes we may need
        num_plots=1
        num_plots += (plot_counts + plot_exposure_fraction)
        fig, ax = plt.subplots(num_plots, sharex=True)

        #assign the axes for each type of plot we may want
        axes_queue = [i for i in range(num_plots)]

        if num_plots>1:
            ax_rate=ax[axes_queue[0]]
            axes_queue.pop(0)

            if plot_counts:
                ax_count=ax[axes_queue[0]]
                axes_queue.pop(0)

            if plot_exposure_fraction:
                ax_exposure=ax[axes_queue[0]]
                axes_queue.pop(0)
        else:
            ax_rate = ax


        #plot everything for the rates by default
        for e_idx, emin, emax in zip(self.ebins["INDEX"], self.ebins["E_MIN"], self.ebins["E_MAX"]):
            plotting=True
            if energybins is not None:
                #need to see if the energy range is what the user wants
                if emin == energybins.min() and emax == energybins.max():
                    plotting = True
                else:
                    plotting = False

            if plotting:
                #use the proper indexing for the array
                if len(self.ebins["INDEX"]) > 1:
                    rate=self.data["RATE"][:,e_idx]
                    rate_error=self.data["ERROR"][:, e_idx]
                else:
                    rate = self.data["RATE"]
                    rate_error = self.data["ERROR"]

                ax_rate.plot(times, rate, ds='steps-mid', label=f'{emin.value}-{emax}')
                ax_rate.errorbar(times, rate_error, ls='None')

        if num_plots>1:
            ax_rate.legend()

        if plot_counts:
            ax_count.plot(times, self.data["TOTCOUNTS"], ds='steps-mid')
            ax_count.set_ylabel('Total counts')


        if plot_exposure_fraction:
            ax_exposure.plot(times, self.data["FRACEXP"], ds='steps-mid')
            ax_count.set_ylabel('Fractional Exposure')

        if T0 is not None:
            #plot the trigger time for all panels
            if num_plots>1:
                for axis in ax:
                    axis.axvline(T0, 0,1, ls='--', label=f"T0={T0:.2f}", color='k')
            else:
                ax_rate.axvline(T0, 0,1, ls='--', label=f"T0={T0:.2f}", color='k')

        if num_plots > 1:
            ax[1].legend()
        else:
            ax_rate.legend()


        #fig.savefig("test.pdf")
        return fig, ax


    def _create_custom_timebins(self, timebins, output_file=None):
        """
        This method creates custom time bins from a user defined set of time bin edges

        This method is here so the call to create a gti can be phased out eventually.
        :return:
        """

        if output_file is None:
            #use the same filename as for the lightcurve file but replace suffix with gti and put it in gti subdir instead of lc
            new_path = self.lightcurve_file.parts
            new_name=self.lightcurve_file.name.replace("lc", "gti")

            output_file=Path(*new_path[:self.lightcurve_file.parts.index('lc')]).joinpath("gti").joinpath(new_name)

        return create_gti_file(timebins, output_file, T0=None, is_relative=False, overwrite=True)

    def _call_battblocks(self, output_file=None, save_durations=False):
        """
        This method calls battblocks for bayesian blocks binning of a lightcurve. This rebins the lightcurve into a 64 ms
        energy integrated energy bin (based on current ebins) to calculate the bayesian block time bins and then restores the lightcurve back to what it was

        :return:
        """

        if len(self.ebins["INDEX"]) > 1:
            recalc_energy=True
            #need to rebin to a single energy and save current energy bins
            old_ebins=self.ebins.copy()
            #see if we have the enrgy integrated bin included in the arrays:
            if (self.ebins["E_MIN"][0] == self.ebins["E_MIN"][-1]) and (self.ebins["E_MAX"][0] == self.ebins["E_MAX"][-1]):
                self.set_energybins(emin=self.ebins["E_MIN"][-1], emax=self.ebins["E_MAX"][-1])
                calc_energy_integrated = True #this is for recalculating the lightcurve later in the method
            else:
                self.set_energybins(emin=self.ebins["E_MIN"][-0], emax=self.ebins["E_MAX"][-1])
                calc_energy_integrated = False
        else:
            recalc_energy=False

        #set the time binning to be 64 ms. This time binning will be over written anyways so dont need to restore anything
        self.set_timebins()


        #get the set of default values which we will modify
        stop
        test = hsp.HSPTask('battblocks')
        input_dict = test.default_params.copy()

        if output_file is None:
            #use the same filename as for the lightcurve file but replace suffix with gti and put it in gti subdir instead of lc
            new_path = self.lightcurve_file.parts
            new_name=self.lightcurve_file.name.replace("lc", "gti")

            output_file=Path(*new_path[:self.lightcurve_file.parts.index('lc')]).joinpath("gti").joinpath(new_name)

        #modify some of the inputs here
        input_dict["infile"] = str(self.lightcurve_file) #this should ideally be a 64 ms lightcurve of a single energy bin
        input_dict["outfile"] = str(output_file)

        #these are used by batgrbproducts:
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
            self.set_energybins(emin=old_ebins["E_MIN"], emax=old_ebins["E_MAX"],
                                calc_energy_integrated=calc_energy_integrated)

        if battblocks_return.returncode != 0:
            raise RuntimeError(f'The call to Heasoft battblocks failed with message: {battblocks_return.output}')

        if save_durations:
            stop

        return output_file, battblocks_return

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
