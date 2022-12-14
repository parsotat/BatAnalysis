"""
This file holds various functions that users can call to interface with bat observation objects
"""
import subprocess
import os
import astropy as ap
from astropy.io import fits
from astropy.time import Time
import numpy as np
import shutil
import matplotlib.pyplot as plt
import glob
import os
import sys
import warnings
from pathlib import Path
import requests
from astroquery.heasarc import Heasarc
from copy import copy
import swifttools.swift_too as swtoo
import datetime


# from xspec import *

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)

#try:
#    import xspec as xsp
#except ModuleNotFoundError as err:
    # Error handling
#    print(err)

import swiftbat.swutil as sbu
import swiftbat


_orig_pdir=os.getenv('PFILES')

def dirtest(directory, clean_dir=True):
    """
    Tests if a directory exists and either creates the directory or removes it and then re-creates it

    :param directory: String of the directory that should be created or deleted and re-created
    :param clean_dir: Boolean to denote if the directory should be deleted and recreated
    :return: None
    """

    directory=Path(directory)

    # see if the directory exists
    if directory.exists():
        if clean_dir:
            # remove the directory and recreate it
            shutil.rmtree(directory)
            #os.mkdir(directory)
            directory.mkdir(parents=True)
    else:
        # create it
        #os.mkdir(directory)
        directory.mkdir(parents=True)


def curdir():
    """
    Get the current working directory. Is legacy, since moving to use the pathlib module.
    """
    cdir = os.getcwd() + '/'
    return cdir


def datadir(new=None, mkdir=False, makepersistent=False, tdrss=False) -> Path:
    """Return the data directory (optionally changing and creating it)

    Args:
        new (Path|str, optional): Use this as the data directory
        mkdir (bool, optional): Create the directory (and its parents) if necessary
        makepersistent (bool, optional): If set, stores the name in ~/.swift/swift_datadir_name and uses it as new default
        tdrss (bool, optional): subdirectory storing tdrss data types
    """
    global _datadir
    datadirnamefile = Path("~/.swift/swift_datadir_name").expanduser()

    if new is not None:
        new = Path(new).expanduser().resolve()
        if mkdir:
            new.mkdir(parents=True, exist_ok=True)
            new.joinpath('tdrss').mkdir(exist_ok=True)
            new.joinpath('trend').mkdir(exist_ok=True)
        if makepersistent:
            persistfile = datadirnamefile
            persistfile.parent.mkdir(exist_ok=True)     # make ~/.swift if necessary
            persistfile.open("wt").write(str(new))
        _datadir = new

    if not globals().get('_datadir', False):
        # Not previously initialized
        try:
            _datadir = Path(datadirnamefile.open().read())
            if not _datadir.exists():
                raise RuntimeError(f'Persistent data directory "{_datadir}" does not exist')
        except FileNotFoundError:
            # No persistent directory exists.  Use cwd
            _datadir = Path.cwd()
            warnings.warn(f"Saving data in current directory {_datadir}")

    assert isinstance(_datadir, Path)
    if tdrss:
        return _datadir.joinpath('tdrss')
    return _datadir

def create_custom_catalog(src_name_list, src_ra_list, src_dec_list, src_glon_list, src_glat_list,
                          catalog_name='custom_catalog.cat', catalog_dir=None,
                          catnum_init=32767):
    """
    This creates a catalog file for a number of sources that the user is interested in. Merges the created catalog with
    a past BAT survey catalog which includes typical bright sources observed by BAT. This allows the sources to be
    appropriately cleaned.

    :param src_name_list: List of the names of the sources that should be added to the catalog
    :param src_ra_list: List of the RA of the sources, in the same order as src_name_list
    :param src_dec_list: List of the Dec of the sources, in the same order as src_name_list
    :param src_glon_list: List of the galactic latitude of the sources, in the same order as src_name_list
    :param src_glat_list: List of the galactic longitude of the sources, in the same order as src_name_list
    :param catalog_name: String of the name of the resulting catalog that is produced
    :param catalog_dir: String (or None) of the directory where the catalog should be saved
    :param catnum_init: Int that denotes the initial catalog number to be assigned to the sources of interest, should
        not overlap with any cat_num values of other BAT survey sourced (this parameter should be ignored except for
        very few scenarios)
    :return: Path object pointing to the new catalog file
    """

    #Add check to make sure that input is not tuple
    if type(src_name_list) is tuple or type(src_ra_list) is tuple or type(src_dec_list) is tuple or \
            type(src_glon_list) is tuple or type(src_glat_list) is tuple:
        raise ValueError("The inputs cannot be tuples, either single values or lists are accepted.")

    # make the inputs lists if they are not
    if type(src_name_list) is not list:
        src_name_list = [src_name_list]
    if type(src_ra_list) is not list:
        src_ra_list = [src_ra_list]
    if type(src_dec_list) is not list:
        src_dec_list = [src_dec_list]
    if type(src_glon_list) is not list:
        src_glon_list = [src_glon_list]
    if type(src_glat_list) is not list:
        src_glat_list = [src_glat_list]

    # set default for catalog name and location
    catalog_name=Path(catalog_name)
    if catalog_dir is None:
        catalog_dir = Path.cwd()
    else:
        catalog_dir=Path(catalog_dir)

    prev_name=catalog_name.stem
    cat = catalog_dir.joinpath(prev_name+"_prev.cat") #os.path.join(catalog_dir, prev_name+"_prev.cat")
    final_cat = catalog_dir.joinpath(catalog_name) #os.path.join(catalog_dir, catalog_name)

    # create the columns of file
    c1 = fits.Column(name='CATNUM', array=np.array([i for i in range(catnum_init - len(src_name_list), catnum_init)]),
                     format='I')  # 2 byte integer
    c2 = fits.Column(name='NAME', array=np.array(src_name_list), format='30A')
    c3 = fits.Column(name='RA_OBJ', array=np.array(src_ra_list), format='D', unit="deg", disp="F9.5")
    c4 = fits.Column(name='DEC_OBJ', array=np.array(src_dec_list), format='D', unit="deg", disp="F9.5")
    c5 = fits.Column(name='GLON_OBJ', array=np.array(src_glon_list), format='D', unit="deg", disp="F9.5")
    c6 = fits.Column(name='GLAT_OBJ', array=np.array(src_glat_list), format='D', unit="deg", disp="F9.5")
    c7 = fits.Column(name='ALWAYS_CLEAN', array=np.array([0] * len(src_name_list)), format='1L')  # 1 byte logical

    cols = fits.ColDefs([c1, c2, c3, c4, c5, c6, c7])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(str(cat))

    # need to get the file name off to get the dir this file is located in
    dir = Path(__file__[::-1].partition('/')[-1][::-1])
    #hsp.ftmerge(infile="%s %s" % (os.path.join(dir, 'data/survey6b_2.cat'), str(cat)), outfile=str(final_cat))
    hsp.ftmerge(infile="%s %s" % (str(dir.joinpath('data').joinpath('survey6b_2.cat')), str(cat)), outfile=str(final_cat))


    os.system("rm %s"%(str(cat)))
    #cat.unlink()

    return final_cat

def _source_name_converter(name):
    '''
    This function converts a source name to one that may correspond to the file name found in a merged_pointings_lc directory.
    This function is needed due to the name change with the batsurvey-catmux script.

    :param name: a string or list of strings of dfferent source names
    :return: string or list of strings
    '''

    return " "


def combine_survey_lc(survey_obsid_list, output_dir=None, clean_dir=True):
    """
    Concatenates a set of *.cat files to produce a fits file containing data over the duration of times specified in the
    BatSurvey objects. This runs for the catalog that was passed to the constructor methods of the BatSurvey objects

    :param survey_obsid_list: List of BatSurvey objects
    :param clean_dir: Boolean set to True by default. Denotes if the whole directory that holds all the compiled light curve
        data for the passed survey observations should be deleted and recreated if the directory exists.
    :return: Returns a string with the directory of the combined light curve files
    """

    if type(survey_obsid_list) is not list:
        survey_obsid_list = [survey_obsid_list]

    #get the main directory where we shoudl create the total_lc directory
    if output_dir is None:
        output_dir = survey_obsid_list[0].result_dir.parent.joinpath("total_lc")  #os.path.join(main_dir, "total_lc")
    else:
        output_dir=Path(output_dir).expanduser().resolve()

    #if not os.path.isdir(output_dir):
    #    raise ValueError('The directory %s needs to exist for this function to save its results.' % (output_dir))
    #if the directory doesnt exist, create it otherwise over write it
    dirtest(output_dir, clean_dir=clean_dir)

    # make the local pfile dir if it doesnt exist and set this value
    _local_pfile_dir = output_dir.joinpath(".local_pfile")
    _local_pfile_dir.mkdir(parents=True, exist_ok=True)
    try:
        hsp.local_pfiles(pfiles_dir=str(_local_pfile_dir))
    except AttributeError:
        hsp.utils.local_pfiles(par_dir=str(_local_pfile_dir))

    ret=[]
    for obs in survey_obsid_list:
        for i in obs.pointing_flux_files:
            dictionary = dict(keycolumn="NAME", infile=str(i), outfile= str(output_dir.joinpath("%s.cat"))  ) #os.path.join(output_dir, "%s.cat"))

            # there is a bug in the heasoftpy code so try to explicitly call it for now
            ret.append(hsp.batsurvey_catmux(**dictionary))
            #input_string = "batsurvey-catmux "
            #for i in dictionary:
            #    input_string = input_string + "%s=%s " % (str(i), dictionary[i])
            #os.system(input_string)

    return output_dir


def read_lc_data(filename, energy_band_index=None, T0=0):
    """
    Reads in a fits file that contains rate information, at different energy bands, at a number of METs

    :param filename: String of the name of the fits file
    :param energy_band_index: int or None to denote which energy band the user wants to choose
        The bands, in order of the index that they would be accessed are: 14-20 keV, 20-24 keV, 24-35 keV, 35-50 keV,
        50-75 keV, 75-100 keV, 100-150 keV, 150-195 keV
    :param T0: float that represents a critial time that observations should be measured in time with respect to
    :return: arrays of the time, time bin size, rate, rate_error, and the SNR of the measurement in time
    """

    ## get fits file data
    time = []
    time_err = []
    rate = []
    rate_err = []
    snr = []

    filename=str(filename)
    lc_fits = fits.open(filename)
    lc_fits_data = lc_fits[1].data

    time_array = lc_fits_data.field('TIME')
    timestop_array = lc_fits_data.field('TIME_STOP')
    exposure_array = lc_fits_data.field('EXPOSURE')
    rate_array = lc_fits_data.field('RATE')
    rate_err_array = lc_fits_data.field('RATE_ERR')
    bkg_var_array = lc_fits_data.field('BKG_VAR')
    snr_array = lc_fits_data.field('VECTSNR')

    for i in range(len(lc_fits_data)):
        time_start = (time_array[i] - T0) #this is in MET
        time_stop = (timestop_array[i] - T0)
        time_mid = (time_start + time_stop) / 2.0
        #time_mid = time_mid / (24 * 60 * 60) #comment because we want to leave units as MET
        time_err_num = (time_stop - time_start) / 2.0
        #time_err_num = time_err_num / (24 * 60 * 60) #comment because we want to leave units as MET

        time.append(time_mid)
        time_err.append(time_err_num)

        if energy_band_index is not None:
            rate.append(rate_array[i][energy_band_index - 1])
            rate_err.append(rate_err_array[i][energy_band_index - 1])
            snr.append(snr_array[i][energy_band_index - 1])
        else:
            if len(rate_array[i])>8:
                rate.append(rate_array[i][-1])
                rate_err.append(rate_err_array[i][-1])
                snr.append(snr_array[i][-1])
            else:
                rate_tot = 0.0
                rate_err_2_tot = 0.0
                bkg_var_2_tot = 0.0
                for j in range(len(rate_array[i])):
                    rate_num = rate_array[i][j]
                    rate_err_2 = rate_err_array[i][j] * rate_err_array[i][j]
                    bkg_var_2 = bkg_var_array[i][j] * bkg_var_array[i][j]
                    rate_tot = rate_tot + rate_num
                    rate_err_2_tot = rate_err_2_tot + rate_err_2
                    bkg_var_2_tot = bkg_var_2_tot + bkg_var_2

                rate.append(rate_tot)
                rate_err_tot = np.sqrt(rate_err_2_tot)
                rate_err.append(rate_err_tot)
                snr_allband_num = rate_tot / np.sqrt(bkg_var_2_tot)
                snr.append(snr_allband_num)

    lc_fits.close

    return time, time_err, rate, rate_err, snr


def calc_response(phafilename, srcname=None, indir=None, outdir=None):
    """
    This function generates the response matrix for a given pha file by calling batdrmgen (this is a HEASOFT function).

    :param phafilename: String that denotes the location and name of the PHA file that the user would like to calculate
        the response matrix for.
    :param srcname: String denoting the source name (no spaces). The source name must match with the one that is in the phafilename.
    :param indir: String denoting the full path/to/the/input-directory. By default it takes the current directory. 
    :param outdir: String denoting the full path/to/the/output-directory. By default it takes the current directory
    :return: Heasoftpy "Result" object obtained from calling heasoftpy batdrmgen. The "Result" object is the entire output, which
helps to debug in case of an error.
    """


    if type(phafilename) is not list:
        phafilename=[phafilename]

    #when passing in tht whole filename, the paths mess up the conection between the response file and the pha file since
    # there seems to be some character limit to this header value. Therefore we need to cd to the directory that the PHA
    #file lives in and create the .rsp file and then cd back to the original location.

    #make sure that all elements are paths
    phafilename=[Path(i) for i in phafilename]

    #we are passing in a whole filepath or
    # we are already located in the PHA directory and are mabe calculating the upperlimit bkg spectrum
    _local_pfile_dir = phafilename[0].resolve().parents[1].joinpath(".local_pfile") #Path(f"/tmp/met2mjd_{os.times().elapsed}")
    _local_pfile_dir.mkdir(parents=True, exist_ok=True)
    try:
        hsp.local_pfiles(pfiles_dir=str(_local_pfile_dir))
    except AttributeError:
        hsp.utils.local_pfiles(par_dir=str(_local_pfile_dir))

    # Check if the phafilename is a string and if it has an extension .pha. If NOT then exit
    for filename in phafilename:

        #if type(filename) is not str or '.pha' not in os.path.splitext(filename)[1]:
        if '.pha' not in filename.name:
            raise ValueError('The file name %s needs to be a string and must have an extension of .pha .' % (str(filename)))


    
    #get the cwd
        current_dir=Path.cwd()

    #get the directory that we have to cd to and the name of the file
        #pha_dir, pha_file=os.path.split(filename)
        pha_dir=filename.parent
        pha_file=filename.name

    #cd to that dir
        #if str(pha_dir) != '':
        if str(pha_dir) != str(current_dir):
            os.chdir(pha_dir)

    # Split the filename by extension, so as to remove the .pha and replace it with .rsp
        out = pha_file.split(".")[0] + '.rsp'

    #create drm
        output=hsp.batdrmgen(infile=pha_file, outfile=out, chatter=2, clobber="YES", hkfile="NONE")


    #cd back
        #if pha_dir != '':
        if str(pha_dir) != str(current_dir):
            os.chdir(current_dir)

    #shutil.rmtree(_local_pfile_dir)

    return output

def fit_spectrum(phafilename,surveyobservation, plotting=True, generic_model=None,setPars=None, indir=None, outdir=None, use_cstat=True, fit_iterations=1000,verbose=True):
    """
    Fits a spectrum that is loaded in from a BAT pha file. The header of the PHA file must have the associated
    response information.

    User has to pass a phafilename and the BatSurvey object "surveyobservation" (mandatory). 
    The user should have already run the "batsurvey" command and created the surveyobservation object.
 
    The user can specfiy their own spectral model that is XSPEC compatible. 
    To learn about how to specify a spectral model in pyXspec the user can
    look at the following link: https://heasarc.gsfc.nasa.gov/xanadu/xspec/python/html/index.html

    For e.g, to specify a model one has to do the following:
    model=xsp.Model(generic_model,setPars={1:45,2:"123,-1"}) Here -1 stands for "frozen".

    User has to specify cflux in their model. This is mandatory because we use this same function (and hence the user specified model) 
    to test any detection in the BAT energy band.


    If no model is specfied by the user, then by default the specrum is fit with the following Xspec model:
    cflux*(powerlaw): with Cflux E_min=14 keV (Frozen), E_max=195 keV (Frozen), flux=-12 (initial value),
    powerlaw Gamma=2 Free, and norm=frozen. Powerlaw norm kept frozen.

   
    
    :param phafilename: String that denotes the location and name of the PHA file.
    :param surveyobservation: Object denoting the batsurvey observation object which contains all the 
     necessary information related to this observation.
    :param plotting: Boolean statement, if the user wants to plot the spectrum.
    :param generic_model: String with XSPEC compatible model, which must include cflux.
    :param setPars: Boolean to set the parameter values of the model specified above.
    :param indir: String denoting the full path/to/the/input-directory. By default it takes the current directory.
    :param outdir: String denoting the full path/to/the/output-directory. By default it takes the current directory.
    :param use_cstat: Boolean to use cstat in case of low counts (Poisson statistics), otherwise use chi squared stats.
    :param fit_iterations: Number of fit iterations to be carried out by XSPEC.
     Since BAT data has just 8 energy channels, a default of 100 is enough.
     But the user can specify any value that may be needed.
    :param verbose: Boolean to show every output during the fitting process.
     Set to True by default, that'll help the user to identify any issues with the fits.
    :return: None
    """

    try:
        import xspec as xsp
    except ModuleNotFoundError as err:
        # Error handling
        print(err)
        raise ModuleNotFoundError('The pyXspec package needs to installed to fit spectra with this function.')

    #In the next few steps we will get into the directory where the PHA files and rsp files are located
    # Do the fitting and then get out to our current directory: current_dir
    #get the cwd.
    phafilename=Path(phafilename)
    current_dir=Path.cwd()
    #plt.ion()

# Check if the phafilename is a string and if it has an extension .pha. If NOT then exit
    #if type(phafilename) is not str or '.pha' not in  os.path.splitext(phafilename)[1]:
    if '.pha' not in phafilename.name:
        raise ValueError('The file name %s needs to be a string and must have an extension of .pha .' % (str(phafilename)))


    #get the directory that we have to cd to and the name of the file
    #pha_dir, pha_file=os.path.split(phafilename)
    pha_dir = phafilename.parent
    pha_file = phafilename.name

    pointing_id=  pha_file.split(".")[0].split("_")[-1]

    if len(pha_file.split("_survey"))>1:
        #weve got a pha for a normal survey catalog
        source_id=pha_file.split("_survey")[0]  #This is the source name compatible with the catalog
    else:
        #we've got a mosaic survey result
        source_id=pha_file.split("_mosaic")[0]

    #cd to that dir
    #if pha_dir != '':
    if str(pha_dir) != str(current_dir):
        os.chdir(pha_dir)
    
    xsp.AllData -= "*"
    s = xsp.Spectrum(pha_file)  # from xspec import * has been done at the top. This is a spectrum object
    # s.ignore("**-15,150-**")	#Ignoring energy ranges below 15 and above 150 keV.

    
    # Define model

    if generic_model is not None:  #User provides a string of model, and a Dictionary for the initial values
        if np.type(generic_model) is str:

            if "cflux" in generic_model: #The user must provide the cflux, or else we will not be able to predict of there is a statistical detection (in the next function).
	    
                try:
                    model=xsp.Model(generic_model,setPars=setPars) #Set the initial value for the fitting using the Model object attribute

                except Exception as e:
                    print(e)
                    raise ValueError("The model needs to be specified correctly")


            else: 
                raise ValueError("The model needs cflux in order to calulate error on the flux in 14-195 keV")



    else:

         # If User does not pass any model

         model = xsp.Model("cflux*po")
         p1 = model(1)  # cflux      Emin = 14 keV
         p2 = model(2)  # cflux      Emax = 195 keV
         p3 = model(3)  # cflux      lg10Flux
         p4 = model(4)  # Photon index Gamma
         p5 = model(5)  # Powerlaw norm

         # Setting the vlaues and freezing them.

         p1.values = 14  # already frozen
         p2.values = 195  # already frozen
         p4.values = 2
         p4.frozen = False
         p5.values = 0.001
         p5.frozen = True


           

                # model_components=model.componentNames  #This is a list of the model components
		#Check if the model is XSPEC compatible : Done
		#Listing down the model parameters in a dictionary: parm1: Value, param2: Value....
		# If no initial values given , default XSPEC values to be used.
        	#We will manipulate these param values to "set a value" or "freeze/thaw" a value, set a range for these viable values.
		#We can call the best fit param values, after fit.


       # Fitting the data with this model

    if use_cstat:
        xsp.Fit.statMethod = "cstat"
    else:
        xsp.Fit.statMethod= "chi"

    # Stop fit at nIterations and do not query.
    xsp.Fit.query = "no"

    xsp.Fit.nIterations = fit_iterations
    xsp.Fit.renorm()

    #try to do the fitting if it doesnt work fill in np.nan values for things
    try:
        xsp.Fit.perform()
        if verbose:
            xsp.AllModels.show()
            xsp.Fit.show()


        # Get coordinates from XSPEC plot to use in matplotlib:
        xsp.Plot.device = '/null'
        xsp.Plot('data')
        chans = xsp.Plot.x()
        rates = xsp.Plot.y()
        xerr=xsp.Plot.xErr()
        yerr=xsp.Plot.yErr()
        folded = xsp.Plot.model()

        # Plot using Matplotlib:
        f, ax=plt.subplots()
        ax.errorbar(x=chans,xerr=xerr,y=rates,yerr=yerr,fmt='ro')
        ax.plot(chans, folded,"k-")
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('counts/cm^2/sec/keV')
        ax.set_xscale('log')
        ax.set_yscale('log')
        f.savefig(phafilename.parent.joinpath(phafilename.stem+".pdf"))     #phafilename.split('.')[0]+'.pdf')
        if plotting:
            plt.show()

        # Capturing the Flux and its error. saved to the model object, can be obtained by calling model(1).error, model(2).error
        model_params=dict()
        for i in range(1,model.nParameters+1):
            xsp.Fit.error("2.706 %d"%(i))

            #get the name of the parameter
            par_name=model(i).name
            model_params[par_name]=dict(val=model(i).values[0], lolim=model(i).error[0], hilim=model(i).error[1], errflag=model(i).error[-1])
        surveyobservation.set_pointing_info(pointing_id,"model_params", model_params, source_id=source_id)

    except Exception as Error_with_Xspec_fitting:
        #this is probably that XSPEC cannot fit because of negative counts
        if verbose:
            print (Error_with_Xspec_fitting)

        #need to fill in nan values for all the model params and 'TTTTTTTTT' for the error flag
        model_params = dict()
        for i in range(1, model.nParameters + 1):

            # get the name of the parameter
            par_name = model(i).name
            model_params[par_name] = dict(val=np.nan, lolim=np.nan, hilim=np.nan,
                                          errflag='TTTTTTTTT')
        surveyobservation.set_pointing_info(pointing_id, "model_params", model_params, source_id=source_id)

    # Incorporating the model names, parameters, errors into the BatSurvey object.
    xsp.Xset.save(pha_file.split(".")[0])
    xspec_savefile = phafilename.parent.joinpath(phafilename.stem+".xcm")  #os.path.join(pha_dir, pha_file.split(".")[0] + ".xcm")
    surveyobservation.set_pointing_info(pointing_id, "xspec_model", xspec_savefile, source_id=source_id)

    #xsp.Fit.error("2.706 3")
    #fluxerr_lolim = p3.error[0]
    #fluxerr_hilim =p3.error[1]
    #pyxspec_error_string=p3.error[2]   #The string which says if everything is correct. Should be checked if there is non-normal value.
    #flux = p3.values[0]
    
    #cd back
    #if pha_dir != '':
    if str(pha_dir) != str(current_dir):
        os.chdir(current_dir)

    #return flux, (fluxerr_lolim, fluxerr_hilim), pyxspec_error_string

def calculate_detection(surveyobservation,source_id, nsigma=3,bkg_nsigma=5, plot_fit=False,verbose=True):
    """
    This function uses the fitting function and statistically checks if there is any significant detection (at a specfied confidence).
    If there is no detection, then the function re-calculates the PHA with a bkg_nsigma times the background to calculate the 
    upper limit on the flux, at a certain confidence level (given by the user specified bkg_nsigma).

    We deal with two cases:
       
     (1) Non-detection:  Checking if nsigma error on  The 14-195 keV flux is consistent with the equation (measured flux - nsigma*error)<=0, 
     then return: upper limit=True
     and then recalculate the PHA +response again.... with count rate= bkg_nsigma*BKG_VAR

     (2) Detection: If (measured flux - nsigma*error)>=0 then return: "detection has been measured"

     This operates on the entire batsurvey object (corresponding to a batobservation id), 
     and we want to see if there is a detection for 'any number of pointings for a given source' in that batobservation id. 

     Note that it operates ONLY on one source.
     For different sources one can specify separate detection threshold ('sigma') for different sources. 
     Thus we have kept this function to operate only ONE source at a time.

    :param surveyobservation: Object denoting the batsurvey observation object which contains all the necessary information related to this observation.
    :param source_id: String denoting the source name exactly as that in the phafilename.
    :param nsigma: Integer, denoting the number fo sigma the user needs to justify a detection
    :param bkg_nsigma: Integer, denoting the number of sigma the user needs to to calculate flux upper limit in case of a non detection.
    :param verbose: Boolean to show every output during the fitting process. Set to True by default, that'll help the user to identify any issues with the fits.
    :return: In case of a non-detection a flux upper limit is returned.
    """

    try:
        import xspec as xsp
    except ModuleNotFoundError as err:
        # Error handling
        print(err)
        raise ModuleNotFoundError('The pyXspec package needs to installed to determine if a source has been detected with this function.')


    #flux=np.power(10,flux)
    #fluxerr=np.power(10,flux)- np.power(10,(flux-fluxerr))


    current_dir=Path.cwd()

    #get the directory that we have to cd to and the name of the file
    #pha_dir,_=os.path.split(surveyobservation.get_pha_filenames(id_list=[source_id])[0])
    pha_dir=surveyobservation.get_pha_filenames(id_list=[source_id])[0].parent
    #original_pha_file_list= surveyobservation.pha_file_names_list.copy()

    pointing_ids=surveyobservation.get_pointing_ids() #This is a list of pointing_ids in this bat survey observation

    #cd to that dir
    #if pha_dir != '':
    if str(pha_dir) != str(current_dir):
        os.chdir(pha_dir)

    flux_upperlim=[]

    phafilename_list=surveyobservation.get_pha_filenames(id_list=[source_id],pointing_id_list=pointing_ids) #By specifying the source_id, we now have the specific PHA filename list corresponding to the pointing_id_list for this given bat survey observation.	
   
    for i in range(len(phafilename_list))  :  #Loop over all phafilename_list, 

        #pha_dir, pha_file=os.path.split(phafilename_list[i])
        pha_dir = phafilename_list[i].parent
        pha_file = phafilename_list[i].name

        pointing_id=pha_file.split(".")[0].split("_")[-1]

        # Within the pointing dictionar we have the "key" called "Xspec_model" which has the parameters, values and errors.
        try:
            pointing_dict= surveyobservation.get_pointing_info(pointing_id,source_id=source_id)
            # xsp.Xset.restore(os.path.split(pointing_dict['xspec_model'])[1])
            # model=xsp.AllModels(1)
            model = pointing_dict["model_params"]["lg10Flux"]
            flux = model["val"]  # ".cflux.lg10Flux.values[0]              #Value
            fluxerr_lolim = model["lolim"]  # .cflux.lg10Flux.error[0]      #Error
            fluxerr_uplim = model["hilim"]  # .cflux.lg10Flux.error[1]

            print('The condition here is', (flux), nsigma, fluxerr_uplim,
                  (10 ** flux) - nsigma * (10 ** (fluxerr_uplim - flux)))
            avg_flux_err = 0.5 * (((10 ** fluxerr_uplim) - (10 ** flux)) + ((10 ** flux) - (10 ** fluxerr_lolim)))

        except ValueError:
            #the fitting wasnt not successful and the dictionary was not created but want to enter the upper limit if
            #statement
            fluxerr_lolim=0
            flux=1
            nsigma=1
            avg_flux_err=1

        if fluxerr_lolim==0 or ( ((10**flux) - nsigma * avg_flux_err) <= 0) or np.isnan(flux):

            print("No detection, just upperlimits for the spectrum:",pha_file)
        # Here redo the PHA calculation with 5*BKG_VAR
            surveyobservation.calculate_pha(calc_upper_lim=True, bkg_nsigma=bkg_nsigma, id_list=source_id,single_pointing=pointing_id)

            # can also do surveyobservation.get_pha_filenames(id_list=source_id,pointing_id_list=pointing_id, getupperlim=True)
            # to get the created upperlimit file
            bkgnsigma_upper_limit_pha_file= pha_file.split(".")[0]+'_bkgnsigma_%d'%(bkg_nsigma) + '_upperlim.pha'

            try:
                calc_response(bkgnsigma_upper_limit_pha_file)
            except:
                #This is a MosaicBatSurvey object which already has the default associated response file
                pass
        
            xsp.AllData -= "*"

            s = xsp.Spectrum(bkgnsigma_upper_limit_pha_file)

            model = xsp.Model("po")
            #p1 = m1(1)  # cflux      Emin = 15 keV
            #p2 = m1(2)  # cflux      Emax = 150 keV
            #p3 = m1(3)  # cflux      lg10Flux
            p4 = model(1)  # Photon index Gamma
            p5 = model(2)  # Powerlaw norm


            #p1.values = 15  # already frozen
            #p2.values = 150  # already frozen
            p4.frozen = True
            p4.values=1
            p5.values = 0.001
            p5.frozen = False

            if verbose:
                print("******************************************************")
                print("Fitting the 5 times bkg of the spectrum ",bkgnsigma_upper_limit_pha_file)

            xsp.Fit.nIterations = 100
            xsp.Fit.perform()
            if plot_fit:
                xsp.AllModels.show()
                xsp.Fit.show()
            xsp.AllModels.calcFlux("14.0 195.0")

            if verbose:
                print("******************************************************")
                print("******************************************************")
                print("******************************************************")

                print (s.flux)

            # Capturing the simple model. saved to the model object, can be obtained by calling model(1).error, model(2).error
            #if the original fit had failed b/c of negative counts
            #try:
            #    pointing_dict = surveyobservation.get_pointing_info(pointing_id, source_id=source_id)
            #except ValueError:
            model_params = dict()
            for i in range(1, model.nParameters + 1):

                # get the name of the parameter
                par_name = model(i).name
                model_params[par_name] = dict(val=model(i).values[0], lolim=model(i).error[0], hilim=model(i).error[1],
                                              errflag='TTTTTTTTT')
            surveyobservation.set_pointing_info(pointing_id, "model_params", model_params, source_id=source_id)

            surveyobservation.set_pointing_info(pointing_id,"nsigma_lg10flux_upperlim",np.log10(s.flux[0]),source_id=source_id)
            
            #pointing_id=pha_file.split(".")[0].split("_")[-1]
            #surveyobservation.pointing_info[pointing_id]["flux"]=0
            #surveyobservation.pointing_info[pointing_id]["flux_lolim"]=0
            #surveyobservation.pointing_info[pointing_id]["flux_hilim"]=s.flux[0]

    
        else:  # Detection


            if verbose:
                print("A detection has been measured at the %d sigma level"%(nsigma))
            #pointing_id=pha_file.split(".")[0].split("_")[-1]
            #surveyobservation.pointing_info[pointing_id]["flux"]=flux[i]
            #surveyobservation.pointing_info[pointing_id]["flux_lolim"]=fluxerr_lolim[i]
            #surveyobservation.pointing_info[pointing_id]["flux_hilim"]=fluxerr_hilim[i]




    #cd back
    #if pha_dir != '':
    if str(pha_dir) != str(current_dir):
        os.chdir(current_dir)


    return  flux_upperlim   	#This is a list for all the Valid non-detection pointings


def print_parameters(obs_list, source_id, values=["met_time","utc_time", "exposure"], latex_table=False, savetable=False, save_file="output.txt", overwrite=True):

    """
    Convenience function to plot various survey data pieces of information in a formatted file/table

    :param obs_list: A list of BatSurvey objects
    :param source_id: A string with the name of the source of interest.
    :param values: A list of strings contaning information that the user would like to be printed out. The strings
        correspond to the keys in the pointing_info dictionaries of each BatSurvey object.
    :param latex_table: Boolean to denote if the output should be formatted as a latex table
    :param savetable: Boolean to denote if the user wants to save the table to a file
    :param save_file: string that specified the location and name of the file that contains the saved table
    :param overwrite: Boolean that says to overwrite the output file if it already exists
    :return: None
    """

    save_file=Path(save_file)

    if save_file.exists() and overwrite:
        save_file.unlink()
 
    if type(obs_list) is not list:
        obs_list=[obs_list]

    if savetable and save_file is not None:
        #open the file to write the output to
        f=open(str(save_file),"w")

    #dont allow xspec to prompt
    #xsp.Xset.allowPrompting = False

    #sort the obs ids by time of 1st pointing id
    all_met=[i.pointing_info[i.pointing_ids[0]]["met_time"] for i in obs_list]
    sorted_obs_idx=np.argsort(all_met)

    outstr="Obs ID  \t Pointing ID\t"
    for i in values:
        outstr+="\t%s"%(i)

    if not savetable:
        print(outstr)
    else:
        f.writelines([str(outstr), "\n"])

    for idx in sorted_obs_idx:
        obs=obs_list[idx]
        try:
            #have obs id for normal survey object
            observation_id=obs.obs_id
        except AttributeError:
            #dont have obs_id for mosaic survey object
            observation_id='mosaic'

        outstr="%s"%(observation_id)

        #sort the pointing IDs too
        sorted_pointing_ids=np.sort(obs.pointing_ids)

        for pointings in sorted_pointing_ids:

            if latex_table:
                outstr += " &"

            outstr += "\t%s" % (pointings)

            if latex_table:
                outstr += " &"

            if source_id in obs.get_pointing_info(pointings).keys():


                #get the model component names
                pointing_dict=obs.get_pointing_info(pointings,source_id=source_id)
                #xsp.Xset.restore(pointing_dict['xspec_model'])
                #model = xsp.AllModels(1)
                model = pointing_dict["model_params"]
                
                if model is not None:
                    model_names= model.keys() #[model(i).name for i in range(1,model.nParameters+1)]
                else:
                    #create a dict keys list that is empty so the
                    #remaining code works
                    model_names = dict().keys()


                for i in values:
                    # get the real key we need if its a xspec model parameter
                    is_model_param = False
                    for key in model_names:
                        if i.capitalize() in key or i in key:
                            model_param_key = key
                            is_model_param = True

                    #see if the parameter exists in the pointings info otherwise look in source_id dictionary, otherwise
                    # in the future look in the spectral model otherwise print nan
                    if i in obs.get_pointing_info(pointings).keys():
                        outstr += "\t%s" % (str(obs.pointing_info[pointings][i]))
                    elif i in obs.get_pointing_info(pointings, source_id=source_id).keys():
                        outstr += "\t%s" % (str(obs.pointing_info[pointings][source_id][i]))
                    elif (is_model_param or ("flux" in i or "Flux" in i)) and model is not None:
                        # print the actual value
                        middle_str = '\t'
                        if latex_table:
                            middle_str += '$'

                        #see if the user wants the flux and if there is an upper limit available
                        if ("flux" in i or "Flux" in i) and "nsigma_lg10flux_upperlim" in pointing_dict.keys():
                            #outstr += "\t  %e  "%(pointing_dict["nsigma_lg10flux_upperlim"])
                            val=10**pointing_dict["nsigma_lg10flux_upperlim"]
                            base = int(str(val).split('e')[-1])

                            middle_str += f'{val/10**base:-.3}'

                            if latex_table:
                                middle_str += f" \\times "
                            else:
                                middle_str += f' x '

                            middle_str += f'10^{{{base:+}}}'

                        else:
                            #get the value and errors if the error calculation worked properly
                            val=model[model_param_key]["val"]
                            if 'T' in model[model_param_key]["errflag"]:
                                err_val="nan"
                                errs = np.array([np.nan, np.nan])
                                #outstr += "\t%s-%s\+%s" % (val, errs[0], errs[1])
                            else:
                                errs = np.array([model[model_param_key]["lolim"], model[model_param_key]["hilim"]])
                                err_val="%e"%(np.abs(val - errs).max())
                                #outstr += "\t%e-%e\+%e"%(val,errs[0], errs[1])


                            #if we've got scientific notation, print it nicely
                            if ("flux" in i or "Flux" in i) or len(str(val).split('e'))>1:
                                if ("flux" in i or "Flux" in i):
                                    val=10**val
                                    errs = 10 ** errs
                                base=int(str(val).split('e')[-1])
                                diff_errs = np.abs(val - errs)
                                middle_str += f'{val/10**base:-.3}^{{{diff_errs[1]/10**base:+.2}}}_{{{-1*diff_errs[0]/10**base:+.2}}}'

                                if latex_table:
                                    middle_str += f" \\times "
                                else:
                                    middle_str += f' x '

                                middle_str += f'10^{{{base:+}}}'
                            else:
                                diff_errs = np.abs(val - errs)
                                middle_str += f'{val:-.3}'

                                if "nsigma_lg10flux_upperlim" not in pointing_dict.keys():
                                    middle_str += f'^{{{diff_errs[1]:+.2}}}_{{{-1*diff_errs[0]:+.2}}}'

                        if latex_table:
                            middle_str += '$'

                        outstr += middle_str
                    else:
                        outstr += "\tnan"

                    if i != values[-1]:
                        if latex_table:
                            outstr += " &"

            else:
                #if the source doesnt exist for the observation print nan so the user can double check these
                for i in values:
                    outstr+="\tnan"

            if latex_table:
                outstr += " \\\\"
            outstr += "\n"
            for j in range(len(observation_id)):
                outstr += " "

        if savetable and save_file is not None:
            f.writelines([str(outstr),"\n"])
        else:
            print(outstr)

    if savetable and save_file is not None:
        f.close()


def download_swiftdata(table,  reload=False,
                        bat=True, auxil=True, log=False, uvot=False, xrt=False,
                        save_dir=None, **kwargs) -> dict:
    """
    Downloads swift data from heasarc.

    :param table: A astropy query table with OBSIDs, or a list of OBSIDs, that the user would like to download
    :param reload: load even if the data is already in the save_dir
    :param bat: load the bat data
    :param auxil: load the bat data
    :param log: load the log data   (mostly diagnostic, defaults to false)
    :param uvot: load the uvot data (high volume, defaults to false)
    :param xrt: load the xrt data (high volume, defaults to false)
    :param save_dir: The output directory where the observation ID directories will be saved
    :param kwargs: passed to swifttools.swift_too.Data
    :return: dict{obsid: {obsoutdir:..., success:..., loaded:..., [, datafiles:swtoo.Data][, ]}
    """
    results = {}
    if save_dir is None:
        save_dir = datadir()
    save_dir = Path(save_dir).resolve()
    if np.isscalar(table) or isinstance(table, ap.table.row.Row):
        table = [table]
    obsids = []
    for entry in table:
        try:    # swiftmastr observation table
            entry = entry["OBSID"]
        except:
            pass
        try: # swifttools.ObsQuery
            entry = entry.obsid   # f"{entry.targetid:08d}{entry.seg:03d}"
        except:
            pass
        if isinstance(entry, int):
            entry = f"{entry:011d}"
        if not isinstance(entry, str):
            raise RuntimeError(f"Can't convert {entry} to OBSID string")
        obsids.append(entry)
    fetch = kwargs.pop('fetch', True)
    nowts = datetime.datetime.now().timestamp()
    for obsid in obsids:
        obsoutdir = save_dir.joinpath(obsid)
        quicklookfile = obsoutdir.joinpath('.quicklook')
        result = dict(success=True, obsoutdir=obsoutdir, quicklook=False)
        try:
            clobber = reload or quicklookfile.exists()
            data = swtoo.Data(obsid=obsid, clobber=clobber,
                            bat=bat, log=log, auxil=auxil, uvot=uvot, xrt=xrt,
                            outdir=str(save_dir), **kwargs)
            result['data'] = data
            if data.quicklook:  # Mark the directory as quicklook
                quicklookfile.open("w").close()
                result['quicklook'] = True
            elif quicklookfile.exists():
                # This directory just transitioned from quicklook to archival version
                oldqlookdir = save_dir.joinpath("old_quicklook",obsid)
                oldqlookdir.mkdir(exist_ok=True, parents=True)
                for stalefile in obsoutdir.glob('**/*'):
                    # Any file older than the time before the data was downloaded
                    if (stalefile.is_file() and stalefile.stat().st_mtime < nowts 
                        and not stalefile.name.startswith(".")):
                        stalefile.replace(oldqlookdir.joinpath(stalefile.name))
                quicklookfile.unlink()
                result.update(datafiles=data, quicklook=data.quicklook,
                              outdir=Path(data.outdir), success=True, downloaded=True)
            if not Path(data.outdir).is_dir():
                raise RuntimeError(f"Data directory {data.outdir} missing")
        except Exception as e:
            print(f"{obsid} {e}", file=sys.stderr)
            result['success'] = False
        results[obsid] = result
    return results


def download_swiftdata_legacy(table, reload=False,
                        bat=True, log=True, auxil=True, uvot=False, xrt=False,
                        save_dir=None) -> dict:
    """
    Downloads swift data from heasarc.
    LEGACY code from before I was aware of swifttools.swift_too

    :param table: A astropy query table with OBSIDs, or a list of OBSIDs, that the user would like to download
    :param reload: load even if the data is already in the save_dir
    :param bat: load the bat data
    :param log: load the log data
    :param auxil: load the bat data
    :param uvot: load the uvot data (high volume, defaults to false)
    :param xrt: load the xrt data (high volume, defaults to false)
    :param save_dir: The output directory where the observation ID directories will be saved
    :return: dict{obsid: dict(localdir=..., [remoteurl=...,] [cut_dirs=...] [starttime=...], [success=...])}
    """

    # data in heasarc is organized by observation day/month and then observation id
    # eg for observation id 00013201221, the download command is:
    # wget -q -nH --no-check-certificate --cut-dirs=5 -r -l0 -c -N -np -R 'index*'
    # -erobots=off --retr-symlinks https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/2021_12//00013201221/
    # for GRBs do eg. object_name='GRB110414A'
    # table = heasarc.query_object(object_name, mission=mission, sortvar="START_TIME")
    # The first entry in the table should be the TTE data observation ID, from when the GRB was triggered

    result = {} 

    # Whatever table is 
    if np.isscalar(table) or isinstance(table, ap.table.row.Row):
        table = [table]

    #base of wget command to download data
    download_base="wget -q -nH --no-check-certificate -r -l0 -c -N -np -R 'index*' -erobots=off --retr-symlinks "
    subdirectories = []
    for subdirectory in "bat log auxil uvot xrt".split():
        if locals()[subdirectory]:
            subdirectories.append(subdirectory)
    if save_dir is None:
        save_dir = datadir()
    else:
        save_dir = Path(save_dir)

    if not save_dir.exists():
        raise ValueError(f"Save directory {save_dir} does not exist")

    for observation in table:
        details = dict()
        find_data(observation=observation, save_dir=save_dir, details=details)
        obsid = details['obsid']
        obsdir = details['localdir']
        for subdirectory in subdirectories:
            if reload or not obsdir.joinpath(subdirectory).is_dir():
                link = details['remoteurl']
                # print(id, t.datetime.year, t.datetime.month, f"{link}/{subdirectory}/")

                input_string=f"{download_base} --directory-prefix={obsdir} --cut-dirs={details['cutdirs']} {link}/{subdirectory}/"
                errout = os.system(input_string)
                if errout != 0:
                    print(f"Command failed with exitcode {os.waitstatus_to_exitcode(errout)}\n   {input_string}", file=sys.stderr)
                    details['success'] = False
                # retcode = subprocess.check_output(input_string, shell=True, stderr=subprocess.STDOUT)
        result[details['obsid']] = details
    return result

    # if save_dir is not None:
    #     os.chdir(current_dir)


def test_remote_URL(url):
    return requests.head(url).status_code < 400


def from_heasarc(object_name=None, tablename='swiftmastr', **kwargs):
    heasarc=Heasarc()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ap.utils.exceptions.AstropyWarning)
        table = heasarc.query_object(object_name=object_name, mission=tablename, **kwargs)
    return table


def find_data_legacy(observation, *, save_dir, details):
    """Where should the data be locally and remotely
    This is legacy code form before swifttools.swift_too came to my knowledge

    Args:
        observation (_type_): _description_
        save_dir (_type_): _description_
        details (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if isinstance(observation, int):
        observation = f"{observation:011d}"
    if isinstance(observation, str):
        obstable = from_heasarc(obsid=observation)
        if len(obstable) > 0:
            observation = obstable[0]
    if isinstance(observation, ap.table.Row):
        obsid = observation["OBSID"]
        tstart = swiftbat.mjd2datetime(float(observation["START_TIME"]))
        remoteurl = f"https://heasarc.gsfc.nasa.gov/FTP/swift/data/obs/{tstart:%Y_%m}/{obsid}"
        localdir = save_dir.joinpath(obsid)
        if test_remote_URL(remoteurl):
            details.update(obsid=obsid, tstart=tstart, 
                           remoteurl=remoteurl, cutdirs=6, 
                           localdir=save_dir.joinpath(obsid),
                           row=copy(observation),
                           success=True)
            return True
    elif isinstance(observation, str) and len(observation) == 11:
        data = swtoo.Data(obsid=observation, bat=True)

        quicklook_available_url = "https://swift.gsfc.nasa.gov/data/swift/"
        afstobs = swtoo.ObsQuery(obsid=observation) # As flown science timeline
        # swtoo doc: https://www.swift.psu.edu/too_api/index.php?md=Introduction.md
        if len(afstobs) > 0: # Found it
            age = afstobs[0].begin - datetime.datetime.utcnow()
            if age.total_seconds()/86400 < 30: #  If less than a month old
                pass
    raise NotImplementedError("Currently, can only ask for fully-processed observations in the swiftmastr table")


def find_trigger_data():
    raise NotImplementedError

def met2mjd(met_time):
    """
    A convenience function that calculates the MJD time from a Swift MET time. Ths function either uses the swiftbat
    code base which is quicker or the heasoftpy swifttime function which is slower.

    :param met_time: a number that is the Swift MET time that will be converted
    :return: a MJD date that includes the Swift time clock correction
    """

    try:
        val=sbu.met2mjd(met_time, correct=True)
    except ModuleNotFoundError:
        _local_pfile_dir=Path(f"/tmp/met2mjd_{os.times().elapsed}")
        _local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(_local_pfile_dir))
        except AttributeError:
            hsp.utils.local_pfiles(par_dir=str(_local_pfile_dir))

        # calculate times in UTC and MJD units as well
        inputs = dict(intime=str(met_time), insystem="MET", informat="s", outsystem="UTC",
                      outformat="m")  # output in MJD
        o = hsp.swifttime(**inputs)
        val=o.params["outtime"]
        shutil.rmtree(_local_pfile_dir)

    atime = Time(val, format="mjd", scale='utc')
    return atime.value

def met2utc(met_time, mjd_time=None):

    """
    A convenience function that calculates the UTC time from a Swift MET time. Ths function first converts the time to
    MJD, which either uses the swiftbat code base which is quicker or the heasoftpy swifttime function which is slower,
    and then converts it to UTC. The user can also supply a MJD time to save on computational time.

    :param met_time: a number that is the Swift MET time that will be converted
    :param mjd_time: default to None, which means that the code will first calculate the MJD time and then convert it to
        UTC time. If the user already has the MJD time, they can specify it here and the function will directly convert it.
    :return: a numpy datetime64 object of the MET time with the Swift clock correction applied
    """
    if mjd_time is None:
        mjd_time=met2mjd(met_time)

    atime = Time(mjd_time, format="mjd", scale='utc')
    return atime.datetime64

def save_progress(obs_list):
    """
    Convience function to save progress for a list of BatSurvey observations

    :param obs_list: list of BatSurvey or MosaicBatSurvey objects
    :return: None
    """
    if type(obs_list) is not list:
        obs_list=[obs_list]

    for i in obs_list:
        i.save()

def set_pdir(pdir):
    """
    Sets the custom pfile directory for calling a heasoftpy function. This ensures that functions can be called in
    parallel. This is depreciated since heasoftpy v1.2.

    :param pdir: None, Path, or string to the custom pfiles directory. a value of None will force heasoftpy to create a
        custom pfiles directory in /tmp, as is specified in their documentation.
    :return:
    """

    #if its not None, make sure that its a string that can be passed to heasoftpy. None will
    if pdir is not None:
        pdir=str(pdir)

    try:
        hsp.local_pfiles(pfiles_dir=pdir)
    except AttributeError:
        hsp.utils.local_pfiles(par_dir=pdir)

def reset_pdir():
    """
    Resets the pfiles environment variable to what it originally was. This is depreciated since heasoftpy v1.2.

    :return:
    """
    os.environ['PFILES'] = _orig_pdir
