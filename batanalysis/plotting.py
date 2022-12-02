from matplotlib import pyplot as plt
import os
import glob
import numpy as np
from astropy.io import fits
from pathlib import Path
from .batlib import combine_survey_lc, read_lc_data,met2utc, met2mjd
from astropy.time import Time, TimeDelta

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)

def plot_survey_lc(survey_obsid_list, id_list=None, energy_range=None, savedir=None, time_unit="MET", \
                   values=["rate","snr"], T0=None, calc_lc=False, clean_dir=False):
    """
    Convenience function to plot light curves for a number of sources.

    :param survey_obsid_list: List of BatSurvey objects
    :param id_list: List of Strings or None Denoting which sources the user wants the light curves to be plotted for
    :param energy_range: None or String to denote the energy range that the user would like to use to calculate the light
        curve. The appropriate string values are: '14-20 keV', '20-24 keV', '24-35 keV', '35-50 keV', '50-75 keV', '75-100 keV', '100-150 keV',
        '150-195 keV'. A value of None means calculate the light curve for 14-195 keV.
    :param savedir: None or a String to denote whether the light curves should be saved (if the passed value is a string)
        or not. If the value is a string, the light curve plots are saved to the provided directory if it exists.
    :param time_unit: String specifying the time unit of the light curve. Can be "MET", "UTC" or "MJD"
    :param values: A list of strings contaning information that the user would like to be plotted out. The strings
        correspond to the keys in the pointing_info dictionaries of each BatSurvey object or to 'rate' or 'snr'.
    :param calc_lc: Boolean set to True, to denote if the light curves across all the BatSurvey objects need to be recombined
    :param clean_dir: Boolean set to True by default. Denotes if the whole directory that holds all the compiled light curve
        data for the passed survey observations should be deleted and recreated if the directory exists.
    :return: None
    """

    image_energy = ['14-20 keV', '20-24 keV', '24-35 keV', '35-50 keV', '50-75 keV', '75-100 keV', '100-150 keV',
                    '150-195 keV']

    if "MET" not in time_unit and "UTC" not in time_unit and "MJD" not in time_unit:
        raise ValueError("This function plots survey data only using MET, UTC, or MJD time")

    #determine the energy range that the user wants
    if energy_range is None:
        e_range_idx=None #'14-195 keV'
    else:
        if energy_range in image_energy:
            e_range_idx=[i for i,j in enumerate(image_energy) if (energy_range in j)][0]
            #e_range_idx=image_energy[e_range_idx]
        else:
            raise ValueError("energy range can only be set to None or one of the following strings:", image_energy)

    if calc_lc:
        lc_dir=combine_survey_lc(survey_obsid_list, clean_dir=clean_dir)
    else:
        # get the main directory where we shoudl create the total_lc directory
        #main_dir = os.path.split(survey_obsid_list[0].result_dir)[0]
        lc_dir = survey_obsid_list[0].result_dir.parent.joinpath("total_lc") #os.path.join(main_dir, "total_lc")

    #determine if the user wants to plot a specific object or list of objects
    if id_list is None:
        # use the ids from the *.cat files produced, these are ones that have been identified in the survey obs_id
        #x = glob.glob(os.path.join(lc_dir, '*.cat'))
        #id_list = [os.path.basename(i).split('.cat')[0] for i in x]
        x = sorted(lc_dir.glob('*.cat'))
        id_list = [i.stem for i in x]
    else:
        if type(id_list) is not list:
            # it is a single string:
            id_list = [id_list]

    #if the user wants to save the plots check if the save dir exists
    if savedir is not None:
        savedir=Path(savedir)
        if not savedir.exists():
            raise ValueError(f"The directory {savedir} does not exist" )


    #loop through each object and plot its light curve
    for id in id_list:
        if T0 is None:
            T0 = 0.0 #in MET time, incase we have a time that we are interested in
        filename = lc_dir.joinpath(f'{id}.cat')  #os.path.join(lc_dir,id + '.cat')
        time_input, time_err_input, rate, rate_err, snr=read_lc_data(filename, e_range_idx, T0=0.0)

        if "MET" in time_unit:
            time=time_input
            time_err=time_err_input
        else:
            #need to set up date/time object for plotting
            #time_input and time_err_input are in units of seconds in MET time
            if "UTC" in time_unit:
                time=np.zeros_like(time_input, dtype='datetime64[ns]')
                time_err =np.zeros_like(time_input,dtype='timedelta64[ns]')
            else:
                time = np.zeros_like(time_input)
                time_err = np.zeros_like(time_input)
            count=0
            for t,t_err in zip(time_input, time_err_input):
                #calculate times in UTC and MJD units as well
                mjdtime = met2mjd(t)

                #inputs = dict(intime=str(t), insystem="MET", informat="s", outsystem="UTC", outformat="m") #output in MJD
                #o=hsp.swifttime(**inputs)
                #atime = Time(o.params["outtime"], format="mjd", scale='utc')
                dt = TimeDelta(t_err, format='sec')
                if "UTC" in time_unit:
                    utctime = met2utc(t, mjd_time=mjdtime)
                    time[count]=utctime
                    time_err[count]=dt.to_value('datetime')
                else:
                    time[count] = mjdtime
                    time_err[count]=dt.to_value('jd')
                count+=1

        fig, axes = plt.subplots(len(values), sharex=True)
        axes_queue = [i for i in range(len(values))]
        plot_value=[i for i in values]



        if energy_range is None:
            e_range_str = '14-195 keV'
        else:
            e_range_str=image_energy[e_range_idx]
        axes[0].set_title(id + '; survey data from ' + e_range_str)

        for i in plot_value:
            ax = axes[axes_queue[0]]
            axes_queue.pop(0)

            if 'rate' in i:
                y=rate
                yerr=rate_err
                label='count rate [1/s]'
                x=time
                xerr=time_err
            elif 'snr' in i:
                y=snr
                label='snr'
                x=time
                xerr = time_err
            else:
                #accumulate data from survey objects
                y=[]
                yerr=[]
                y_upperlim=[]
                obs_time=[]
                obs_time_err=[]
                if 'lg10' in i:
                    label=i.split('lg10')[-1]
                else:
                    label=i

                for obs in survey_obsid_list:
                    # sort the pointing IDs too
                    sorted_pointing_ids = np.sort(obs.pointing_ids)

                    for pointings in sorted_pointing_ids:

                        if "MET" in time_unit:
                            t_0 = obs.get_pointing_info(pointings)['met_time']
                            if 'mosaic' not in pointings:
                                t_f=t_0+obs.get_pointing_info(pointings)['exposure']
                            else:
                                t_f = t_0 + obs.get_pointing_info(pointings)['elapse_time']

                            t=0.5*(t_0+t_f)
                            dt=0.5*(t_f-t_0)
                        else:
                            t_0 = obs.get_pointing_info(pointings)['mjd_time']
                            if 'mosaic' not in pointings:
                                dt = TimeDelta(obs.get_pointing_info(pointings)['exposure'], format='sec')
                            else:
                                dt = TimeDelta(obs.get_pointing_info(pointings)['elapse_time'], format='sec')
                            t_f = t_0 + dt.to_value('jd')
                            t = 0.5 * (t_0 + t_f)
                            if "UTC" in time_unit:
                                t=Time(t, format="mjd", scale='utc').datetime64
                                dt=0.5*np.timedelta64(dt.to_value('datetime'))#need this to be half since error_bar plots t+/-dt
                            else:
                                dt = 0.5 * (t_f - t_0)

                        obs_time.append(t)
                        obs_time_err.append(dt)


                        if id in obs.get_pointing_info(pointings).keys():
                            # get the model component names
                            pointing_dict = obs.get_pointing_info(pointings, source_id=id)
                            # xsp.Xset.restore(pointing_dict['xspec_model'])
                            # model = xsp.AllModels(1)
                            model = pointing_dict["model_params"]

                            model_names = model.keys()
                            #stop

                            # get the real key we need if its a xspec model parameter
                            is_model_param=False
                            for key in model_names:
                                if i.capitalize() in key or i in key:
                                    model_param_key=key
                                    is_model_param=True

                            if i in obs.get_pointing_info(pointings).keys():
                                #outstr += "\t%s" % (str(obs.pointing_info[pointings][i]))
                                y.append(obs.pointing_info[pointings][i])
                                yerr.append(np.nan)
                            elif i in obs.get_pointing_info(pointings, source_id=id).keys():
                                #outstr += "\t%s" % (str(obs.pointing_info[pointings][source_id][i]))
                                y.append(obs.pointing_info[pointings][id][i])
                                yerr.append(np.nan)
                            elif is_model_param or ("flux" in i or "Flux" in i):
                                #see if the user wants the flux and if there is an upper limit available
                                if ("flux" in i or "Flux" in i) and "nsigma_lg10flux_upperlim" in pointing_dict.keys():
                                    #outstr += "\t  %e  "%(pointing_dict["nsigma_lg10flux_upperlim"])
                                    y.append(10**pointing_dict["nsigma_lg10flux_upperlim"])
                                    yerr.append(np.nan)

                                    y_upperlim.append(1)
                                else:
                                    #get the value and errors if the error calculation worked properly
                                    val=model[model_param_key]["val"]
                                    if ("flux" in i or "Flux" in i):
                                        y.append(10**model[model_param_key]["val"])
                                    else:
                                        y.append(model[model_param_key]["val"])

                                    if 'T' in model[model_param_key]["errflag"]:
                                        err_val="nan"
                                        errs = np.array(["nan", "nan"])
                                        #outstr += "\t%s-%s\+%s" % (val, errs[0], errs[1])
                                        yerr.append(np.nan)
                                    else:
                                        errs = np.array([model[model_param_key]["lolim"], model[model_param_key]["hilim"]])
                                        err_val=np.abs(val - errs).max() #"%e"%(np.abs(val - errs).max())
                                        #outstr += "\t%e-%e\+%e"%(val,errs[0], errs[1])
                                        if ("flux" in i or "Flux" in i):
                                            err_val=0.5 * (((10 ** errs[1]) - (10 ** val)) + ((10 ** val) - (10 ** errs[0])))
                                        yerr.append(err_val)

                                    y_upperlim.append(0)

                            else:
                                #outstr += "\tnan"
                                y.append(np.nan)
                                yerr.append(np.nan)
                        else:
                            # outstr += "\tnan"
                            y.append(np.nan)
                            yerr.append(np.nan)
                x=obs_time
                xerr=obs_time_err


            if 'snr' in i:
                ax.plot(x, y, 'ro', zorder=10)
            else:
                if ("flux" in i or "Flux" in i):
                    uplims= np.array(y_upperlim)

                    #find where we have upper limits and set the error to 1 since the nan error value isnt
                    #compatible with upperlimits
                    idx=np.where(uplims==1)
                    yerr = np.array(yerr)
                    y = np.array(y)
                    yerr[idx]=0.1*y[idx]
                    #stop
                else:
                    uplims=np.zeros(len(xerr)) #np.zeros(len(xerr))

                #plot the lc
                ax.errorbar(x,y,xerr=xerr, yerr=yerr, uplims=uplims, linestyle="None", marker="o", markersize=3, color="red", zorder=10)

                if ("flux" in i or "Flux" in i):
                    ax.set_yscale('log')

            ax.set_ylabel(label)

        mjdtime = met2mjd(T0)
        utctime = met2utc(T0, mjd_time=mjdtime)

        #if T0==0:
        if "MET" in time_unit:
            label_string='MET Time (s)'
            val=T0
        elif "MJD" in time_unit:
            label_string = 'MJD Time (s)'
            val = mjdtime
        else:
            label_string = 'UTC Time (s)'
            val = utctime
        """
        else:
            #plot the time

            if "MET" in time_unit:
                val=T0
                label_string = 'Time since T0 (MET = ' + str(val) + ' s)'
            elif "MJD" in time_unit:
                val=mjdtime
                label_string = 'Time since T0 (MJD = ' + str(val) + ' s)'
            else:
                val=utctime
                label_string = 'Time since T0 (UTC = ' + str(val) + ' s)'
        """
        if T0 != 0:
            for ax in axes:
                ax.axvline(val, 0,1, ls='--', label=f"T0={val}")
            axes[0].legend(loc='best')

        axes[-1].set_xlabel(label_string)

        if savedir is not None:
            plot_filename=id+'_survey_lc.pdf'
            #fig.savefig(os.path.join(savedir, plot_filename), bbox_inches="tight")
            fig.savefig(savedir.joinpath(plot_filename), bbox_inches="tight")

    plt.show()

    return fig, axes

