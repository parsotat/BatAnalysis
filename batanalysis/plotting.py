from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from .batlib import met2utc, met2mjd, concatenate_data
from astropy.time import Time, TimeDelta
from .batproducts import Spectrum, Lightcurve
import astropy.units as u

# for python>3.6
try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)


def plot_survey_lc(
    survey_obsid_list,
    id_list,
    energy_range=[14, 195],
    savedir=None,
    time_unit="MET",
    values=["rate", "snr"],
    T0=None,
    same_figure=False,
):
    """
    Convenience function to plot light curves for a number of sources.

    :param survey_obsid_list: List of BatSurvey or MosaicBatSurvey objects, can also be a list of lists of BatSurvey or
        MosaicBatSurvey objects which the user would like to plot. By default each sublist will be plotted on its own
        set of axis.
    :param id_list: List of Strings or string denoting which source(s) the user wants the light curves to be plotted for
    :param energy_range: An array for the lower and upper energy ranges to be used.
        the default value is 14-195 keV.
    :param savedir: None or a String to denote whether the light curves should be saved
        (if the passed value is a string)
        or not. If the value is a string, the light curve plots are saved to the provided directory if it exists.
    :param time_unit: String specifying the time unit of the light curve. Can be "MET", "UTC" or "MJD"
    :param values: A list of strings contaning information that the user would like to be plotted out. The strings
        correspond to the keys in the pointing_info dictionaries of each BatSurvey object or to 'rate' or 'snr'.
    :param T0: None or a MET time of interest that should be highlighted on the plot.
    :param same_figure: Boolean to denote if the passed in list of BatSurvey lists should be plotted on the same set of
        axis, alongside one another. Default is False.
    :return: None
    """

    # save this for use later
    base_values = values.copy()

    if "MET" not in time_unit and "UTC" not in time_unit and "MJD" not in time_unit:
        raise ValueError(
            "This function plots survey data only using MET, UTC, or MJD time"
        )

    # determine if the user wants to plot a specific object or list of objects
    if type(id_list) is not list:
        # it is a single string:
        id_list = [id_list]

    if type(survey_obsid_list[0]) is not list:
        survey_obsid_list = [survey_obsid_list]

    # if the user wants to save the plots check if the save dir exists
    if savedir is not None:
        savedir = Path(savedir)
        if not savedir.exists():
            raise ValueError(f"The directory {savedir} does not exist")

    # loop through each object and plot its light curve
    for source in id_list:
        obs_list_count = 0
        for observation_list in survey_obsid_list:
            values = base_values.copy()

            # collect all the data that we need
            time_str_start = ""
            if "mosaic" in observation_list[0].pointing_ids:
                time_str_start += "user_timebin/"

            if "MET" in time_unit:
                time_str_start += "met_time"
            elif "MJD" in time_unit:
                time_str_start += "mjd_time"
            else:
                time_str_start += "utc_time"
            values.insert(0, time_str_start)

            if "mosaic" in observation_list[0].pointing_ids:
                # need to get the stop time
                time_str_end = "user_timebin/"
                if "MET" in time_unit:
                    time_str_end += "met_stop_time"
                elif "MJD" in time_unit:
                    time_str_end += "mjd_stop_time"
                else:
                    time_str_end += "utc_stop_time"

                values.append(time_str_end)
            else:
                if "exposure" not in values:
                    # need to get the exposure without duplication
                    values.append("exposure")

            # if we want to plot the rate we also need the rate_err
            if "rate" in values and "rate_err" not in values:
                values.append("rate_err")

            # get all the data that we need
            all_data = concatenate_data(
                observation_list, source, values, energy_range=energy_range
            )
            data = all_data[source]

            if T0 is None:
                T0 = 0.0  # in MET time, incase we have a time that we are interested in

            # get the time centers and errors
            if "mosaic" in observation_list[0].pointing_ids:
                if "MET" in time_unit:
                    t0 = TimeDelta(data[time_str_start], format="sec")
                    tf = TimeDelta(data[time_str_end], format="sec")
                elif "MJD" in time_unit:
                    t0 = Time(data[time_str_start], format="mjd")
                    tf = Time(data[time_str_end], format="mjd")
                else:
                    t0 = Time(data[time_str_start])
                    tf = Time(data[time_str_end])
            else:
                if "MET" in time_unit:
                    t0 = TimeDelta(data[time_str_start], format="sec")
                elif "MJD" in time_unit:
                    t0 = Time(data[time_str_start], format="mjd")
                else:
                    t0 = Time(data[time_str_start])
                tf = t0 + TimeDelta(data["exposure"], format="sec")

            dt = tf - t0

            if "MET" in time_unit:
                time_center = 0.5 * (tf + t0).value
                time_diff = 0.5 * (tf - t0).value
            elif "MJD" in time_unit:
                time_diff = 0.5 * (tf - t0)
                time_center = t0 + time_diff
                time_center = time_center.value
                time_diff = time_diff.value

            else:
                time_diff = TimeDelta(0.5 * dt)  # dt.to_value('datetime')
                time_center = t0 + time_diff

                time_center = np.array([i.to_value("datetime64") for i in time_center])
                time_diff = np.array(
                    [np.timedelta64(0.5 * i.to_datetime()) for i in dt]
                )

            x = time_center
            xerr = time_diff

            if not same_figure:
                fig, axes = plt.subplots(len(base_values), sharex=True)
            else:
                if obs_list_count == 0:
                    fig, axes = plt.subplots(len(base_values), sharex=True)

            axes_queue = [i for i in range(len(base_values))]

            e_range_str = f"{np.min(energy_range)}-{np.max(energy_range)} keV"
            axes[0].set_title(source + "; survey data from " + e_range_str)

            for i in base_values:
                ax = axes[axes_queue[0]]
                axes_queue.pop(0)

                y = data[i]
                yerr = np.zeros(x.size)
                y_upperlim = np.zeros(x.size)

                label = i

                if "rate" in i:
                    yerr = data[i + "_err"]
                    label = "Count rate (cts/s)"
                elif i + "_lolim" in data.keys():
                    # get the errors
                    lolim = data[i + "_lolim"]
                    hilim = data[i + "_hilim"]

                    yerr = np.array([lolim, hilim])
                    y_upperlim = data[i + "_upperlim"]

                    # find where we have upper limits and set the error to 1 since the nan error value isnt
                    # compatible with upperlimits
                    yerr[:, y_upperlim] = 0.2 * y[y_upperlim]

                if "mosaic" in observation_list[0].pointing_ids:
                    zorder = 10
                    c = "red"
                else:
                    zorder = 5
                    c = "gray"

                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    uplims=y_upperlim,
                    linestyle="None",
                    marker="o",
                    markersize=3,
                    zorder=zorder,
                )  # color="red"

                if "flux" in i.lower():
                    ax.set_yscale("log")

                ax.set_ylabel(label)

            mjdtime = met2mjd(T0)
            utctime = met2utc(T0, mjd_time=mjdtime)

            if "MET" in time_unit:
                label_string = "MET Time (s)"
                val = T0
                l = f"T0={val:.2f}"
            elif "MJD" in time_unit:
                label_string = "MJD Time (s)"
                val = mjdtime
                l = f"T0={val:.2f}"
            else:
                label_string = "UTC Time (s)"
                val = utctime
                l = f"T0={val}"

            if not same_figure:
                l = l
            else:
                if obs_list_count != 0:
                    l = ""

            if T0 != 0:
                for ax in axes:
                    ax.axvline(val, 0, 1, ls="--", label=l)
                axes[0].legend(loc="best")

            axes[-1].set_xlabel(label_string)

            if savedir is not None and not same_figure:
                plot_filename = source + "_survey_lc.pdf"
                fig.savefig(savedir.joinpath(plot_filename), bbox_inches="tight")

            obs_list_count += 1

        if savedir is not None and same_figure:
            plot_filename = source + "_survey_lc.pdf"
            fig.savefig(savedir.joinpath(plot_filename), bbox_inches="tight")

        plt.show()

    return fig, axes

def plot_TTE_lightcurve(lightcurves, spectra, values=["flux", "phoindex"], T0=None, time_unit="MET", is_relative=False, energy_range=[15,350]*u.keV):
    """
    This convenience function allows one to plot a set of lightcurves alongside a set of spectra. The spectra should all
    be fit with the same model.

    Thsi doesnt plot rate lightcurves

    :return:
    """

    #first see how many lightcurves we need to plot
    if type(lightcurves) is not list:
        lcs=[lightcurves]
    else:
        lcs=lightcurves

    if type(spectra) is not list:
        spect=[spectra]
    else:
        spect=spectra

    #do some error checking for types
    if np.any([not isinstance(i, Lightcurve) for i in lcs]):
        raise ValueError("Not all the elements of the values passed in to the lightcurves variable are Lightcurve objects.")

    if np.any([not isinstance(i, Spectrum) for i in spect]):
        raise ValueError("Not all the elements of the values passed in to the spectra variable are Spectrum objects.")

    #make sure all the spectra have been fitted with a model and that all the models are the same
    #TODO: get the times of the spectra here
    spect_models=[]
    for i in spect:
        try:
            spect_models.append(i.spectral_model)
        except AttributeError as e:
            raise AttributeError("Not all of the spectra that have been passed in have been fit with a spectral model")

    #if the user has passed in explicit values to plot we want to check for these, otherwise plot everything
    if values is None:
        template=None
    else:
        #if values has flux in it, change it to lg10flux isnce this is what xspec provides
        if "flux" in values:
            values[values.index("flux")]="lg10Flux"
        template=[i.lower() for i in values]
    for model in spect_models:
        #if we do not have an upper limit and the template is None save the parameters as the template model parameters
        if template is None and "nsigma_lg10flux_upperlim" not in model.keys():
            template = model["parameters"].keys()
        elif template is not None:
            #otherwise if we have a template, compare it to the model parameters
            if set(template)!=set([i.lower() for i in model["parameters"].keys()]):
                raise ValueError("The input spectra do not all have the same fitted model parameters (excluding spectra"
                                 " that were used to calculate flux upper limits.")


    #then see how many figure axes we need
    # one for LC and some number for the spectral ligthcurve parameters we want to plot
    num_ax=1+len(template)
    fig, axes = plt.subplots(len(num_ax), sharex=True)

    if len(lcs)>1 and energy_range.size>2:
        #we can only plot a single energy range for each LC otherwise we get plots that are too cluttered
        raise ValueError("There can only be a single energy range plotted when there are multiple lightcurves to plot")

    #plot the ligthcurves on the first axis
    #if a single lc is passed in and energy range is None, we want to plot all the energy bins of the Lightcurve object
    for lc in lcs:
        #need to get the times here for the LC
        if "MET" in time_unit:
            start_times = lc.tbins["TIME_START"]
            end_times = lc.tbins["TIME_STOP"]
            mid_times = lc.tbins["TIME_CENT"]
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
            start_times = met2mjd(lc.tbins["TIME_START"].value)
            end_times = met2mjd(lc.tbins["TIME_STOP"].value)
            mid_times = met2mjd(lc.tbins["TIME_CENT"].value)
            xlabel = "MJD"

            if plot_relative:
                if T0 is None:
                    raise ValueError('The plot_relative value is set to True however there is no T0 that is defined ' +
                                     '(ie the time from which the time bins are defined relative to is not specified).')
                else:
                    raise NotImplementedError("plot_relative with MJD time unit is not implemented at this time.")
        else:
            start_times = met2utc(lc.tbins["TIME_START"])
            end_times = met2utc(lc.tbins["TIME_STOP"])
            mid_times = met2utc(lc.tbins["TIME_CENT"].value)
            xlabel = "UTC"

            if plot_relative:
                if T0 is None:
                    raise ValueError('The plot_relative value is set to True however there is no T0 that is defined ' +
                                     '(ie the time from which the time bins are defined relative to is not specified).')
                else:
                    raise ValueError("plot_relative with UTC time unit is not a permitted combination.")


        for e_idx, emin, emax in zip(lc.ebins["INDEX"], lc.ebins["E_MIN"], lc.ebins["E_MAX"]):
            plotting=True
            if energybins is not None:
                # need to see if the energy range is what the user wants
                if emin == energybins.min() and emax == energybins.max():
                    plotting = True
                else:
                    plotting = False

            if plotting:
                # use the proper indexing for the array
                if len(lc.ebins["INDEX"]) > 1:
                    rate = lc.data[data_key][:, e_idx]
                    rate_error = self.data["ERROR"][:, e_idx]
                    l = f'{self.ebins["E_MIN"][e_idx].value}-{self.ebins["E_MAX"][e_idx].value} ' + f'{self.ebins["E_MAX"][e_idx].unit}'
                else:
                    rate = self.data[data_key]
                    rate_error = self.data["ERROR"]
                    l = f'{self.ebins["E_MIN"][0].value}-{self.ebins["E_MAX"][0].value} ' + f'{self.ebins["E_MAX"].unit}'

                line = ax_rate.plot(start_times, rate, ds='steps-post')
                line_handle, = ax_rate.plot(end_times, rate, ds='steps-pre', color=line[-1].get_color(), label=l)
                all_lines.append(line_handle)
                all_labels.append(l)
                ax_rate.errorbar(mid_times, rate, yerr=rate_error, ls='None', color=line[-1].get_color())

                axes[0].plot()




    return fig, axes
