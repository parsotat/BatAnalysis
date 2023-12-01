from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from .batlib import met2utc, met2mjd, concatenate_data
from astropy.time import Time, TimeDelta

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
