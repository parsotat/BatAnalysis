# Warning: running this script as is will produce ~2.5 TB of data. Make sure that your computer has this much storage, otherwise you can simply run part of this analysis

import glob
import os
import sys
import batanalysis as ba
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.io import fits
from pathlib import Path
import swiftbat
import swiftbat.swutil as sbu
import pickle

plt.ion()

# set the path with the BAT survey data
newdir = Path("/Users/tparsota/Documents/CRAB_SURVEY_DATA")
ba.datadir(newdir, mkdir=True)

# query heasarc for all the data within the time period of interest and download it
object_name = "Crab_Nebula_Pulsar"
queryargs = dict(time="2004-12-15 .. 2006-10-27", fields="All", resultmax=0)

# use swiftbat to create a bat source object
object_location = swiftbat.simbadlocation("Crab")
object_batsource = swiftbat.source(
    ra=object_location[0], dec=object_location[1], name=object_name
)
table_everything = ba.from_heasarc(name=None, **queryargs)
minexposure = 1000  # cm^2 after cos adjust

# calculate the exposure with partial coding
exposures = np.array(
    [object_batsource.exposure(ra=row["RA"],
                               dec=row["DEC"], 
                               roll=row["ROLL_ANGLE"])[0]
        for row in table_everything
    ])

# select the observations that have greater than the minimum desired exposure
table_exposed = table_everything[exposures > minexposure]
print(
    f"Finding everything finds {len(table_everything)} observations, of which {len(table_exposed)} have more than {minexposure:0} cm^2 coded"
)

# download the data
# result = ba.download_swiftdata(table_exposed)

# get a list of the fully downloaded observation IDs
obs_ids = [i for i in table_exposed["OBSID"] if result[i]["success"]]

# To reload data wthout querying the database again
# obs_ids=[i.name for i in sorted(ba.datadir().glob("*")) if i.name.isnumeric()]

# run batsurvey in parallel with pattern maps
input_dict = dict(
    cleansnr=6, cleanexpr="ALWAYS_CLEAN==T"
)  # this is set by default but we are explicily mentioning this here for the user's knowledge that these parameters are passed to HEASoftpy's batsurvey
noise_map_dir = Path("/Users/tparsota/Documents/PATTERN_MAPS/")
batsurvey_obs = ba.parallel.batsurvey_analysis(
    obs_ids, input_dict=input_dict, patt_noise_dir=noise_map_dir, nprocs=20
)

# creat the pha files and the appropriate rsp file in parallel.
# use xspec to fit each spectrum with a default powerlaw spectrum
batsurvey_obs = ba.parallel.batspectrum_analysis(
    batsurvey_obs, object_name, ul_pl_index=2.15, recalc=True, nprocs=14
)

# plot the snapshot pointing values of rate, snr, and the fitted flux and photon index
fig, axes = ba.plot_survey_lc(
    batsurvey_obs,
    id_list=object_name,
    time_unit="UTC",
    values=["rate", "snr", "flux", "PhoIndex", "exposure"],
    calc_lc=True,
)

# combine all the pointings into a single file to sort into binned fits files
outventory_file = ba.merge_outventory(batsurvey_obs)

# when loading information:
# outventory_file=Path("./path/to/outventory_all.fits")

# bin into 1 month cadence
time_bins = ba.group_outventory(
    outventory_file, np.timedelta64(1, "M"), end_datetime=Time("2006-10-27")
)

# do the parallel construction of each mosaic for each time bin
mosaic_list, total_mosaic = ba.parallel.batmosaic_analysis(
    batsurvey_obs, outventory_file, time_bins, nprocs=8
)

mosaic_list = ba.parallel.batspectrum_analysis(
    mosaic_list, object_name, recalc=True, nprocs=11
)
total_mosaic = ba.parallel.batspectrum_analysis(
    total_mosaic, object_name, use_cstat=False, recalc=True, nprocs=1
)

# bin into weekly cadence too
outventory_file_weekly = ba.merge_outventory(
    batsurvey_obs, savedir=Path("./weekly_mosaiced_surveyresults/")
)
time_bins_weekly = ba.group_outventory(
    outventory_file_weekly,
    np.timedelta64(1, "W"),
    start_datetime=Time("2004-12-01"),
    end_datetime=Time("2006-10-27"),
)
weekly_mosaic_list, weekly_total_mosaic = ba.parallel.batmosaic_analysis(
    batsurvey_obs, outventory_file_weekly, time_bins_weekly, nprocs=8
)

weekly_mosaic_list = ba.parallel.batspectrum_analysis(
    weekly_mosaic_list, object_name, recalc=True, nprocs=11
)
weekly_total_mosaic = ba.parallel.batspectrum_analysis(
    weekly_total_mosaic, object_name, recalc=True, use_cstat=False, nprocs=1
)


# look at the results preliminarily
fig, axes = ba.plot_survey_lc(
    mosaic_list,
    id_list=object_name,
    time_unit="UTC",
    values=["rate", "snr", "flux", "PhoIndex", "exposure"],
    calc_lc=True,
)


# save the data for plotting
all_data = ba.concatenate_data(
    batsurvey_obs,
    object_name,
    ["met_time", "utc_time", "exposure", "rate", "rate_err", "snr", "flux", "PhoIndex"],
)
with open("all_data_dictionary.pkl", "wb") as f:
    pickle.dump(all_data, f)

all_data_monthly = ba.concatenate_data(
    mosaic_list,
    object_name,
    [
        "user_timebin/met_time",
        "user_timebin/utc_time",
        "user_timebin/met_stop_time",
        "user_timebin/utc_stop_time",
        "rate",
        "rate_err",
        "snr",
        "flux",
        "PhoIndex",
    ],
)
with open("monthly_mosaic_dictionary.pkl", "wb") as f:
    pickle.dump(all_data_monthly, f)

all_data_weekly = ba.concatenate_data(
    weekly_mosaic_list,
    object_name,
    [
        "user_timebin/met_time",
        "user_timebin/utc_time",
        "user_timebin/met_stop_time",
        "user_timebin/utc_stop_time",
        "rate",
        "rate_err",
        "snr",
        "flux",
        "PhoIndex",
    ],
)
with open("weekly_mosaic_dictionary.pkl", "wb") as f:
    pickle.dump(all_data_weekly, f)

# create the plots in the paper
energy_range = None
time_unit = "MET"
values = ["rate", "snr", "flux", "PhoIndex"]

survey_obsid_list = [
    "all_data_dictionary_test",
    "monthly_mosaic_dictionary",
    "weekly_mosaic_dictionary_test",
]

obs_list_count = 0
for observation_list in survey_obsid_list:
    with open(observation_list + ".pkl", "rb") as f:
        all_data = pickle.load(f)
        data = all_data[object_name]

    # get the time centers and errors
    if "mosaic" in observation_list:
        if "MET" in time_unit:
            t0 = TimeDelta(data["user_timebin/met_time"], format="sec")
            tf = TimeDelta(data["user_timebin/met_stop_time"], format="sec")
        elif "MJD" in time_unit:
            t0 = Time(data[time_str_start], format="mjd")
            tf = Time(data[time_str_end], format="mjd")
        else:
            t0 = Time(data["user_timebin/utc_time"])
            tf = Time(data["user_timebin/utc_stop_time"])
    else:
        if "MET" in time_unit:
            t0 = TimeDelta(data["met_time"], format="sec")
        elif "MJD" in time_unit:
            t0 = Time(data[time_str_start], format="mjd")
        else:
            t0 = Time(data["utc_time"])
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
        time_diff = np.array([np.timedelta64(0.5 * i.to_datetime()) for i in dt])

    x = time_center
    xerr = time_diff

    if obs_list_count == 0:
        fig, axes = plt.subplots(len(values), sharex=True, figsize=(10, 12))

    axes_queue = [i for i in range(len(values))]
    # plot_value=[i for i in values]

    e_range_str = f"{14}-{195} keV"
    # axes[0].set_title(object_name + '; survey data from ' + e_range_str)

    for i in values:
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
            yerr[:, y_upperlim] = 0.1 * y[y_upperlim]

        if "mosaic" in observation_list:
            if "weekly" in observation_list:
                zorder = 9
                c = "blue"
                m = "o"
                l = "Weekly Mosaic"
                ms = 5
                a = 0.8
            else:
                zorder = 9
                c = "green"
                m = "s"
                l = "Monthly Mosaic"
                ms = 7
                a = 1
        else:
            zorder = 4
            c = "gray"
            m = "."
            l = "Survey Snapshot"
            ms = 3
            a = 0.3

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            uplims=y_upperlim,
            linestyle="None",
            marker=m,
            markersize=ms,
            zorder=zorder,
            color=c,
            label=l,
            alpha=a,
        )

        if "flux" in i.lower():
            ax.set_yscale("log")

        if "snr" in i.lower():
            ax.set_yscale("log")

        ax.set_ylabel(label)

    # if T0==0:
    if "MET" in time_unit:
        label_string = "MET Time (s)"
    elif "MJD" in time_unit:
        label_string = "MJD Time (s)"
    else:
        label_string = "UTC Time (s)"

    plt.gca().ticklabel_format(useMathText=True)
    axes[-1].set_xlabel(label_string)

    obs_list_count += 1

# add the UTC times as well
met_values = [
    126230399.334,
    157766399.929,
]  # [i.get_position()[0] for i in axes[-1].get_xticklabels()]
utc_values = [np.datetime64(sbu.met2datetime(i)) for i in met_values]

for i, j in zip(met_values, [2005, 2006]):
    for ax in axes:
        ax.axvline(i, 0, 1, ls="--", color="k")
        if ax == axes[0]:
            ax.text(i, ax.get_ylim()[1] * 1.01, str(j), fontsize=14, ha="center")

axes[0].legend(loc="best")

axes[1].set_ylabel("SNR")
axes[2].set_ylabel(r"Flux (erg/s/cm$^2$)")
axes[3].set_ylabel(r"$\Gamma$")

for ax, l in zip(axes, ["a", "b", "c", "d"]):
    ax.text(
        0.99, 0.95, f"({l})", ha="right", va="top", transform=ax.transAxes, fontsize=14
    )

axes[-1].axhline(2.15, 0, 1)

axes[-2].axhline(23342.70e-12, 0, 1)

fig.tight_layout()
plot_filename = object_name + "_survey_lc.pdf"
fig.savefig(plot_filename, bbox_inches="tight")


values = ["rate"]
survey_obsid_list = ["monthly_mosaic_dictionary"]

obs_list_count = 0
for observation_list in survey_obsid_list:
    with open(observation_list + ".pkl", "rb") as f:
        all_data = pickle.load(f)
        data = all_data[object_name]

    # get the time centers and errors
    if "mosaic" in observation_list:
        if "MET" in time_unit:
            t0 = TimeDelta(data["user_timebin/met_time"], format="sec")
            tf = TimeDelta(data["user_timebin/met_stop_time"], format="sec")
        elif "MJD" in time_unit:
            t0 = Time(data[time_str_start], format="mjd")
            tf = Time(data[time_str_end], format="mjd")
        else:
            t0 = Time(data["user_timebin/utc_time"])
            tf = Time(data["user_timebin/utc_stop_time"])
    else:
        if "MET" in time_unit:
            t0 = TimeDelta(data["met_time"], format="sec")
        elif "MJD" in time_unit:
            t0 = Time(data[time_str_start], format="mjd")
        else:
            t0 = Time(data["utc_time"])
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
        time_diff = np.array([np.timedelta64(0.5 * i.to_datetime()) for i in dt])

    x = time_center
    xerr = time_diff

    if obs_list_count == 0:
        fig, axes = plt.subplots()

    axes_queue = [i for i in range(len(values))]
    # plot_value=[i for i in values]

    e_range_str = f"{14}-{195} keV"
    # axes[0].set_title(object_name + '; survey data from ' + e_range_str)

    for i in values:
        ax = axes  # [axes_queue[0]]
        # axes_queue.pop(0)

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
            yerr[:, y_upperlim] = 0.0 * y[y_upperlim]

        if "mosaic" in observation_list:
            if "weekly" in observation_list:
                zorder = 9
                c = "blue"
                m = "o"
                l = "BatAnalysis Weekly Mosaic"
                ms = 5
                a = 0.5
            else:
                zorder = 8
                c = "green"
                m = "s"
                l = "BatAnalysis Monthly Mosaic"
                ms = 7
                a = 0.8
        else:
            zorder = 4
            c = "gray"
            m = "."
            l = "BatAnalysis Survey Snapshot"
            ms = 3
            a = 0.3

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            uplims=y_upperlim,
            linestyle="None",
            marker=m,
            markersize=ms,
            zorder=zorder,
            color=c,
            label=l,
            alpha=a,
        )

        if "flux" in i.lower():
            ax.set_yscale("log")

        if "snr" in i.lower():
            ax.set_yscale("log")

        ax.set_ylabel(label)

    # if T0==0:
    if "MET" in time_unit:
        label_string = "MET Time (s)"
    elif "MJD" in time_unit:
        label_string = "MJD Time (s)"
    else:
        label_string = "UTC Time (s)"

    plt.gca().ticklabel_format(useMathText=True)
    axes.set_xlabel(label_string)

    obs_list_count += 1

# add the UTC times as well
met_values = [126230399.334, 157766399.929]
utc_values = [np.datetime64(sbu.met2datetime(i)) for i in met_values]

for i, j in zip(met_values, [2005, 2006]):
    ax.axvline(i, 0, 1, ls="--", color="k")
    ax.text(i, ax.get_ylim()[1] * 1.001, str(j), fontsize=14, ha="center")

# this file can be obtained from: https://swift.gsfc.nasa.gov/results/bs157mon/287
with fits.open("BAT_157m_eight_band_SWIFT_J0534.6+2204.lc.txt") as crab_file:
    data = crab_file[1].data
    idx = np.where(data["Mission_month"] < 24)
    survey_rates = data["RATE"][idx]
    survey_rates_err = data["RATE_ERR"][idx]
    survey_time = data["TIME"][idx]
    survey_dt = data["TIMEDEL"][idx]
    survey_exposure = data["EXPOSURE"][idx]

axes.errorbar(
    survey_time,
    survey_rates[:, -1],
    yerr=survey_rates_err[:, -1],
    color="red",
    marker="*",
    zorder=7,
    linestyle="None",
    markersize=13,
    label="Tueller et al. (2010)",
    alpha=1,
)

axes.legend(loc="best", fontsize=12)


fig.tight_layout()
plot_filename = object_name + "_survey_compare_lc.pdf"
fig.savefig(plot_filename, bbox_inches="tight")
