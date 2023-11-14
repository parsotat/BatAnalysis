# Running this script can produce ~330 GB of data in total. Please make sure that you have enough storage on your computer.
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
from matplotlib import ticker


object_name = "MAXI J0637-430"
queryargs = dict(time="2019-11-01 .. 2020-01-30", fields="All", resultmax=0)
object_location = swiftbat.simbadlocation(object_name)
object_batsource = swiftbat.source(
    ra=object_location[0], dec=object_location[1], name=object_name
)
table_everything = ba.from_heasarc(**queryargs)
minexposure = 1000  # cm^2 after cos adjust
exposures = np.array(
    [object_batsource.exposure(ra=row["RA"],
                               dec=row["DEC"], 
                               roll=row["ROLL_ANGLE"])[0]
        for row in table_everything
    ])

table_exposed = table_everything[exposures > minexposure]
print(
    f"Finding everything finds {len(table_everything)} observations, of which {len(table_exposed)} have more than {minexposure:0} cm^2 coded"
)

# result = ba.download_swiftdata(table_exposed)
obs_ids = [i for i in table_exposed["OBSID"] if result[i]["success"]]

# The below excluded observation IDs had too few detectors which led to batcelldetect not being able to analyze the background variance within certain energy bins. This led to batsurvey being stuck when analyzing these observation IDs.
# obs_ids=[i.parent.name.split("_")[0] for i in sorted(ba.datadir().glob("*_surveyresult/batsurvey.pickle"))]

# incat=ba.create_custom_catalog(object_name, 99.09830, -42.86781 ,251.51841, -20.67087)
incat = Path("./custom_catalog.cat")

input_dict = dict(
    cleansnr=6,
    cleanexpr="ALWAYS_CLEAN==T",
    incatalog=f"{incat}",
    detthresh=8000,
    detthresh2=8000,
)
noise_map_dir = Path("/local/data/tparsota/PATTERN_MAPS/")
batsurvey_obs = ba.parallel.batsurvey_analysis(
    obs_ids, input_dict=input_dict, patt_noise_dir=noise_map_dir, nprocs=10
)

batsurvey_obs = ba.parallel.batspectrum_analysis(
    batsurvey_obs, object_name, use_cstat=True, ul_pl_index=2, nprocs=14
)


outventory_file = ba.merge_outventory(batsurvey_obs)
time_bins = ba.group_outventory(outventory_file, np.timedelta64(1, "W"))
mosaic_list, total_mosaic = ba.parallel.batmosaic_analysis(
    batsurvey_obs, outventory_file, time_bins, catalog_file=incat, nprocs=3
)

mosaic_list = ba.parallel.batspectrum_analysis(
    mosaic_list, object_name, use_cstat=True, ul_pl_index=2, recalc=True, nprocs=5
)
total_mosaic = ba.parallel.batspectrum_analysis(
    total_mosaic, object_name, use_cstat=True, ul_pl_index=2, recalc=True, nprocs=1
)

# take a look at the data quickly
fig, axes = ba.plot_survey_lc(
    [batsurvey_obs, mosaic_list],
    id_list=object_name,
    time_unit="UTC",
    values=["rate", "snr", "flux", "PhoIndex", "exposure"],
    same_figure=True,
)


# save the data in a dictionary for convenient custom plotting for publication quality figures
all_data = ba.concatenate_data(
    batsurvey_obs,
    object_name,
    ["met_time", "utc_time", "exposure", "rate", "rate_err", "snr", "flux", "PhoIndex"],
)

with open("all_data_dictionary.pkl", "wb") as f:
    pickle.dump(all_data, f)

all_data_weekly = ba.concatenate_data(
    mosaic_list,
    object_name,
    [
        "user_timebin/met_time",
        "user_timebin/met_stop_time",
        "user_timebin/utc_time",
        "user_timebin/utc_stop_time",
        "exposure",
        "rate",
        "rate_err",
        "snr",
        "flux",
        "PhoIndex",
    ],
)

with open("weekly_mosaic_dictionary.pkl", "wb") as f:
    pickle.dump(all_data_weekly, f)

# make the plot
energy_range = None
time_unit = "MET"
values = ["rate", "snr", "flux"]

survey_obsid_list = ["all_data_dictionary", "weekly_mosaic_dictionary"]

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
            t0 = Time(data["user_timebin/utc_time"]).mjd
            tf = Time(data["user_timebin/utc_stop_time"]).mjd
        else:
            t0 = Time(data["user_timebin/utc_time"])
            tf = Time(data["user_timebin/utc_stop_time"])
    else:
        if "MET" in time_unit:
            t0 = TimeDelta(data["met_time"], format="sec")
        elif "MJD" in time_unit:
            t0 = Time(data["utc_time"]).mjd
        else:
            t0 = Time(data["utc_time"])

        if "MJD" in time_unit:
            tf = t0 + TimeDelta(data["exposure"], format="sec").jd
        else:
            tf = t0 + TimeDelta(data["exposure"], format="sec")

    dt = tf - t0

    if "MET" in time_unit:
        time_center = 0.5 * (tf + t0).value
        time_diff = 0.5 * (tf - t0).value
    elif "MJD" in time_unit:
        if "mosaic" in observation_list:
            time_diff = 0.5 * (tf - t0)
            time_center = t0 + time_diff

            time_center = time_center
            time_diff = time_diff
        else:
            time_center = 0.5 * (tf + t0)
            time_diff = 0.5 * (tf - t0)

    else:
        time_diff = TimeDelta(0.5 * dt)  # dt.to_value('datetime')
        time_center = t0 + time_diff

        time_center = np.array([i.to_value("datetime64") for i in time_center])
        time_diff = np.array([np.timedelta64(0.5 * i.to_datetime()) for i in dt])

    x = time_center
    xerr = time_diff

    if obs_list_count == 0:
        fig, axes = plt.subplots(len(values), sharex=True)  # , figsize=(10,12))

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
            yerr[:, y_upperlim] = 0.2 * y[y_upperlim]

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

        # plt.gca().ticklabel_format(useMathText=True)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

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

    axes[-1].set_xlabel(label_string)

    obs_list_count += 1


# add the UTC times as well
utc_time = Time(["2019-11-01", "2019-12-01", "2020-01-01", "2020-01-30"])
met_time = []
for i in utc_time:
    met_time.append(sbu.datetime2met(i.datetime, correct=True))

for i, j in zip(met_time, utc_time.ymdhms):
    for ax in axes:
        ax.axvline(i, 0, 1, ls="--", color="k")
        if ax == axes[0]:
            ax.text(
                i,
                ax.get_ylim()[1] * 1.03,
                f'{j["year"]}-{j["month"]}-{j["day"]}',
                fontsize=13,
                ha="center",
            )

axes[1].set_ylabel("SNR")
axes[2].set_ylabel(r"Flux (erg/s/cm$^2$)")

axes[1].legend(loc="lower center", ncol=2)

for ax, l in zip(axes, ["a", "b", "c", "d"]):
    ax.text(
        1.0, 0.95, f"({l})", ha="right", va="top", transform=ax.transAxes, fontsize=13
    )

fig.tight_layout()
plot_filename = object_name + "_survey_lc.pdf"
fig.savefig(plot_filename, bbox_inches="tight")
