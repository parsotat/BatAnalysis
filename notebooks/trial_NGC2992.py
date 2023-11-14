import glob
import os
import sys
import batanalysis as ba
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.io import fits
from pathlib import Path
import swiftbat
import swiftbat.swutil as sbu
import pickle

from xspec import *

object_name = "NGC2992"
object_location = swiftbat.simbadlocation(object_name)
object_batsource = swiftbat.source(
    ra=object_location[0], dec=object_location[1], name=object_name
)

queryargs = dict(time="2004-12-15 .. 2005-12-16", fields="All", resultmax=0)
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

noise_map_dir = Path("/local/data/tparsota/PATTERN_MAPS/")
batsurvey_obs = ba.parallel.batsurvey_analysis(
    obs_ids, patt_noise_dir=noise_map_dir, nprocs=30
)

batsurvey_obs = ba.parallel.batspectrum_analysis(
    batsurvey_obs, object_name, ul_pl_index=1.9, use_cstat=True, nprocs=30
)

outventory_file = ba.merge_outventory(batsurvey_obs)
time_bins = ba.group_outventory(
    outventory_file,
    np.timedelta64(1, "M"),
    start_datetime=Time("2004-12-15"),
    end_datetime=Time("2005-12-16"),
)

mosaic_list, total_mosaic = ba.parallel.batmosaic_analysis(
    batsurvey_obs, outventory_file, time_bins, nprocs=3
)

mosaic_list = ba.parallel.batspectrum_analysis(
    mosaic_list, object_name, ul_pl_index=1.9, use_cstat=True, nprocs=5
)
total_mosaic = ba.parallel.batspectrum_analysis(
    total_mosaic, object_name, ul_pl_index=1.9, use_cstat=True, nprocs=1
)

fig, axes = ba.plot_survey_lc(
    [batsurvey_obs, mosaic_list],
    id_list=object_name,
    time_unit="UTC",
    values=["rate", "snr", "flux", "PhoIndex", "exposure"],
    same_figure=True,
)

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

# make the plot
energy_range = None
time_unit = "MET"
values = ["rate", "snr", "flux"]

survey_obsid_list = ["all_data_dictionary", "monthly_mosaic_dictionary"]

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
            yerr[:, y_upperlim] = 0.4 * y[y_upperlim]

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
utc_time = Time(["2005-01-01", "2006-01-01"])
met_time = []
for i in utc_time:
    met_time.append(sbu.datetime2met(i.datetime, correct=True))

for i, j in zip(met_time, utc_time.ymdhms):
    for ax in axes:
        ax.axvline(i, 0, 1, ls="--", color="k")
        if ax == axes[0]:
            ax.text(
                i, ax.get_ylim()[1] * 1.03, f'{j["year"]}', fontsize=10, ha="center"
            )

axes[1].set_ylabel("SNR")
axes[2].set_ylabel(r"Flux (erg/s/cm$^2$)")

axes[1].legend(loc="lower center", ncol=2)

for ax, l in zip(axes, ["a", "b", "c", "d"]):
    ax.text(
        0.01, 0.95, f"({l})", ha="left", va="top", transform=ax.transAxes, fontsize=13
    )

fig.tight_layout()
plot_filename = object_name + "_survey_lc.pdf"
fig.savefig(plot_filename, bbox_inches="tight")

# create the plot of the time integrated spectrum
fig, ax = plt.subplots(1)
pha_file = total_mosaic.get_pha_filenames(id_list=object_name)[0]
emax = np.array(total_mosaic.emax)
emin = np.array(total_mosaic.emin)
ecen = 0.5 * (emin + emax)

os.chdir(pha_file.parent)

with fits.open(pha_file.name) as file:
    pha_data = file[1].data
    energies = file[-2].data

# get the xspec model info
mosaic_pointing_info = total_mosaic.get_pointing_info("mosaic", source_id=object_name)
xspec_session_name = mosaic_pointing_info["xspec_model"].name
flux = 10 ** mosaic_pointing_info["model_params"]["lg10Flux"]["val"]
flux_err = 10 ** np.array(
    [
        mosaic_pointing_info["model_params"]["lg10Flux"]["lolim"],
        mosaic_pointing_info["model_params"]["lg10Flux"]["hilim"],
    ]
)
flux_diff = np.abs(flux - flux_err)


phoindex = mosaic_pointing_info["model_params"]["PhoIndex"]["val"]
phoindex_err = np.array(
    [
        mosaic_pointing_info["model_params"]["PhoIndex"]["lolim"],
        mosaic_pointing_info["model_params"]["PhoIndex"]["hilim"],
    ]
)
phoindex_diff = np.abs(phoindex - phoindex_err)


xsp.Xset.restore(xspec_session_name)
xsp.Plot.device = "/null"
xsp.Plot("data resid")
energies = xsp.Plot.x()
edeltas = xsp.Plot.xErr()
rates = xsp.Plot.y(1, 1)
errors = xsp.Plot.yErr(1, 1)
foldedmodel = xsp.Plot.model()
dataLabels = xsp.Plot.labels(1)
residLabels = xsp.Plot.labels(2)

foldedmodel.append(foldedmodel[-1])
xspec_energy = total_mosaic.emin.copy()
xspec_energy.append(total_mosaic.emax[-1])
xspec_energy = np.array(xspec_energy)


ax.loglog(emin, pha_data["RATE"], color="k", drawstyle="steps-post")
ax.loglog(emax, pha_data["RATE"], color="k", drawstyle="steps-pre")
ax.errorbar(
    ecen,
    pha_data["RATE"],
    yerr=pha_data["STAT_ERR"],
    color="k",
    marker="None",
    ls="None",
    label=object_name + " 1 Year Mosaic Spectrum",
)
ax.set_ylabel("Count Rate (cts/s)", fontsize=14)
ax.set_xlabel("E (keV)", fontsize=14)

ax.tick_params(axis="both", which="major", labelsize=14)

l = (
    f"Folded Model:\nFlux={flux/1e-11:-.3}$^{{{flux_diff[1]/1e-11:+.3}}}_{{{-1*flux_diff[0]/1e-11:+.3}}} \\times 10^{{-11}}$ erg/s/cm$^2$"
    + f"\n$\Gamma$={phoindex:-.3}$^{{{phoindex_diff[1]:+.2}}}_{{{-1*phoindex_diff[0]:+.2}}}$"
)
ax.loglog(xspec_energy, foldedmodel, color="r", drawstyle="steps-post", label=l)
ax.legend(loc="best")

fig.tight_layout()
fig.savefig(object_name + "_1year_spectrum.pdf")
