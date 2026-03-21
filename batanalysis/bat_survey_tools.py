"""
Python translation of HEASoft batsurvey Perl driver (v6.33.x style).

This module mirrors the control-flow of the original `batsurvey` script:
- energy rebinning
- GTI filtering
- per-snapshot loop
- iterative cleaning / imaging / source-detection
- point/obs statistics products

It intentionally shells out to the same HEASoft tools (baterebin, batsurvey-gti,
batbinevt, batclean, batfftimage, batcelldetect, ftcopy, ftimgcalc, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil
import subprocess
import time
import copy
import glob
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy import units as u
from .bat_truncated import identify_truncated_images, patch_truncated_results
from .batlib import convert_met_to_utc
from ._version import __version__ as batsurvey_version

try:
    import heasoftpy.swift as hsp  # type: ignore
    from heasoftpy import heatools  # type: ignore
    import heasoftpy as hsp
    import heasoftpy.utils as hsp_util
    import heasoftpy as hsp_core
    try:
        hsp.Config.allow_failure = True
    except AttributeError as e:
        print(f"heasoftpy version {hsp.__version__} doesnt allow for the Config syntax. Now disabling failure messages.")
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "heasoftpy is not installed. Install HEASoftPy to run BatSurveyTools."
    ) from exc

@dataclass
class ToolResult:
    task: str
    params: Dict[str, Any]
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    raw: Any = None

@dataclass
class _PointStats:
    image_id: str
    status: bool
    descr: str
    tstart: float
    tstop: float
    raw_exposure: float
    exposure: float
    ra_pnt: float
    dec_pnt: float
    pa_pnt: float
    ndets: int
    date_obs: str = ""
    date_end: str = ""
    numband: int = 0
    chi2: List[float] = field(default_factory=list)
    bkg_counts: List[float] = field(default_factory=list)


class HeasoftPyBackend:
    def __init__(self) -> None:

        self._hsp = hsp
        self._heatools = heatools

    @staticmethod
    def _normalize_names(task_name: str) -> List[str]:
        return [task_name, task_name.replace("-", "_"), task_name.replace("_", "-")]

    def _resolve_callable(self, task_name: str):
        for name in self._normalize_names(task_name):
            if hasattr(self._hsp, name):
                return getattr(self._hsp, name)
            if hasattr(self._heatools, name):
                return getattr(self._heatools, name)
        raise AttributeError(
            f"Task '{task_name}' not found in heasoftpy.swift/heatools"
        )

    def run(self, task_name: str, **params: Any) -> ToolResult:
        func = self._resolve_callable(task_name)
        out = func(**params)
        # return ToolResult(
        #     task=task_name,
        #     params=dict(params),
        #     returncode=getattr(out, "returncode", 0),
        #     stdout=getattr(out, "stdout", ""),
        #     stderr=getattr(out, "stderr", ""),
        #     raw=out,
        # )
        return getattr(out, "stdout", "")


def initalize_heasoft_task(
    routine: str,
    input_params: Dict[str, Any],
    soft_fail: bool = True,
) -> Tuple[hsp_core.HSPTask, Dict[str, Any]]:
    """
    Validate user parameters against HEASARC fhelp-allowed parameters.

    Args:
        routine: Name of the HEASoft routine (e.g. "baterebin").
        input_params: Dictionary of user-provided parameters to validate.
        soft_fail: If True, will return allowed parameters without raising an error on unknowns.

    Returns:
        A copy of input_params containing only allowed keys.
    Raises:
        KeyError if unknown parameters are present and soft_fail=False.
    """
    try:
        hsp_task=hsp_core.HSPTask(routine)
        hsp_task_params=hsp_task.default_params
    except hsp_core.HSPTaskException as e:
        raise AttributeError(
            f"The requested routine '{routine}' is not found. Please check the name."
        )

    allowed = [k for k in input_params.keys() if k in hsp_task_params.keys()]
    not_allowed = [k for k in input_params.keys() if k not in hsp_task_params.keys()]
    if not_allowed and not soft_fail:
        raise KeyError(
            f"Invalid parameter(s) for '{routine}': {not_allowed}. "
            f"Allowed parameters are: {sorted(hsp_task_params.keys())}"
        )

    #determine the parameters that go into this task's parameter dict from what the user provided
    for key in hsp_task_params.keys():
        if key in input_params.keys():
            hsp_task_params[key] = input_params[key]

    #check that the required parameters are passed in too ie no empty strings
    #include check that the parameters that we check dont have _opts in the name which indicate optional
    # eg batoccultgti_opts parameter in batsurvey-gti
    missing_req_params=[i for i in hsp_task_params.keys() if len(str(hsp_task_params[i]))==0 and "_opts" not in i]
    if missing_req_params and not soft_fail:
        raise KeyError(f"The user supplied parameters for the {routine} task is missing these required parameters: {','.join(missing_req_params)}")

    return hsp_task, hsp_task_params




class BatTools:
    def __init__(
        self,
        indir: str,
        outdir: Optional[str] = None,
        truncated: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.indir = Path(indir).resolve()
        self.outdir = (
            Path(outdir).resolve()
            if outdir
            else self.indir.parent / f"{self.indir.name}_surveyresult"
        )
        self.obs_id = self.indir.name

        # Store the truncated flag for later use (e.g. source extraction)
        self.truncated = truncated
        self.success = False

        # Make sure the output directory exists
        if params and "clobber" in params and params["clobber"]:
            if self.outdir.is_dir():
                shutil.rmtree(self.outdir)

        self.outdir.mkdir(parents=True, exist_ok=True)

        self.all_logs: Dict[str, List[ToolResult]] = {}
        self.all_params: Dict[str, Dict] = {}
        self.user_params = params.copy() if params else {}
        self.current_pointstatus = "unknown"
        self.current_pointreason = ""
        if not "patt_noise_dir" in self.user_params:
            self.patt_noise_dir = self.indir.parent / "pattern_noise_dir"
        else:
            self.patt_noise_dir = Path(self.user_params["patt_noise_dir"]).resolve()

        # Set the local pfile directory, so that heasoftpy utils run
        self._local_pfile_dir = self.outdir.joinpath(".local_pfile")

        # make the local pfile dir if it doesnt exist and set this value
        self._local_pfile_dir.mkdir(parents=True, exist_ok=True)
        try:
            hsp.local_pfiles(pfiles_dir=str(self._local_pfile_dir))
        except AttributeError:
            hsp_util.local_pfiles(par_dir=str(self._local_pfile_dir))

        self.backend = HeasoftPyBackend()
        self._get_survey_parameters()
        self._init_paths()
        self._get_pattern_noise_maps()
        self.success = self.run()

        # print("Survey parameters:")
        # for key, value in self.params.items():
        #     print(f"  {key}: {value}")

    def _init_paths(self):
        """Initialize the various paths to store the results"""
        if not self.indir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.indir}")
        if not (self.indir / "auxil").is_dir():
            raise FileNotFoundError(f"Missing auxil directory: {self.indir / 'auxil'}")
        if not (self.indir / "bat" / "survey").is_dir():
            raise FileNotFoundError(
                f"Missing survey directory: {self.indir / 'bat' / 'survey'}"
            )

        self.status_file = self.outdir / "global_status.txt"
        self.point_stats: List[_PointStats] = []
        self.outventory_file = self.outdir / "stats_point.dat"
        self.outventory_fits = self.outdir / "stats_point.fits"
        self.inventory_file = self.outdir / "stats_obs.dat"
        self.inventory_fits = self.outdir / "stats_obs.fits"

        for sub in ("scratch", "dph", "dpi", "gti", "lc", "auxil"):
            (self.outdir / sub).mkdir(parents=True, exist_ok=True)

        for stat_file in (
            self.outventory_file,
            self.outventory_fits,
            self.inventory_file,
            self.inventory_fits,
        ):
            stat_file.unlink(missing_ok=True)

        self.scratchdir = self.outdir / "scratch"

        self.attitude_glob = self.parseglob(self.params["attitude_pattern"])
        self.dph_glob = self.parseglob(self.params["dph_pattern"])
        self.sao_glob = self.parseglob(self.params["sao_pattern"])
        self.go_glob = self.parseglob(self.params["go_pattern"])
        self.det_flag_glob = self.parseglob(self.params["de_pattern"])

        self.att_file = glob.glob(self.attitude_glob)
        self.cal_file = glob.glob(self.go_glob)
        self.sao_file = glob.glob(self.sao_glob)
        self.det_flag_file = glob.glob(self.det_flag_glob)

        if not self.att_file:
            self.att_file = None
            print(f"WARNING: No attitude file found with pattern {self.attitude_glob}")
        else:
            self.att_file = self.att_file[0]
        if not self.sao_file:
            self.sao_file = None
            print(f"WARNING: No SAO file found with pattern {self.sao_glob}")
        else:
            self.sao_file = self.sao_file[0]
        if not self.cal_file:
            self.cal_file = None
            print(f"WARNING: No GO file found with pattern {self.go_glob}")
        else:
            self.cal_file = self.cal_file[0]
        if not self.det_flag_file:
            self.det_flag_file = None
            print(f"WARNING: No DE file found with pattern {self.det_flag_glob}")
        else:
            self.det_flag_file = self.det_flag_file[0]

        self.global_att_file = self.att_file
        # Next store the dph files
        self.dph_files = glob.glob(self.dph_glob)
        # Now save this
        self.survey_list = self.outdir / "scratch" / "survey.lis"
        np.savetxt(self.survey_list, self.dph_files, fmt="%s")

    def make_pointings_from_gtis(self, times: Sequence[float]):
        """Given a list of GTI times, create pointing files for each GTI segment.

        Args:
            times: List of GTI times (in seconds since the Swift epoch) to create pointings for.
        """
        # Then sve the individual pointings
        self.pointing_info = {}
        pids = Time(times).strftime("%Y%j%H%M")
        for ind, pid in enumerate(pids):
            pdir = self.outdir / f"point_{pid}"
            pid = pdir.name
            pdir.mkdir(parents=True, exist_ok=True)
            self.pointing_info[pid] = {"dir": pdir}
            self.pointing_info[pid]["pid"] = pid
            self.pointing_info[pid]["tstart"] = Time(times[ind])

    def _get_pattern_noise_maps(self):
        """Helper function to get the pattern noise map and mask for the observation
        ID based on the time of the observation and the files available in the pattern
        noise map directory. If the appropriate pattern noise map for the day of the
        observation is not available, then the function will search for the nearest
        pattern noise map based on the time of the observation. If there are no pattern
        noise maps available at all, then the function will return "NONE" for both the
        pattern noise map and mask.

        Returns:
            patt_map_name (str): The filename of the pattern noise map to be used for the
            observation ID. If no pattern noise maps are available, then this will be "NONE".
            patt_mask_name (str): The filename of the pattern noise mask to be used for the
            observation ID. If no pattern noise maps are available, then this will be "NONE".
        """
        input_files = self.dph_files
        if len(input_files) == 0:
            raise ValueError(
                f"The observation ID folder {self.indir} does not contain any survey data files."
            )
        else:
            input_file = input_files[0]
        with fits.open(str(input_file)) as hdu:
            tstart = hdu[1].header["DATE-OBS"]

        time = Time(tstart, format="isot", scale="tai").datetime64

        # get the day of the year, need to add 1 since day 1 is the first day of the year
        obs_doy = str(
            np.timedelta64((time - np.datetime64(time.astype("M8[Y]"), "D")), "D")
            + np.timedelta64(1, "D")
        ).split(" ")[0]
        obs_year = str(time.astype("M8[Y]"))

        # see if the directory exists
        patt_noise_dir = self.patt_noise_dir
        if patt_noise_dir.is_dir():
            # if so then find the files with the year/doy combo that we need for this obs_id
            if len(sorted(patt_noise_dir.glob(f"*_{obs_year}{obs_doy}*"))) > 0:
                # these should be the files names
                patt_map_name = patt_noise_dir.joinpath(
                    f"pattern_noise_survey8a_{obs_year}{obs_doy}.dpi"
                )
                patt_mask_name = patt_noise_dir.joinpath(
                    f"pattern_noise_survey8a_{obs_year}{obs_doy}_inbands.detmask"
                )

                # make sure that the files exist
                if patt_map_name.is_file() and patt_mask_name.is_file():
                    patt_map_name = str(patt_map_name)
                    patt_mask_name = str(patt_mask_name)
                else:
                    # if the files dont exist then set these values to None
                    patt_map_name = "NONE"
                    patt_mask_name = "NONE"
            else:
                # if that file doesnt exist then search for file with nearest year/doy stamp
                # get allthe filenames and the years/days associated with them
                all_patt_map = sorted(patt_noise_dir.glob("*.dpi"))
                all_patt_mask = sorted(patt_noise_dir.glob("*_inbands.detmask"))

                years = [i.stem.split("_")[-1][:4] for i in all_patt_map]
                days = [i.stem.split("_")[-1][4:] for i in all_patt_map]

                # turn them into numpy dates
                patt_dates = np.array(
                    [
                        np.datetime64(datetime(int(i), 1, 1) + timedelta(int(j) - 1))
                        for i, j in zip(years, days)
                    ]
                )

                # find the date closest to the time of the observation start time
                idx = np.abs(time - patt_dates).argmin()

                # save the name
                patt_map_name = str(all_patt_map[idx])
                patt_mask_name = str(all_patt_mask[idx])

        else:
            # if the directory doesnt exist then set these values to None
            patt_map_name = "NONE"
            patt_mask_name = "NONE"

        if not "global_pattern_map" in self.user_params:
            self.params["global_pattern_map"] = patt_map_name
        if not "global_pattern_mask" in self.user_params:
            self.params["global_pattern_mask"] = patt_mask_name

    def _get_survey_parameters(self):
        """Helper function to get the survey parameters that will be passed to batsurvey
        based on the input dictionary provided by the user and the default values for
        these parameters. If the user does not provide a value for a parameter, then
        the default value will be used. If the user provides a value of None for a
        parameter, then the default value will be used. If the user provides a value
        for a parameter, then that value will be used.

        Raises:
            ValueError: _description_
        """
        self.obs_id = self.indir.name
        input_dict = self.user_params

        batsurvey = hsp_core.HSPTask("batsurvey")


        if input_dict is None:
            input_dict_copy = batsurvey.default_params.copy()

            input_dict_copy["indir"] = str(self.indir)
            input_dict_copy["outdir"] = str(self.outdir)

            input_dict_copy["incatalog"] = str(
                Path(__file__).parent.joinpath("data/survey6b_2.cat")
            )
            input_dict_copy["detthresh"] = "10000"
            input_dict_copy["detthresh2"] = "10000"

            input_dict_copy["cleansnr"] = 6
            input_dict_copy["cleanexpr"] = "ALWAYS_CLEAN==T"
        else:
            # need to create copy of input dict so we dont overwrite it
            input_dict_copy = batsurvey.default_params.copy()

            # And then overwrite with the user-provided values, ensuring that
            # user-provided values take precedence over defaults
            for key, value in input_dict.items():
                if value is not None:
                    input_dict_copy[key] = value
            # see if the user wanted the indir and outdir to be the defaults presented above, even though they
            # specify other preferences to the call to batsurvey
            if (
                "indir" not in input_dict
                or str(input_dict.get("indir", "NONE")).upper() != "NONE"
            ):
                input_dict_copy["indir"] = str(self.indir)

            if (
                "outdir" not in input_dict
                or str(input_dict.get("outdir", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["outdir"] = str(self.outdir)

            # if detthresh/detthresh2 isnt defined need to set default detthresh to prevent gti identification
            # errors
            if (
                "detthresh" not in input_dict
                or str(input_dict.get("detthresh", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["detthresh"] = "10000"

            if (
                "detthresh2" not in input_dict
                or str(input_dict.get("detthresh2", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["detthresh2"] = "10000"

            if "incatalog" not in input_dict:
                # If the user choses to provide no input catalog,
                # then respect the choice and do not provide any catalog to batsurvey
                input_dict_copy["incatalog"] = str(
                    Path(__file__).parent.joinpath("data/survey6b_2.cat")
                )
                if not Path(input_dict_copy["incatalog"]).is_file():
                    input_dict_copy["incatalog"] = "NONE"

            if (
                "cleansnr" not in input_dict
                or str(input_dict.get("cleansnr", "NONE")).upper() == "NONE"
            ):
                input_dict_copy["cleansnr"] = 6

            if "cleanexpr" not in input_dict:
                # If the user choses to provide no clean expression,
                # then respect the choice and do not provide any clean expression to batsurvey
                if input_dict_copy["incatalog"] != "NONE":
                    input_dict_copy["cleanexpr"] = "ALWAYS_CLEAN==T"
                else:
                    input_dict_copy["cleanexpr"] = "NONE"
            else:
                if input_dict_copy["incatalog"] == "NONE":
                    input_dict_copy["cleanexpr"] = "NONE"

        self.params = input_dict_copy
        self.numbins = len(str(self.params["energybins"]).split(","))
        # Dump this into the log file for later reference

    def parseglob(self, expr: str) -> str:
        expr = expr.strip()
        expr = expr.replace("@INDIR", f"@{self.indir}")
        expr = expr.replace("INDIR", str(self.indir))
        return expr

    def timecheckpoint(self) -> None:
        now = datetime.now(timezone.utc)
        print(f"batsurvey: time-check - {now.strftime('%Y-%m-%dT%H:%M:%S')}")

    def pntstat(
        self,
        pointname: str,
        code: str,
        desc: str,
        ndets: int = 0,
        exposure: float = 0.0,
    ) -> None:
        status = "SUCCESS" if code == "ok" else "FAIL"
        self.current_pointstatus = code
        self.current_pointreason = desc
        line = (
            f'status="{status}";code="{code}";reason="{desc}";'
            f'obsid="{self.indir.name}";point="{pointname}";'
            f'ndets="{ndets}";exposure="{exposure}"'
        )
        point_status = self.outdir / pointname / f"{pointname}_status.txt"
        point_status.write_text(line + "\n")
        if code != "ok":
            print(f"WARNING: {desc} ({code})")

    def obsstat(
        self,
        code: str,
        desc: str,
        ndets: int = 0,
        exposure: float = 0.0,
    ) -> None:
        status = "SUCCESS" if code == "ok" else "FAIL"
        self.current_pointstatus = code
        self.current_pointreason = desc
        line = (
            f'status="{status}";code="{code}";reason="{desc}";'
            f'obsid="{self.indir.name}";'
            f'ndets="{ndets}";exposure="{exposure}"'
        )
        obs_status = self.outdir / f"{self.obs_id}_status.txt"
        obs_status.write_text(line + "\n")
        if code != "ok":
            print(f"WARNING: {desc} ({code})")

    def _met_to_datestr(self, met: float) -> str:
        tref = datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()
        dt = datetime.fromtimestamp(tref + float(met), tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _build_fail_pointstats(
        self,
        pid: str,
        med_ra: float,
        med_dec: float,
        med_roll: float,
        ndets: int = 0,
        raw_exposure: float = 0.0,
        exposure: float = 0.0,
        chi2: Optional[List[float]] = None,
        bkg_counts: Optional[List[float]] = None,
    ) -> _PointStats:
        tstart = float(self.pointing_info[pid]["tstart_met"])
        tstop = float(self.pointing_info[pid]["tstop_met"])
        return _PointStats(
            image_id=pid,
            status=False,
            descr=self.current_pointstatus,
            tstart=tstart,
            tstop=tstop,
            raw_exposure=raw_exposure,
            exposure=exposure,
            ra_pnt=float(med_ra),
            dec_pnt=float(med_dec),
            pa_pnt=float(med_roll),
            ndets=int(ndets),
            date_obs=self._met_to_datestr(tstart),
            date_end=self._met_to_datestr(tstop),
            numband=self.numbins,
            chi2=chi2 if chi2 is not None else [0.0] * self.numbins,
            bkg_counts=bkg_counts if bkg_counts is not None else [0.0] * self.numbins,
        )

    def _ensure_stats_point_fits(self) -> None:
        if self.outventory_fits.exists():
            return

        cols = [
            fits.Column(name="OBS_ID", format="20A"),
            fits.Column(name="IMAGE_ID", format="20A"),
            fits.Column(name="BSURVER", format="10A"),
            fits.Column(name="BSURSEQ", format="20A"),
            fits.Column(name="DATE_OBS", format="22A"),
            fits.Column(name="DATE_END", format="22A"),
            fits.Column(name="TSTART", format="1D", unit="s"),
            fits.Column(name="TSTOP", format="1D", unit="s"),
            fits.Column(name="RAW_EXPOSURE", format="1D", unit="s"),
            fits.Column(name="EXPOSURE", format="1D", unit="s"),
            fits.Column(name="IMAGE_STATUS", format="L"),
            fits.Column(name="IMAGE_DESCR", format="20A"),
            fits.Column(name="RA_PNT", format="1D", unit="deg"),
            fits.Column(name="DEC_PNT", format="1D", unit="deg"),
            fits.Column(name="PA_PNT", format="1D", unit="deg"),
            fits.Column(name="NBATDETS", format="J"),
            fits.Column(name="NUMBAND", format="I"),
            fits.Column(name="CHI2", format=f"{self.numbins}D"),
            fits.Column(name="BKG_COUNTS", format=f"{self.numbins}D", unit="count"),
        ]
        hdu = fits.BinTableHDU.from_columns(cols, nrows=0, name="STATS_POINT")
        hdu.header["BSURVER"] = (batsurvey_version, "BAT survey processing version")
        bsurseq = str(self.params.get("bsurseq", "NONE"))
        if bsurseq.upper() != "NONE":
            hdu.header["BSURSEQ"] = (bsurseq, "BAT survey processing sequence")

        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            self.outventory_fits, overwrite=True
        )

    def _ensure_stats_obs_fits(self) -> None:
        if self.inventory_fits.exists():
            return

        cols = [
            fits.Column(name="OBS_ID", format="20A"),
            fits.Column(name="BSURVER", format="10A"),
            fits.Column(name="BSURSEQ", format="20A"),
            fits.Column(name="DATE_OBS", format="22A"),
            fits.Column(name="DATE_END", format="22A"),
            fits.Column(name="TSTART", format="1D", unit="s"),
            fits.Column(name="TSTOP", format="1D", unit="s"),
            fits.Column(name="RAW_EXPOSURE", format="1D", unit="s"),
            fits.Column(name="EXPOSURE", format="1D", unit="s"),
            fits.Column(name="N_RAW_IMAGES", format="I"),
            fits.Column(name="N_IMAGES", format="I"),
        ]
        hdu = fits.BinTableHDU.from_columns(cols, nrows=0, name="STATS_OBS")
        hdu.header["BSURVER"] = (batsurvey_version, "BAT survey processing version")
        bsurseq = str(self.params.get("bsurseq", "NONE"))
        if bsurseq.upper() != "NONE":
            hdu.header["BSURSEQ"] = (bsurseq, "BAT survey processing sequence")

        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            self.inventory_fits, overwrite=True
        )

    def _append_row_to_bintable(
        self, filename: Path, extname: str, row: Dict[str, Any]
    ) -> None:
        with fits.open(filename, mode="update") as hdul:
            hdu = hdul[extname]
            data = hdu.data
            n_old = len(data) if data is not None else 0
            dtype = data.dtype if data is not None else hdu.columns.dtype
            new_rows = np.zeros(n_old + 1, dtype=dtype)
            if data is not None and n_old > 0:
                new_rows[:n_old] = data
            for name, value in row.items():
                if name in new_rows.dtype.names:
                    new_rows[n_old][name] = value
            hdu.data = new_rows
            hdul.flush()

    def _read_per_band_scalar_fits(self, filename: Path) -> List[float]:
        """Read one scalar per energy band from an ftimgcalc output FITS.

        Supports:
        - one image extension per band (HDU 1..N)
        - one vectorized extension containing multiple values
        """
        values = [0.0] * self.numbins
        if not filename.exists():
            return values

        try:
            with fits.open(filename) as hdul:
                for iband in range(self.numbins):
                    hdu_idx = iband + 1

                    if (
                        hdu_idx < len(hdul)
                        and getattr(hdul[hdu_idx], "data", None) is not None
                    ):
                        arr = np.asarray(hdul[hdu_idx].data)
                        if arr.size > 0:
                            values[iband] = float(arr.ravel()[0])
                            continue

                    if len(hdul) > 1 and getattr(hdul[1], "data", None) is not None:
                        arr = np.asarray(hdul[1].data)
                        flat = arr.ravel()
                        if flat.size > iband:
                            values[iband] = float(flat[iband])
        except Exception as exc:
            print(f"WARNING: Could not parse {filename.name}: {exc}")

        return values

    def _record_snapshot_stats(self, s: _PointStats) -> None:
        with self.outventory_file.open("a", encoding="utf-8") as f:
            chi_str = " ".join(str(float(x)) for x in s.chi2)
            f.write(
                f"B {batsurvey_version} {self.indir.name} {s.image_id} {int(s.status)} {s.descr} "
                f"{s.tstart} {s.tstop} {s.ra_pnt} {s.dec_pnt} {s.pa_pnt} "
                f"{s.raw_exposure} {s.exposure} {s.ndets} {self.numbins} {chi_str}\n"
            )

        self._ensure_stats_point_fits()
        bsurseq = str(self.params.get("bsurseq", "NONE"))
        self._append_row_to_bintable(
            self.outventory_fits,
            "STATS_POINT",
            {
                "OBS_ID": self.obs_id,
                "IMAGE_ID": s.image_id,
                "BSURVER": batsurvey_version,
                "BSURSEQ": (bsurseq if bsurseq.upper() != "NONE" else "NULL"),
                "DATE_OBS": s.date_obs,
                "DATE_END": s.date_end,
                "TSTART": float(s.tstart),
                "TSTOP": float(s.tstop),
                "RAW_EXPOSURE": float(s.raw_exposure),
                "EXPOSURE": float(s.exposure),
                "IMAGE_STATUS": bool(s.status),
                "IMAGE_DESCR": s.descr,
                "RA_PNT": float(s.ra_pnt),
                "DEC_PNT": float(s.dec_pnt),
                "PA_PNT": float(s.pa_pnt),
                "NBATDETS": int(s.ndets),
                "NUMBAND": int(self.numbins),
                "CHI2": np.array(s.chi2, dtype=float),
                "BKG_COUNTS": np.array(s.bkg_counts, dtype=float),
            },
        )

    def _write_obs_stats(self) -> None:
        starts = [float(s.tstart) for s in self.point_stats]
        stops = [float(s.tstop) for s in self.point_stats]
        totexpo = float(sum(float(s.raw_exposure) for s in self.point_stats))
        goodexpo = float(sum(float(s.exposure) for s in self.point_stats if s.status))
        ngood = sum(1 for s in self.point_stats if s.status)

        self.inventory_file.write_text(
            f"A {self.indir.name} {self.outdir.name} {totexpo} {goodexpo} {len(starts)} {ngood} 1\n",
            encoding="utf-8",
        )

        self._ensure_stats_obs_fits()
        bsurseq = str(self.params.get("bsurseq", "NONE"))
        date_obs = min(
            (s.date_obs for s in self.point_stats if s.exposure > 0), default=""
        )
        date_end = max(
            (s.date_end for s in self.point_stats if s.exposure > 0), default=""
        )
        self._append_row_to_bintable(
            self.inventory_fits,
            "STATS_OBS",
            {
                "OBS_ID": self.obs_id,
                "BSURVER": batsurvey_version,
                "BSURSEQ": (bsurseq if bsurseq.upper() != "NONE" else "NULL"),
                "DATE_OBS": date_obs,
                "DATE_END": date_end,
                "TSTART": min(starts) if starts else 0.0,
                "TSTOP": max(stops) if stops else 0.0,
                "RAW_EXPOSURE": totexpo,
                "EXPOSURE": goodexpo,
                "N_RAW_IMAGES": len(starts),
                "N_IMAGES": ngood,
            },
        )

    def baterebin(self, params: Dict[str, Any]) -> List[str]:
        """Run baterebin on the survey list to create erebinned DPH files and masks.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to pass to baterebin.
            Only valid parameters will be used.
        """

        params = self.params
        # Only pass those that are valid to this
        _, baterebin_params = initalize_heasoft_task("baterebin", params)

        erebin_log = []

        mandatory_params = {}

        default_params = {}
        default_params["residfile"] = "CALDB"
        default_params["pulserfile"] = "CALDB"
        default_params["fltpulserfile"] = "CALDB"
        default_params["history"] = "YES"
        default_params["ebins"] = (
            "0-14,14-20,20-24,24-35,35-50,50-75,75-100,100-150,150-195"
        )

        #update the heasoftpy task params
        baterebin_params.update(default_params)

        for dph_file in self.dph_files:
            dph_file = Path(dph_file)
            mandatory_params["infile"] = str(dph_file)
            mandatory_params["outfile"] = str(
                self.outdir
                / "dph"
                / f"{dph_file.name.replace('.dph.gz', '_erebin.dph')}"
            )
            mandatory_params["outmap"] = str(
                self.outdir / "dph" / f"{dph_file.stem.replace('.dph', '.mask')}"
            )
            mandatory_params["calfile"] = self.cal_file

            # Replace mandatory params into the user-provided params, ensuring that mandatory params take precedence
            baterebin_params.update(mandatory_params)

            #check the parameters and ensure they are what we need to run the task
            baterebin_task, baterebin_params = initalize_heasoft_task("baterebin", baterebin_params, soft_fail=False)

            # print("Running baterebin with parameters:", baterebin_params)
            #erebin_log.append(self.backend.run("baterebin", **baterebin_params))
            erebin_log.append(baterebin_task(**baterebin_params))
            # print(erebin_log[-1].stdout)

        erebinned = list((self.outdir / "dph").glob("*_erebin.dph"))
        erebinned = self.select_good_files_for_gtis(files=erebinned)
        dph_masks = [f.replace("_erebin.dph", ".mask") for f in erebinned]
        erebinned = [f"{erebin}[DATA_FLAGS == 0]" for erebin in erebinned]
        if not erebinned:
            raise RuntimeError("No erebinned files were created")

        if not dph_masks:
            raise RuntimeError("No erebin masks were created")

        # Now store the list of erebinned files for later use
        self.esurvey_lis = self.scratchdir / "esurvey.lis"
        np.savetxt(self.esurvey_lis, erebinned, fmt="%s")

        self.dph_mask_lis = self.scratchdir / "dph_masks.lis"
        np.savetxt(self.dph_mask_lis, dph_masks, fmt="%s")

        self.all_params["baterebin"] = baterebin_params
        self.all_logs["baterebin"] = erebin_log

    def select_good_files_for_gtis(self, files: List[str]) -> List[str]:
        """When there are multiple DPHs with combinations of DATA_FLAG==0 and DATA_FLAG==1,
        gti generation crashed even if some of them have 1 instead of 0. So we will filter
        out bad files and only keep good files.
        """
        good_files = []
        for f in files:
            data = fits.getdata(f, 1)
            flags = data["DATA_FLAGS"]
            if np.any(flags == 0):
                good_files.append(str(f.resolve()))
        return good_files

    def batsurvey_gti(self, params: Dict[str, Any]):

        params = self.params
        _, batsurvey_gti_params = initalize_heasoft_task("batsurvey-gti", params)

        mandatory_params = {}
        mandatory_params["indir"] = str(self.indir)
        mandatory_params["dphfiles"] = f"@{self.esurvey_lis}"
        mandatory_params["outdir"] = str(self.outdir / "gti")

        default_params = {}

        default_params["sepdph"] = "NO"
        default_params["filters"] = "all"
        default_params["elimits"] = "14-195"
        default_params["rateminthresh"] = 3000
        default_params["ratemaxthresh"] = 12000
        default_params["detthresh"] = 6000
        default_params["stlossfcnthresh"] = 1.0e-9
        default_params["filtexpr"] = "NONE"
        default_params["saofiltexpr"] = "ELV > 30.0"
        default_params["dphfiltexpr"] = "DATA_FLAGS == 0"
        default_params["gtifile"] = "NONE"

        # Replace mandatory params into the user-provided params, ensuring that mandatory params take precedence
        batsurvey_gti_params.update(default_params)
        batsurvey_gti_params.update(mandatory_params)

        # check the parameters and ensure they are what we need to run the task
        batsurvey_gti_task, batsurvey_gti_params = initalize_heasoft_task("batsurvey-gti", batsurvey_gti_params, soft_fail=False)


        # print("Running batsurvey-gti with parameters:", batsurvey_gti_params)
        batsurvey_gti_log = batsurvey_gti_task(**batsurvey_gti_params) #self.backend.run("batsurvey-gti", **batsurvey_gti_params)
        self.all_params["batsurvey-gti"] = batsurvey_gti_params
        self.all_logs["batsurvey-gti"] = batsurvey_gti_log
        # print(batsurvey_gti_log.stdout)

        master = self.outdir / "gti" / "master.gti"
        if not master.exists():
            raise RuntimeError("Could not create master.gti")
        else:
            self.mastergti = master.resolve()

        # Now read the GTI file and get the good times
        data = Table.read(self.mastergti, hdu=1)
        tstart = np.asarray(data["START"].data, dtype=float)
        tstop = np.asarray(data["STOP"].data, dtype=float)
        # This is in MET, so we need to convert to UTC for the pointing checks later
        utcs = convert_met_to_utc(tstart)

        # Now make pointings from the GTI times
        self.make_pointings_from_gtis(utcs)
        for i, pid in enumerate(self.pointing_info.keys()):
            self.pointing_info[pid]["igtirow"] = i
            self.pointing_info[pid]["tstart_met"] = float(tstart[i])
            self.pointing_info[pid]["tstop_met"] = float(tstop[i])

    def ftcopy_gti_row(self):
        for point_name in self.pointing_info.keys():
            point_dir = self.pointing_info[point_name]["dir"]
            pnt0 = point_dir / f"{point_name}_pnt0.gti"
            pnt0 = point_dir / f"{point_name}_pnt0.gti"
            master = self.outdir / "gti" / "master.gti"
            igtirow = int(self.pointing_info[point_name]["igtirow"])
            timeslop = float(self.params.get("timeslop", 0.0))
            res = heatools.ftcopy(
                infile=f"{master}[1][#ROW == ({igtirow}+1)][col START=START-{timeslop}; STOP=STOP+{timeslop}]",
                outfile=pnt0,
                clobber=True,
            )

            if "ftcopy" not in self.all_logs:
                self.all_logs["ftcopy"] = [res]
            else:
                self.all_logs["ftcopy"].append(res)

            self.pointing_info[point_name]["gti_files"] = [pnt0]

    def batbinevt(
        self, pid, infile: str, outfile: str, gtifile: str, params: Dict[str, Any]
    ):
        """Run batbinevt on the discovered DPH file with internal path
        defaults and optional parameter overrides.

        Args:
            pid: Pointing ID to process (e.g. "20233341529")
            infile: Input DPH file to process (e.g. "point_20233341529_erebin.dph")
            outfile: Output DPI file to create (e.g. "point_20233341529_erebin.dpi")
            gtifile: GTI file to use for filtering (e.g. "point_20233341529_pnt0.gti")
            params: Dictionary of parameters to override defaults for this step. Only valid parameters will be used.

        """
        params = self.params
        # Only pass those that are valid to this
        _, batbinevt_params = initalize_heasoft_task("batbinevt", params)

        mandatory_params = {}
        mandatory_params["infile"] = infile
        mandatory_params["outfile"] = outfile
        mandatory_params["gtifile"] = gtifile

        # And these are defaults, so if the user didnt provide them then we set them,
        # but if they did then we respect the user choice
        default_params = {
            "outtype": "DPI",
            "timedel": 0,
            "timebinalg": "u",
            "weighted": "NO",
            "outunits": "RATE",
            "min_dph_frac_overlap": 0.75,
            "max_dph_time_nonoverlap": 40,
        }
        #TODO: figure out the best logic for this since initalize_heasoft_task populates this with the default values from
        # heasoft and the keys will always be there
        # for key, value in default_params.items():
        #     if key not in batbinevt_params:
        #         batbinevt_params[key] = value

        batbinevt_params.update(default_params)

        # And then replace these mandatory params into the user-provided params,
        # ensuring that mandatory params take precedence
        batbinevt_params.update(mandatory_params)



        batbinevt_task, batbinevt_params = initalize_heasoft_task("batbinevt", batbinevt_params, soft_fail=False)

        # print("Running batbinevt with parameters:", batbinevt_params)
        batbinevtlog = batbinevt_task(**batbinevt_params) #self.backend.run("batbinevt", **batbinevt_params)
        # print(batbinevtlog.stdout)
        if "batbinevt" not in self.all_params:
            self.all_params["batbinevt"] = [batbinevt_params]
        else:
            self.all_params["batbinevt"].append(batbinevt_params)
        if "batbinevt" not in self.all_logs:
            self.all_logs["batbinevt"] = [batbinevtlog]
        else:
            self.all_logs["batbinevt"].append(batbinevtlog)

    def batsurvey_aspect(
        self,
        pid: str,
        gti_file: str,
        outgti_file: str,
        finalgti_file: str,
        params: Dict[str, Any],
    ):
        """Run batsurvey-aspect on the GTI file with internal path defaults
        and optional parameter overrides.

        Args:
            pid: Pointing ID to process (e.g. "20233341529")
            gti_file: GTI file to process (e.g. "point_20233341529_pnt0.gti")
            outgti_file: Output GTI file to create (e.g. "point_20233341529_pnt1.gti")
            finalgti_file: Final GTI file to create (e.g. "point_20233341529_pnt.gti")
            params: Dictionary of parameters to override defaults for this step. Only valid parameters will be used.

        """
        params = self.params
        _, batsurvey_aspect_params = initalize_heasoft_task("batsurvey-aspect", params)

        mandatory_params = {}
        mandatory_params["gtifile"] = f"{gti_file}[STDGTI]"
        mandatory_params["outgtifile"] = outgti_file


        default_params = {}
        default_params["attfile"] = self.att_file
        default_params["point_toler"] = self.params["point_toler"]
        default_params["roll_toler"] = self.params["roll_toler"]
        default_params["alignfile"] = self.params["alignfile"]
        default_params["outattfile"] = str(
            self.pointing_info[pid]["dir"] / f"{pid}.att"
        )
        self.pointing_info[pid]["attfile"] = default_params["outattfile"]


        batsurvey_aspect_params.update(default_params)

        batsurvey_aspect_params.update(mandatory_params)

        batsurvey_aspect_task, batsurvey_aspect_params = initalize_heasoft_task("batsurvey-aspect", batsurvey_aspect_params, soft_fail=False)


        # print("Running batsurvey-aspect with parameters:", batsurvey_aspect_params)
        #batsurvey_aspect_log = self.backend.run(
        #    "batsurvey-aspect", **batsurvey_aspect_params
        #)
        batsurvey_aspect_log = batsurvey_aspect_task(**batsurvey_aspect_params)

        if "batsurvey-aspect" not in self.all_params:
            self.all_params["batsurvey-aspect"] = [batsurvey_aspect_params]
        else:
            self.all_params["batsurvey-aspect"].append(batsurvey_aspect_params)
        if "batsurvey-aspect" not in self.all_logs:
            self.all_logs["batsurvey-aspect"] = [batsurvey_aspect_log]
        else:
            self.all_logs["batsurvey-aspect"].append(batsurvey_aspect_log)

        expotot = float(batsurvey_aspect_log.params.get("expotot"))

        expobad = float(batsurvey_aspect_log.params.get("expobad"))
        med_ra = float(batsurvey_aspect_log.params.get("med_ra"))
        med_dec = float(batsurvey_aspect_log.params.get("med_dec"))
        med_roll = float(batsurvey_aspect_log.params.get("med_roll"))

        pnt = finalgti_file
        use_pnt = outgti_file
        if expotot > 0:
            if (expobad / expotot) <= float(
                self.params["pointerr_frac_time"]
            ) and expobad <= float(self.params["pointerr_abs_time"]):
                use_pnt = outgti_file
            else:
                use_pnt = gti_file
        shutil.copy2(use_pnt, pnt)
        self.pointing_info[pid]["gti"] = pnt
        self.pointing_info[pid]["gti_files"].append(outgti_file)
        return expotot, expobad, med_ra, med_dec, med_roll

    def fimgstat(self, infile: str, params: Optional[Dict[str, Any]] = {}):
        defaults = {
            "threshlo": "INDEF",
            "threshup": "INDEF",
            "min": -999,
            "max": 999,
            "clobber": "YES",
        }

        #get the merged dictionary of params, where the user supplied parameters override these defaults
        merged_params=defaults | params

        _, fimgstat_params = initalize_heasoft_task("fimgstat", merged_params)

        fimgstat_params["infile"]=infile

        #do this again to get the task
        fimgstat_task, fimgstat_params = initalize_heasoft_task("fimgstat", fimgstat_params, soft_fail=False)

        hsp_return=fimgstat_task(**fimgstat_params)

        dmin = float(hsp_return.params.get("min") or 0)

        dmax = float(hsp_return.params.get("max") or 0)

        dsum = float(hsp_return.params.get("sum") or 0)
        return dmin, dmax, dsum

    def batsurvey_detmask(
        self, pid: str, infile: str, maskfile: str, params: Dict[str, Any]
    ):
        """Run batsurvey-detmask on the given input file with the provided parameters and output the detector mask.

        Args:
            pid (str): Pointing ID to process (e.g. "20233341529")
            infile (str): Input DPI file to process (e.g. "point_20233341529_1.dpi")
            maskfile (str): Detector mask file to create (e.g. "point_20233341529.detmask")
            params (Dict[str, Any]): _description_
        """
        point_dir = self.pointing_info[pid]["dir"]
        dpi_file = str(infile)
        dmask = str(maskfile)

        # Write it back
        self.pointing_info[pid]["detmask"] = dmask

        mandatory_params = {}
        mandatory_params["infile"] = dpi_file
        mandatory_params["outfile"] = dmask

        default_params = {}
        if self.det_flag_file:
            default_params["detflagfile"] = self.det_flag_file
            default_params["repoquery"] = "NO"
        else:
            default_params["detflagfile"] = "NONE"
            default_params["repoquery"] = "YES"
        default_params["patternmask"] = self.params["global_pattern_mask"]
        default_params["detmask"] = f"@{self.dph_mask_lis}"
        default_params["outenamask"] = f"{point_dir / pid}.enamask"
        default_params["outcaldbmask"] = f"{point_dir / pid}.caldbmask"
        default_params["clobber"] = "YES"
        default_params["chatter"] = 2
        default_params["cleanup"] = "YES"

        _, batsurvey_detmask_params = initalize_heasoft_task("batsurvey-detmask", params)
        # Replace mandatory params into the user-provided params, ensuring that mandatory params take precedence
        batsurvey_detmask_params.update(mandatory_params)

        # And these are defaults, so if the user didnt provide them then we set them,
        # but if they did then we respect the user choice
        batsurvey_detmask_params.update(default_params)

        batsurvey_detmask_task, batsurvey_detmask_params = initalize_heasoft_task("batsurvey-detmask", batsurvey_detmask_params, soft_fail=False)


        batsurvey_detmask_log = batsurvey_detmask_task(**batsurvey_detmask_params)
        # print(batsurvey_detmask_log.stdout)
        if "batsurvey-detmask" not in self.all_params:
            self.all_params["batsurvey-detmask"] = [batsurvey_detmask_params]
        else:
            self.all_params["batsurvey-detmask"].append(batsurvey_detmask_params)
        if "batsurvey-detmask" not in self.all_logs:
            self.all_logs["batsurvey-detmask"] = [batsurvey_detmask_log]
        else:
            self.all_logs["batsurvey-detmask"].append(batsurvey_detmask_log)

    def batclean(self, infile, outfile, detmask, params: Dict[str, Any]):
        """Run batclean on the given input file with the provided parameters and output the cleaned file.

        Args:
            infile (str): _input DPI file to process (e.g. "point_20233341529_1.dpi")
            outfile (str): _output cleaned DPI file (e.g. "point_20233341529_1.bkgdpi")
            detmask (str): _detector mask file to use (e.g. "point_20233341529.detmask")
            params (Dict[str, Any]): Dictionary of parameters to pass to batclean, which may include overrides for defaults.
        """

        mandatory_params = {}
        mandatory_params["infile"] = infile
        mandatory_params["outfile"] = outfile
        mandatory_params["detmask"] = detmask

        defaults_params = {}
        defaults_params["outversion"] = "fit"
        defaults_params["srcclean"] = "NO"
        defaults_params["aperture"] = "NONE"
        defaults_params["balance"] = "ShortEdges,LongEdges,InOut"
        defaults_params["maskfit"] = "YES"
        defaults_params["clobber"] = "YES"

        _, batclean_params = initalize_heasoft_task("batclean", params)

        # Overwrite mandatory params
        batclean_params.update(mandatory_params)
        batclean_params.update(defaults_params)

        batclean_task, batclean_params = initalize_heasoft_task("batclean", batclean_params, soft_fail=False)

        # print("Running batclean with parameters:", batclean_params)
        batclean_log = batclean_task(**batclean_params)
        # print(batclean_log.stdout)
        if "batclean" not in self.all_params:
            self.all_params["batclean"] = [batclean_params]
        else:
            self.all_params["batclean"].append(batclean_params)
        if "batclean" not in self.all_logs:
            self.all_logs["batclean"] = [batclean_log]
        else:
            self.all_logs["batclean"].append(batclean_log)

    def batfftimage(
        self,
        pid: str,
        infile: str,
        outfile: str,
        detmask: str,
        bkgfile: str,
        bkgvarmap: str,
        params: Dict[str, Any],
    ):
        """Run batfftimage on the given input file with the provided parameters and output the FFT image.

        Args:
            pid (str): Pointing ID to process (e.g. "20233341529")
            infile (str): _input DPH file to process (e.g. "point_20233341529_1.dpi")
            outfile (str): _output FFT image file (e.g. "point_20233341529_1.img")
            detmask (str): _detector mask file to use (e.g. "point_20233341529.detmask")
            bkgfile (str): _background DPH file to use for background subtraction (e.g. "point_20233341529_1_bkg.dpi")
            bkgvarmap (str): _background variance map file to use for weighting (e.g. "point_20233341529_1.pointvar")
            params (Dict[str, Any]): Dictionary of parameters to pass to batfftimage, which may include overrides for defaults.
        """

        mandatory_params = {}
        mandatory_params["infile"] = infile
        mandatory_params["outfile"] = outfile
        mandatory_params["detmask"] = detmask
        mandatory_params["bkgfile"] = bkgfile
        mandatory_params["bkgvarmap"] = bkgvarmap

        defaults_params = {}
        defaults_params["attitude"] = self.pointing_info[pid]["attfile"]
        defaults_params["clobber"] = "YES"
        defaults_params["aperture"] = "CALDB:FLUX"
        defaults_params["teldef"] = "CALDB"
        defaults_params["pcodemap"] = "APPEND_LAST"
        defaults_params["keepbits"] = "7"
        defaults_params["bkgvartype"] = "STDDEV"

        _, batfftimage_params = initalize_heasoft_task("batfftimage", params)

        # Overwrite mandatory params
        batfftimage_params.update(mandatory_params)
        batfftimage_params.update(defaults_params)


        batfftimage_task, batfftimage_params = initalize_heasoft_task("batfftimage", batfftimage_params, soft_fail=False)

        batfftimage_log = batfftimage_task(**batfftimage_params)
        # print(batfftimage_log.stdout)
        if "batfftimage" not in self.all_params:
            self.all_params["batfftimage"] = [batfftimage_params]
        else:
            self.all_params["batfftimage"].append(batfftimage_params)
        if "batfftimage" not in self.all_logs:
            self.all_logs["batfftimage"] = [batfftimage_log]
        else:
            self.all_logs["batfftimage"].append(batfftimage_log)

    def batoccultmap(self, infile: str, outfile: str, params: Dict[str, Any]):
        """Run batoccultmap on the given input file with the provided parameters
        and output the occultation map.

        Args:
            infile (str): Image file to process (e.g. "point_20233341529_1.img")
            outfile (str): Occultation map file to create (e.g. "point_20233341529.occmap")
            params (Dict[str, Any]): Dictionary of parameters to pass to batoccultmap,
            which may include overrides for defaults.
        """
        mandatory_params = {}
        mandatory_params["infile"] = infile
        mandatory_params["outfile"] = outfile

        default_params = {}
        default_params["saofile"] = self.sao_file
        default_params["algorithm"] = "CONTOUR"
        default_params["constraints"] = "EARTH"
        default_params["gtifile"] = "INFILE"
        default_params["occultation"] = "fraction"
        default_params["method"] = "POSITION"
        default_params["timesegtol"] = 5.0
        default_params["multfiles"] = f"{infile}[BAT_PCODE_1]"
        default_params["divfiles"] = ",".join(
            f"{infile}[{i}]" for i in range(self.numbins)
        )
        default_params["clobber"] = "YES"

        _, batoccultmap_params = initalize_heasoft_task("batoccultmap", params)

        # Replace mandatory params into the user-provided params, ensuring that mandatory params take precedence
        batoccultmap_params.update(mandatory_params)
        batoccultmap_params.update(default_params)

        batoccultmap_task, batoccultmap_params = initalize_heasoft_task("batoccultmap", batoccultmap_params)


        batoccultmap_log = batoccultmap_task(**batoccultmap_params)
        # print(batoccultmap_log.stdout)
        if "batoccultmap" not in self.all_params:
            self.all_params["batoccultmap"] = [batoccultmap_params]
        else:
            self.all_params["batoccultmap"].append(batoccultmap_params)
        if "batoccultmap" not in self.all_logs:
            self.all_logs["batoccultmap"] = [batoccultmap_log]
        else:
            self.all_logs["batoccultmap"].append(batoccultmap_log)

    def batcelldetect(
        self, pid: str, infile: str, outfile: str, params: Dict[str, Any]
    ):
        """Source finder that runs on the image with the provided parameters and outputs a source list.

        Args:
            pid (str): Process ID for the observation (e.g. "00080153006")
            infile (str): Input image file to process (e.g. "point_20233341529_1.img")
            outfile (str): Catalog file to create with detected sources (e.g. "point_20233341529_1.cat")
            params (Dict[str, Any]): Dictionary of parameters to pass to batcelldetect,
            which may include overrides for defaults.
        """

        mandatory_params = {}
        mandatory_params["infile"] = infile
        mandatory_params["outfile"] = outfile

        defaults_params = {}
        defaults_params["snrthresh"] = "5"
        defaults_params["incatalog"] = "NONE"
        defaults_params["pcodefile"] = f"{infile}[BAT_PCODE_1]"
        defaults_params["regionfile"] = str(Path(outfile).with_suffix(".reg"))
        defaults_params["newsrcname"] = f"{pid}_%02d"
        defaults_params["newsrcind"] = 1
        defaults_params["pcodethresh"] = "0.05"
        defaults_params["bkgpcodethresh"] = "0.01"
        defaults_params["nullborder"] = "YES"
        defaults_params["bkgvarmap"] = str(Path(outfile).with_suffix(".var"))
        defaults_params["posfit"] = "NO"
        defaults_params["posfitwindow"] = 0.0
        defaults_params["distfile"] = "CALDB"
        defaults_params["clobber"] = "YES"
        defaults_params["vectorflux"] = "YES"
        defaults_params["keepbits"] = "7"
        defaults_params["pospeaks"] = "YES"
        defaults_params["posfluxfit"] = "NO"
        defaults_params["keepkeywords"] = "*APP,OBS_ID,IMAGE_ID,RA_PNT,DEC_PNT,PA_PNT"

        batcelldetect_params = initalize_heasoft_task("batcelldetect", params)

        # Replace mandatory params into the user-provided params, ensuring that mandatory params take precedence
        for key, value in mandatory_params.items():
            batcelldetect_params[key] = value

        for key, value in defaults_params.items():
            if key not in batcelldetect_params:
                batcelldetect_params[key] = value
        # print("Running batcelldetect with parameters:", batcelldetect_params)
        batcelldetect_log = self.backend.run("batcelldetect", **batcelldetect_params)
        # print(batcelldetect_log.stdout)
        if "batcelldetect" not in self.all_params:
            self.all_params["batcelldetect"] = [batcelldetect_params]
        else:
            self.all_params["batcelldetect"].append(batcelldetect_params)

        if "batcelldetect" not in self.all_logs:
            self.all_logs["batcelldetect"] = [batcelldetect_log]
        else:
            self.all_logs["batcelldetect"].append(batcelldetect_log)

    def ftcopy(self, infile: str, outfile: str):
        """Helper function to do basic manipulation of files with ftcopy
        and copy files

        Args:
            infile (str): Input catalog file to process (e.g. "point_20233341529_1.cat")
            outfile (str): Output catalog file to create (e.g. "point_20233341529_1_clean.cat")
        """
        res = heatools.ftcopy(
            infile=infile,
            outfile=outfile,
            clobber="YES",
            copyall="YES",
        )
        if "ftcopy" not in self.all_logs:
            self.all_logs["ftcopy"] = [res]
        else:
            self.all_logs["ftcopy"].append(res)
        # print(res)

    def batmaskwtimg(
        self, pid: str, infile: str, maskfile: str, outfile: str, params: Dict[str, Any]
    ):
        """Run batmaskwtimg on the given input file with the provided parameters and output the masked image.

        Args:
            pid (str): Pointing ID (e.g. "point_20233341529_1")
            infile (str): Input image file to process (e.g. "point_20233341529_1.img")
            maskfile (str): Detector mask file to use (e.g. "point_20233341529.detmask")
            outfile (str): Output masked image file to create (e.g. "point_20233341529_1.maskwtimg")
            params (Dict[str, Any]): Dictionary of parameters to pass to batmaskwtimg, which may include overrides for defaults.
        """
        mandatory_params = {}
        mandatory_params["infile"] = infile
        mandatory_params["detmask"] = maskfile
        mandatory_params["outfile"] = outfile

        default_params = {}
        default_params["attitude"] = self.pointing_info[pid]["attfile"]
        default_params["aperture"] = "CALDB:FLUX"
        default_params["teldef"] = "CALDB"
        default_params["combmeth"] = "MAX"
        default_params["outtype"] = "NONZERO"
        default_params["clobber"] = "YES"

        batmaskwtimg_params = initalize_heasoft_task("batmaskwtimg", params)

        # Replace mandatory params into the user-provided params, ensuring that mandatory params take precedence
        for key, value in mandatory_params.items():
            batmaskwtimg_params[key] = value

        for key, value in default_params.items():
            if key not in batmaskwtimg_params:
                batmaskwtimg_params[key] = value

        batmaskwtimg_log = self.backend.run("batmaskwtimg", **batmaskwtimg_params)
        if "batmaskwtimg" not in self.all_params:
            self.all_params["batmaskwtimg"] = [batmaskwtimg_params]
        else:
            self.all_params["batmaskwtimg"].append(batmaskwtimg_params)
        if "batmaskwtimg" not in self.all_logs:
            self.all_logs["batmaskwtimg"] = [batmaskwtimg_log]
        else:
            self.all_logs["batmaskwtimg"].append(batmaskwtimg_log)

    def ftimgcalc(self, params: Dict[str, Any]):
        """Run ftimgcalc on the given input file with the provided parameters
        and output the calculated image.

        Args:
            params (Dict[str, Any]): Dictionary of parameters to pass to ftimgcalc, which may
            include overrides for defaults. Must include the "expr" parameter with the
            expression to calculate, and any of the optional
            parameters a,b,c,d,e,f,g,h,z,nvectimages,wcsimage,resultname,replicate as needed.
        """
        ftimgcalc_params = initalize_heasoft_task("ftimgcalc", params)
        res = self.backend.run(
            "ftimgcalc", **ftimgcalc_params
        )  # hsp.ftimgcalc(**ftimgcalc_params)
        if "ftimgcalc" not in self.all_params:
            self.all_params["ftimgcalc"] = [ftimgcalc_params]
        else:
            self.all_params["ftimgcalc"].append(ftimgcalc_params)
        if "ftimgcalc" not in self.all_logs:
            self.all_logs["ftimgcalc"] = [res]
        else:
            self.all_logs["ftimgcalc"].append(res)
        # print(res)

    @staticmethod
    def _met_to_stamp(met: float) -> str:
        tref = datetime(2001, 1, 1, tzinfo=timezone.utc).timestamp()
        dt = datetime.fromtimestamp(tref + met, tz=timezone.utc)
        return dt.strftime("%Y%j%H%M")

    def _copy_cleaned_sources_into_map(
        self,
        prevcat: str,
        previmg: str,
        img: str,
        proot: str,
        nebins: int,
        cleanrad: float,
    ) -> None:
        """
        Direct port of the "cheesemap" logic.
        Uses ftimgcalc iteratively to avoid long command lines, same as Perl.
        """
        try:
            with fits.open(prevcat, mode="readonly") as hdul:
                tab = None
                # try ext 2 then ext 1
                for idx in (2, 1):
                    if idx < len(hdul) and isinstance(hdul[idx], fits.BinTableHDU):
                        tab = hdul[idx].data
                        break
                if tab is None:
                    return
                imx = np.array(tab["IMX"], dtype=float)
                imy = np.array(tab["IMY"], dtype=float)
                cleaned = (
                    np.array(tab["CLEANED"], dtype=bool)
                    if "CLEANED" in tab.columns.names
                    else np.zeros(len(tab), dtype=bool)
                )
        except Exception:
            return

        exprlist: List[str] = []
        for i in range(len(cleaned)):
            if cleaned[i]:
                exprlist.append(f"circle({imx[i]},{imy[i]},{cleanrad},A.IMX,A.IMY)")

        cheesemap = proot + ".cheesemap"
        if Path(cheesemap).exists():
            Path(cheesemap).unlink(missing_ok=True)

        while len(exprlist) > 0:
            expr = ""
            # batch up to ~800 chars
            while exprlist and len(expr) < 800:
                e = exprlist.pop()
                expr = (expr + "||" + e) if expr else e

            opts = ""
            cheesemap1 = cheesemap + "1"
            if Path(cheesemap).exists():
                shutil.move(cheesemap, cheesemap1)
                expr = expr + " || (B == 1)"
                opts = f"b='{cheesemap1}'"

            self.ftimgcalc(
                params={
                    "outfile": cheesemap,
                    "expr": f"({expr})?1:0",
                    "a": img,
                    "wcsimage": ":A",
                    "clobber": "YES",
                    **({"b": cheesemap1} if opts else {}),
                }
            )
            if Path(cheesemap1).exists():
                Path(cheesemap1).unlink(missing_ok=True)

        if Path(cheesemap).exists():
            oldimg = img + ".orig"
            if Path(img).exists():
                shutil.move(img, oldimg)

            self.ftimgcalc(
                params={
                    "outfile": img,
                    "expr": "(CHEESE>0)?(OLD):(NEW)",
                    "a": f"NEW={str(oldimg)}",
                    "b": f"CHEESE={str(cheesemap)}",
                    "c": f"OLD={str(previmg)}",
                    "clobber": "YES",
                    "replicate": "YES",
                    "bunit": ":NEW",
                    "wcsimage": ":NEW",
                    "nvectimages": nebins,
                    "otherext": "+NEW",
                    "bitpix": "E",
                    "resultname": "BAT_IMAGE",
                }
            )

            Path(oldimg).unlink(missing_ok=True)

    def _bright_source_filtering(
        self,
        pid: str,
        dpi: str,
        dmask: str,
        cat: str,
        proot: str,
        maskedge: str,
    ) -> tuple[str, bool, str, str]:
        """
        Port of the iter-1 bright filtering block. Returns True to continue, False to fail snapshot.
        """
        brightexpr = (
            f"[(SUM(RATE) > {self.params['brightthresh']} || "
            f"MAX(RATE) > {self.params['brightthresh']}) && "
            f"PCODEFR > {self.params['pcodethresh']}]"
        )

        temp_cat = self.scratchdir / "temp.cat"

        self.ftcopy(infile=f"{cat}{brightexpr}", outfile=str(temp_cat))
        if not temp_cat.exists():
            return (
                dmask,
                False,
                "bright_cat_failed1",
                "Could not create bright source catalog",
            )

        nbright = 0
        with fits.open(temp_cat, mode="readonly") as hdul:
            if len(hdul) < 2 or not isinstance(hdul[1], fits.BinTableHDU):
                nbright = 0
            else:
                nbright = len(hdul[1].data)

        print(f"Found {nbright} sources for bright filtering")

        if nbright <= 0:
            return dmask, True, "ok", "success"

        # block mask edge shadows
        scrdpi1 = self.scratchdir / "bright_temp1.dpi"
        newdmask = f"{proot}_1_maskedge.detmask"

        try:
            self.batmaskwtimg(
                pid=pid,
                outfile=scrdpi1,
                infile=dpi,
                maskfile=dmask,
                params={
                    "ra": 0.0,
                    "dec": 0.0,
                    "aperture": maskedge,
                    "incatalog": str(temp_cat),
                },
            )
        except Exception as e:
            print(f"Error running batmaskwtimg for bright source filtering: {e}")
            return dmask, False, "maskedge_failed1", "Could not create mask edge map"

        try:
            self.ftimgcalc(
                params={
                    "outfile": newdmask,
                    "expr": "(A>0 || B>0)?(1):(0)",
                    "a": scrdpi1,
                    "b": dmask,
                    "wcsimage": ":A",
                    "resultname": "BAT_DPI",
                    "clobber": "YES",
                }
            )
        except Exception as e:
            print(f"Error running ftimgcalc for bright source filtering: {e}")
            return (
                dmask,
                False,
                "maskedge_failed2",
                "Could not create combined mask edge map",
            )
        return newdmask, True, "ok", "success"

    def _sigma_cut(
        self,
        dpi: str,
        bkgdpi1: str,
        dmask: str,
        newdmask: str,
    ) -> tuple[bool, str, str]:

        # First get exposure
        try:
            with fits.open(dpi, mode="readonly") as hdul:
                for hdu in hdul[1:]:
                    if "EXPOSURE" in hdu.header:
                        expo = float(hdu.header["EXPOSURE"])
                        break
        except Exception:
            expo = 0.0

        scrdpi1 = self.scratchdir / "sigma_temp1.dpi"
        model_expr = "MODEL"
        patternmap = (
            self.params["global_pattern_map"]
            if self.params["global_pattern_map"] != "NONE"
            else None
        )
        if patternmap:
            pattern_par = f"PATTERN={patternmap}"
            model_expr = "(MODEL + DEFNULL(PATTERN,0))"
        else:
            pattern_par = "NONE"

        expr1 = f"(ABS(DATA - {model_expr} )*SQRT({expo}/(MODEL + 1E-20)) > {self.params['badpixthresh']}) ? 1 : 0"

        try:
            self.ftimgcalc(
                params={
                    "outfile": scrdpi1,
                    "expr": expr1,
                    "a": f"DATA={dpi}",
                    "b": f"MODEL={bkgdpi1}",
                    "c": pattern_par,
                    "wcsimage": ":DATA",
                    "resultname": "BAT_DPI",
                    "clobber": "YES",
                    "nvectimages": self.numbins,
                }
            )
        except Exception as e:
            print(f"Error running ftimgcalc for sigma cut: {e}")
            return False, "sigmamask1_failed", "Could not create sigma cut map (1)"

        expr2 = "MASTER"
        extra_params = {}
        extra_params["z"] = f"MASTER={dmask}"
        for nn in range(0, self.numbins):
            expr2 += f"+M{nn}"
            extra_params[f"{chr(65+32+nn)}"] = f"M{nn}={scrdpi1}[{nn}]"

        params = {
            "outfile": newdmask,
            "expr": f"({expr2}) != 0 ? 1 : 0",
            "wcsimage": ":M0",
            "clobber": "YES",
            "nvectimages": 1,
            "resultname": "BAT_DPI",
        }
        for key, value in extra_params.items():
            params[key] = value
        try:
            self.ftimgcalc(params=params)
        except Exception as e:
            print(f"Error running ftimgcalc for sigma cut master mask: {e}")
            return False, "sigmamask2_failed", "Could not create sigma cut map (2)"
        return True, "ok", "success"

    def _process_pointing(self, pid, params) -> _PointStats:
        """This is the main function that processes a single pointing,
        which is defined by a GTI row. It will run through the various
        steps of the survey processing for that pointing, including
        creating the initial GTI file for that pointing, running batbinevt
        to create the DPI, running batsurvey-aspect to get the aspect
        solution and clean the GTI, running fimgstat to check the DPI,
        creating the detector mask with batsurvey-detmask, running
        batfftimage to create the FFT image, running batoccultmap to
        create the occultation map, running batcelldetect to detect sources,
        and then running batclean to clean the DPI based on the detected
        sources. It will also handle the iterations of cleaning and
        keep track of the intermediate files and logs for each step.

        Args:
            pid: Pointing ID to process (e.g. "20233341529")
            params: Dictionary of parameters to pass to each step, which may include overrides for defaults.
        Returns:
            _PointStats: _description_
        """
        self.current_pointstatus = "unknown"
        self.current_pointreason = ""
        pdir = self.pointing_info[pid]["dir"]
        pdir.mkdir(parents=True, exist_ok=True)
        proot = pdir / pid
        med_ra, med_dec, med_roll = -999.0, -999.0, -999.0
        expo = 0.0
        ndets = 0

        # First copy the master GTI row for this pointing to the pointing directory as pnt0
        self.ftcopy_gti_row()

        gti_files = self.pointing_info[pid].get("gti_files", [])
        if not gti_files:
            self.pntstat(
                pid, "pntgti0_failed", "Could not create GTI for this pointing"
            )
            return self._build_fail_pointstats(pid, med_ra, med_dec, med_roll)
        pnt0 = gti_files[0]
        if not Path(pnt0).exists():
            self.pntstat(
                pid, "pntgti0_failed", "Could not create GTI for this pointing"
            )
            return self._build_fail_pointstats(pid, med_ra, med_dec, med_roll)

        # Make necessary files for binevt
        survey_files_list = f"@{self.esurvey_lis}"
        dpi1 = Path(f"{proot}_1.dpi")
        pnt1 = Path(f"{proot}_pnt1.gti")
        pnt = Path(f"{proot}_pnt.gti")

        self.batbinevt(
            pid=pid,
            infile=survey_files_list,
            outfile=str(dpi1),
            gtifile=str(pnt0),
            params=params,
        )
        if not dpi1.exists():
            self.pntstat(pid, "dpi1_failed", "Could not create first output DPI")
            return self._build_fail_pointstats(pid, med_ra, med_dec, med_roll)
        # Write back to info for subsequent steps
        self.pointing_info[pid]["dpi"] = [dpi1]

        # Then run batsurvey-aspect to get the aspect solution and clean the GTI,
        # which will produce the pnt1 and att files, and then decide whether to
        # use pnt0 or pnt1 based on the exposure and bad time fractions, and
        # copy the chosen GTI to the final pnt file for this pointing
        _expotot, _expobad, med_ra, med_dec, med_roll = self.batsurvey_aspect(
            pid=pid,
            gti_file=str(pnt0),
            outgti_file=str(pnt1),
            finalgti_file=str(pnt),
            params=params,
        )

        if (
            str(self.params.get("pointing_check", "YES")).upper() == "YES"
            and not pnt.exists()
        ):
            self.pntstat(
                pid,
                "pointcheck_failed",
                "Attitude check failed, too much attitude drift",
            )
            return self._build_fail_pointstats(pid, med_ra, med_dec, med_roll)

        # goodfrac = 1.0 - float(self.params["pointerr_frac_time"])
        second_pass_params = dict(params or {})
        second_pass_params["min_dph_frac_overlap"] = 1.0 - float(
            self.params["pointerr_frac_time"]
        )
        second_pass_params["max_dph_time_nonoverlap"] = float(
            self.params["pointerr_abs_time"]
        )
        self.batbinevt(
            pid=pid,
            infile=survey_files_list,
            outfile=str(dpi1),
            gtifile=self.pointing_info[pid]["gti"],
            params=second_pass_params,
        )
        if not dpi1.exists():
            self.pntstat(pid, "dpi2_failed", "Could not create second output DPI")
            return self._build_fail_pointstats(pid, med_ra, med_dec, med_roll)

        try:
            expo = float(fits.getval(dpi1, "EXPOSURE", ext=1))
        except Exception:
            expo = 0.0
        if expo < float(self.params.get("expothresh", 150.0)):
            self.pntstat(
                pid,
                "expo_small",
                f"Exposure {expo} < {self.params.get('expothresh', 150.0)}",
                exposure=expo,
            )
            return self._build_fail_pointstats(
                pid, med_ra, med_dec, med_roll, raw_exposure=expo
            )

        dmin, dmax, dsum = self.fimgstat(str(dpi1))
        if dmin == 0 and dmax == 0:
            self.pntstat(pid, "zero_counts", "Output DPI was zero")
            return self._build_fail_pointstats(
                pid, med_ra, med_dec, med_roll, raw_exposure=expo
            )

        # then generate detector mask with batsurvey-detmask, which will
        # produce the detmask file for this pointing, and then use that
        # same detmask for all subsequent steps for this pointing

        detmask = Path(f"{proot}.detmask")
        self.batsurvey_detmask(
            pid=pid, infile=str(dpi1), maskfile=str(detmask), params=params
        )
        if not detmask.exists():
            self.pntstat(pid, "batdetmask_failed", "Could not create pointing detmask")
            return self._build_fail_pointstats(
                pid, med_ra, med_dec, med_roll, raw_exposure=expo
            )

        # Run once to get background image
        bkgdpi = Path(f"{proot}_1.bkgdpi")

        batclean_params = params.copy() if params else {}
        batclean_params["aperture"] = "NONE"
        batclean_params["outversion"] = "fit"
        batclean_params["srclean"] = "NO"
        batclean_params["aperture"] = "NONE"
        batclean_params["maskfit"] = "YES"
        self.batclean(
            infile=str(dpi1),
            outfile=str(bkgdpi),
            detmask=str(detmask),
            params=batclean_params,
        )
        if not bkgdpi.exists():
            self.pntstat(pid, "batclean1_failed", "batclean failure (1)")
            return self._build_fail_pointstats(
                pid, med_ra, med_dec, med_roll, raw_exposure=expo
            )

        # Make var map
        poivar = Path(f"{proot}.poivar")
        occult_map = Path(f"{proot}.occimg")
        srccat = self.params["incatalog"]

        ncleaniter = int(self.params["ncleaniter"])
        for cleaniter in range(1, ncleaniter + 1):
            cleaniterp1 = cleaniter + 1

            img = Path(f"{proot}_{cleaniter}.img")
            var = Path(f"{proot}_{cleaniter}.var")
            cat = Path(f"{proot}_{cleaniter}.cat")

            self.batfftimage(
                pid=pid,
                infile=str(dpi1),
                outfile=str(img),
                detmask=str(detmask),
                bkgfile=str(bkgdpi),
                bkgvarmap=str(poivar),
                params=params,
            )

            if not img.exists():
                self.pntstat(
                    pid,
                    "batfftimage_failed",
                    "Could not create initial sky maps",
                    exposure=expo,
                )
                return self._build_fail_pointstats(
                    pid, med_ra, med_dec, med_roll, raw_exposure=expo
                )

            self.batoccultmap(
                infile=str(img),
                outfile=str(occult_map),
                params=params,
            )
            if not occult_map.exists():
                self.pntstat(
                    pid,
                    "batoccultmap_failed",
                    "Could not create occultation exposure map",
                    exposure=expo,
                )
                return self._build_fail_pointstats(
                    pid, med_ra, med_dec, med_roll, raw_exposure=expo
                )

            if cleaniter > 1:
                self._copy_cleaned_sources_into_map(
                    prevcat=str(prev_cat),
                    previmg=str(prev_img),
                    img=str(img),
                    proot=str(proot),
                    nebins=self.numbins,
                    cleanrad=float(self.params.get("copy_cleaned_radius", 0.008)),
                )

            batcelldetect_params = params.copy() if params else {}
            batcelldetect_params["incatalog"] = srccat
            if cleaniter > 1:
                batcelldetect_params["newsrcind"] = 10
            if self.truncated:
                good_rows, bad_rows = identify_truncated_images(infile=str(img), force=True)
                batcelldetect_params["rows"] = ",".join(np.array(good_rows).astype(str))

            self.batcelldetect(
                pid=pid.replace("point_", ""),
                infile=str(img),
                outfile=str(cat),
                params=batcelldetect_params,
            )

            # Then patch the results
            if not cat.exists():
                self.pntstat(
                    pid,
                    "batcelldetect_failed",
                    "Could not create initial catalog",
                    exposure=expo,
                )
                return self._build_fail_pointstats(
                    pid, med_ra, med_dec, med_roll, raw_exposure=expo
                )
            elif self.truncated:
                patch_truncated_results(
                    files=[str(img), str(var), str(cat)],
                    good_rows=good_rows,
                    bad_rows=bad_rows,
                )

            tmpcat = Path(str(cat) + "s")
            cleanlev_expr = (
                f"CLEANLEV = {cleaniter}; "
                f'#TTYPE#(batclean iteration number) = "CLEANLEV"; '
            )
            if cleaniter > 1:
                # Only reset CLEANED flag upon first iteration? (Perl comment inconsistent; we follow code structure)
                cleanlev_expr = (
                    "CLEANED=F; " '#TTYPE#(Has this source been cleaned?) = "CLEANED";'
                )

            expr = (
                f"{cleanlev_expr}"
                "VECTSNR = SNR; "
                '#TTYPE#(S/N ratios for each band) = "VECTSNR";'
                '#TDISP#(display format) = "F8.2"; '
                "-SNR; "
                "TOTSNR=SUM(RATE)/SQRT(SUM(BKG_VAR**2)); "
                '#TTYPE#(S/N ratio - all bands combined) = "TOTSNR"; '
                '#TDISP#(display format) = "F8.2"; '
                "SNR = MAX({VECTSNR,TOTSNR}); "
                '#TTYPE#(Maximum S/N ratio found - {VECT,TOT}SNR) = "SNR"; '
                '#TDISP#(display format) = "F8.2"; '
            )

            self.ftcopy(
                f"{cat}[1][col *; {expr}]",
                str(tmpcat),
            )
            if not tmpcat.exists():
                self.pntstat(
                    pid,
                    "modcat_failed",
                    "Could not create modified catalog",
                    exposure=expo,
                )
                return self._build_fail_pointstats(
                    pid, med_ra, med_dec, med_roll, raw_exposure=expo
                )

            # Then move it back
            shutil.move(str(tmpcat), str(cat))

            if cleaniter == ncleaniter and ncleaniter > 1:
                break

            elif cleaniter == 1:
                detmask, success, fail_code, fail_desc = self._bright_source_filtering(
                    pid=pid,
                    dpi=str(dpi1),
                    dmask=str(detmask),
                    cat=str(cat),
                    proot=str(proot),
                    maskedge="CALDB:MASK_EDGES",
                )
                if not success:
                    self.pntstat(pid, fail_code, fail_desc, exposure=expo)
                    return self._build_fail_pointstats(
                        pid, med_ra, med_dec, med_roll, raw_exposure=expo
                    )

                # transition catalog
                bkgdpi1 = Path(f"{proot}_{cleaniter}a.bkgdpi")
                trancat = Path(f"{proot}_{cleaniter}a.cat")
                clean_expr = f"(SNR > {self.params['cleansnr']})"
                if str(self.params.get("cleanexpr", "NONE")).upper() != "NONE":
                    clean_expr = f"{clean_expr} || ({self.params['cleanexpr']})"
                # print(f"Using clean expression: {clean_expr}")

                try:
                    self.ftcopy(
                        f"{cat}[col *;CLEANED=({clean_expr});]",
                        str(trancat),
                    )
                except Exception as e:
                    print(f"Error running ftcopy for transition catalog: {e}")
                    self.pntstat(
                        pid,
                        "cleancat_failed",
                        "Could not update catalog CLEANED flag",
                        exposure=expo,
                    )
                    return self._build_fail_pointstats(
                        pid, med_ra, med_dec, med_roll, raw_exposure=expo
                    )

                # Match Perl batsurvey flow: transition catalog is the catalog used
                # for the remaining cleaning analysis in this iteration.
                cat = trancat

                ncatrows = 0
                with fits.open(str(cat)) as cat_hdul:
                    if len(cat_hdul) > 1 and cat_hdul[1].data is not None:
                        ncatrows = len(cat_hdul[1].data)

                if ncatrows > 0:
                    print(f"{ncatrows} sources cleaned during iteration {cleaniter}")
                    self.batclean(
                        infile=str(dpi1),
                        outfile=str(bkgdpi1),
                        detmask=str(detmask),
                        params={
                            "incatalog": f"{cat}[1][CLEANED == T]",
                            "bkgmodel": "SIMPLE",
                            "cleansnr": "0.0001",
                            "snrcol": "SNR",
                            "srcclean": "YES",
                            "aperture": "CALDB:FLUX",
                            "maskfit": "YES",
                            "clobber": "YES",
                            "outversion": "fit",
                        },
                    )
                else:
                    # Match Perl behavior: skip intermediate source-clean stage.
                    bkgdpi1 = bkgdpi
                    print("(no bright sources; skipping intermediate clean step)")

                sigma_mask = Path(f"{proot}_1_sigma.detmask")
                success, fail_code, fail_desc = self._sigma_cut(
                    dpi=str(dpi1),
                    bkgdpi1=str(bkgdpi1),
                    dmask=str(detmask),
                    newdmask=str(sigma_mask),
                )

                if not success:
                    self.pntstat(pid, fail_code, fail_desc, exposure=expo)
                    return self._build_fail_pointstats(
                        pid, med_ra, med_dec, med_roll, raw_exposure=expo
                    )

                detmask = sigma_mask

                dmin, dmax, dsum = self.fimgstat(infile=str(detmask))
                ndets = 49478 - dsum
                print(f"Number of enabled detectors: {ndets}")
                if ndets < float(self.params["detthresh2"]):
                    self.pntstat(
                        pid,
                        "ndets_low",
                        f"Too few good detectors in detector mask ({ndets} < {self.params['detthresh2']})",
                    )
                    return self._build_fail_pointstats(
                        pid,
                        med_ra,
                        med_dec,
                        med_roll,
                        ndets=int(ndets),
                        raw_exposure=expo,
                    )

                # One last final clean to make sure everything is cleaned for
                # the next iteration, even if no bright sources were found.
                # This matches the Perl flow, which always does this final
                # clean step regardless of whether bright sources were found
                # or not.
                cleaniterp1 = cleaniter + 1
                bkgdpi_next = Path(f"{proot}_{cleaniterp1}.bkgdpi")
                if self.params.get("global_pattern_map", "NONE") != "NONE":
                    adjdpi = self.scratchdir / "raw_minus_pattern.dpi"
                    try:
                        self.ftimgcalc(
                            params={
                                "outfile": adjdpi,
                                "expr": f"RAW - DEFNULL(PATTERN,0)",
                                "a": f"DATA={dpi1}",
                                "b": f"PATTERN={self.params['global_pattern_map']}",
                                "replicate": "YES",
                                "bunit": ":RAW",
                                "wcsimage": ":RAW",
                                "clobber": "YES",
                                "nvectimages": self.numbins,
                                "otherext": "+RAW",
                                "resultname": "BAT_DPI",
                            }
                        )
                    except Exception as e:
                        print(
                            f"Error running ftimgcalc for global pattern adjustment: {e}"
                        )
                        self.pntstat(
                            pid,
                            "pre_batclean_fudge_failed",
                            "Could not create fudged rate map for final clean",
                            exposure=expo,
                        )
                        return self._build_fail_pointstats(
                            pid,
                            med_ra,
                            med_dec,
                            med_roll,
                            ndets=int(ndets),
                            raw_exposure=expo,
                        )
                else:
                    adjdpi = dpi1

                if ("backexp" not in params) or (
                    str(params.get("backexp", "NONE")).upper() == "NONE"
                ):
                    params["backexp"] = f"{proot}.backexp"

                final_clean_params = {
                    "incatalog": f"{cat}[1][CLEANED == T]",
                    "bkgmodel": "SIMPLE",
                    "cleansnr": "0.0001",
                    "maskfit": "YES",
                    "srcclean": "YES",
                    "aperture": "CALDB:FLUX",
                    "backexp": params["backexp"],
                    "balance": "ShortEdges,LongEdges,InOut",
                }

                self.batclean(
                    infile=str(adjdpi),
                    outfile=str(bkgdpi_next),
                    detmask=str(detmask),
                    params=final_clean_params,
                )
                if not bkgdpi_next.exists():
                    self.pntstat(
                        pid,
                        "batclean3_failed",
                        "Could not create source clean map (final)",
                        exposure=expo,
                    )
                    return self._build_fail_pointstats(
                        pid,
                        med_ra,
                        med_dec,
                        med_roll,
                        ndets=int(ndets),
                        raw_exposure=expo,
                    )

                # Add back the pattern if we had subtracted it, to
                # maintain consistency for the next iteration
                if self.params.get("global_pattern_map", "NONE") != "NONE":
                    rawname = bkgdpi_next + ".raw"
                    shutil.move(bkgdpi_next, rawname)
                    self.ftimgcalc(
                        params={
                            "outfile": bkgdpi_next,
                            "expr": f"BKG + DEFNULL(PATTERN,0)",
                            "a": f"BKG={rawname}",
                            "b": f"PATTERN={self.params['global_pattern_map']}",
                            "replicate": "YES",
                            "bunit": ":BKG",
                            "wcsimage": ":BKG",
                            "clobber": "YES",
                            "nvectimages": self.numbins,
                            "otherext": "+BKG",
                            "resultname": "BAT_DPI",
                        }
                    )
                    if not bkgdpi_next.exists():
                        self.pntstat(
                            pid,
                            "post_batclean_fudge_failed",
                            "Could not create fudged background map after final clean",
                            exposure=expo,
                        )
                        return self._build_fail_pointstats(
                            pid,
                            med_ra,
                            med_dec,
                            med_roll,
                            ndets=int(ndets),
                            raw_exposure=expo,
                        )

            newdmask = Path(f"{proot}_{cleaniterp1}.detmask")
            shutil.copy2(detmask, newdmask)
            detmask = newdmask
            bkgdpi = bkgdpi_next

            # Copy variables
            prev_img = img
            prev_cat = cat
            srccat = cat
            prev_var = var
            prev_bkgdpi = bkgdpi

        # Post-processing diagnostics
        chifile = Path(f"{proot}_chi.fits")
        self.ftimgcalc(
            params={
                "outfile": str(chifile),
                "expr": "SUM((MASK == 0)?((DATA-MODEL)**2/MODEL*#EXPOSURE):0)",
                "a": f"DATA={dpi1}",
                "b": f"MODEL={bkgdpi}",
                "c": f"MASK={detmask}",
                "nvectimages": self.numbins,
                "replicate": "YES",
                "wcsimage": ":DATA",
                "clobber": "YES",
            },
        )

        chi = [0.0] * self.numbins
        if chifile.exists():
            chi = self._read_per_band_scalar_fits(chifile)

        bkgfile = Path(f"{proot}_totbkg.fits")
        self.ftimgcalc(
            params={
                "outfile": str(bkgfile),
                "expr": "SUM((MASK == 0)?(DATA*#EXPOSURE):0)",
                "a": f"DATA={dpi1}",
                "b": f"MASK={detmask}",
                "nvectimages": self.numbins,
                "replicate": "YES",
                "wcsimage": ":DATA",
                "clobber": "YES",
            },
        )
        bkg_counts = [0.0] * self.numbins
        if bkgfile.exists():
            bkg_counts = self._read_per_band_scalar_fits(bkgfile)

        try:
            expo = float(fits.getval(dpi1, "EXPOSURE", ext=1))
        except Exception:
            expo = float(expo) if expo else 0.0
        ndets = int(
            49478
            - float(
                subprocess.check_output(
                    "pget fimgstat sum", shell=True, text=True
                ).strip()
                or 0
            )
        )
        return _PointStats(
            image_id=pid,
            status=True,
            descr="ok",
            tstart=self.pointing_info[pid]["tstart_met"],
            tstop=self.pointing_info[pid]["tstop_met"],
            raw_exposure=expo,
            exposure=expo,
            ra_pnt=med_ra,
            dec_pnt=med_dec,
            pa_pnt=med_roll,
            ndets=ndets,
            date_obs=self._met_to_datestr(self.pointing_info[pid]["tstart_met"]),
            date_end=self._met_to_datestr(self.pointing_info[pid]["tstop_met"]),
            numband=self.numbins,
            chi2=chi,
            bkg_counts=bkg_counts,
        )

    def _write_stats(
        self,
        starts: Optional[List[float]] = None,
        stops: Optional[List[float]] = None,
        totexpo: Optional[float] = None,
        goodexpo: Optional[float] = None,
    ):
        obs_id = getattr(self, "obs_id", self.indir.name)

        if starts is None:
            starts = [float(s.tstart) for s in self.point_stats]
        if stops is None:
            stops = [float(s.tstop) for s in self.point_stats]
        if totexpo is None:
            totexpo = float(sum(float(s.raw_exposure) for s in self.point_stats))
        if goodexpo is None:
            goodexpo = float(
                sum(float(s.exposure) for s in self.point_stats if bool(s.status))
            )

        out_point = self.outdir / "stats_point.dat"
        with out_point.open("w") as f:
            for s in self.point_stats:
                f.write(
                    f"B 6.16 {self.indir.name} {s.image_id} {int(s.status)} {s.descr} "
                    f"{s.tstart} {s.tstop} {s.ra_pnt} {s.dec_pnt} {s.pa_pnt} {s.raw_exposure} {s.exposure} {s.ndets} "
                    f"{len(s.chi2)} {' '.join(str(x) for x in s.chi2)}\n"
                )

        stats_tab = Table(
            rows=[
                (
                    obs_id,
                    s.image_id,
                    b"6.16",
                    b"NONE",
                    s.date_obs,
                    s.date_end,
                    s.tstart,
                    s.tstop,
                    s.raw_exposure,
                    s.exposure,
                    s.status,
                    s.descr,
                    s.ra_pnt,
                    s.dec_pnt,
                    s.pa_pnt,
                    s.ndets,
                    s.numband,
                    s.chi2,
                    s.bkg_counts,
                )
                for s in self.point_stats
            ],
            names=[
                "OBS_ID",
                "IMAGE_ID",
                "BSURVER",
                "BSURSEQ",
                "DATE_OBS",
                "DATE_END",
                "TSTART",
                "TSTOP",
                "RAW_EXPOSURE",
                "EXPOSURE",
                "IMAGE_STATUS",
                "IMAGE_DESCR",
                "RA_PNT",
                "DEC_PNT",
                "PA_PNT",
                "NBATDETS",
                "NUMBAND",
                "CHI2",
                "BKG_COUNTS",
            ],
        )
        stats_tab.write(self.outdir / "stats_point.fits", overwrite=True)

        n_good = sum(1 for s in self.point_stats if s.status)
        obs = Table(
            rows=[
                (
                    obs_id,
                    min(starts) if starts else 0,
                    max(stops) if stops else 0,
                    totexpo,
                    goodexpo,
                    len(starts),
                    n_good,
                )
            ],
            names=[
                "OBS_ID",
                "TSTART",
                "TSTOP",
                "RAW_EXPOSURE",
                "EXPOSURE",
                "N_RAW_IMAGES",
                "N_IMAGES",
            ],
        )
        obs.write(self.outdir / "stats_obs.fits", overwrite=True)
        (self.outdir / "stats_obs.dat").write_text(
            f"A {self.indir.name} {self.outdir.name} {totexpo} {goodexpo} {len(starts)} {n_good} 1\n"
        )

    def _dump_all_logs(self, logfile: Optional[Path] = None) -> Path:
        if logfile is None:
            logfile = self.outdir / "batsurvey_translated.log"

        lines: List[str] = []
        lines.append("==========================================================")
        lines.append("batsurvey_translated task log")
        lines.append("==========================================================")

        for task_name, raw_entries in self.all_logs.items():
            entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
            lines.append(f"\n## {task_name} ({len(entries)} call(s))")

            for idx, entry in enumerate(entries, start=1):
                lines.append(f"-- call {idx} --")
                if isinstance(entry, ToolResult):
                    lines.append(f"task={entry.task}")
                    lines.append(f"returncode={entry.returncode}")
                    lines.append(
                        "params="
                        + json.dumps(entry.params, default=str, sort_keys=True)
                    )
                    if entry.stdout:
                        lines.append("stdout:")
                        lines.append(entry.stdout.rstrip("\n"))
                    if entry.stderr:
                        lines.append("stderr:")
                        lines.append(entry.stderr.rstrip("\n"))
                else:
                    lines.append(f"raw_entry={repr(entry)}")

        logfile.write_text("\n".join(lines) + "\n")
        return logfile

    def run(self) -> bool:
        chatter = int(self.params.get("chatter", 2))
        if chatter >= 1:
            print("=" * 58)
            print("Running batsurvey modules on the OBSID:", self.obs_id)
            print("=" * 58)

        # First run steps that are common to all pointing times, then we can divide
        # The processing into individual pointings for the later steps that require it

        # First run baterebin to create the erebinned DPH files and masks
        try:
            self.baterebin(params=self.params)
        except Exception as exc:
            self.obsstat(
                code="baterebin_failed",
                desc=str(exc),
            )
            log_path = self._dump_all_logs()
            print(f"Wrote task log: {log_path}")
            raise ValueError("baterebin failed, see log for details")

        # Second get the GTIs for the pointings using batsurvey-gti, which
        # also creates the master GTI file and makes the pointing GTI files
        try:
            self.batsurvey_gti(params=self.params)
        except Exception as exc:
            self.obsstat(
                code="batsurvey_gti_failed",
                desc=str(exc),
            )
            log_path = self._dump_all_logs()
            print(f"Wrote task log: {log_path}")
            raise ValueError("batsurvey-gti failed, see log for details")

        # Then work on individual pointings

        for pid in self.pointing_info.keys():
            try:
                s = self._process_pointing(pid, params=self.params)
                # If this is sucessful, write out point_stat file
                self.pntstat(
                    pid, desc="success", code="ok", ndets=s.ndets, exposure=s.exposure
                )
            except Exception as exc:
                self.pntstat(pid, code="failed", desc=str(exc))
                s = self._build_fail_pointstats(pid, -999.0, -999.0, -999.0)
                raise ValueError(
                    f"Processing failed for pointing {pid}, see log for details"
                )
            self.point_stats.append(s)
            self._record_snapshot_stats(s)

        if len(self.point_stats) > 0:
            self._write_obs_stats()
            log_path = self._dump_all_logs()
            # Overwrite all stats again
            self._write_stats()

        print(f"Wrote task log: {log_path}")
        print("batsurvey-translated: COMPLETE")
        return True


if __name__ == "__main__":
    input_dict = dict(
        cleanexpr="ALWAYS_CLEAN==T",
        detthresh=6000,
        detthresh2=6000,
        snrthresh=5,
        cleansnr=5,
        clobber="YES",
        chatter=4,
        dph_pattern="INDIR/bat/survey/sw*.dph*",
        energybins="14-20,20-24,24-35,35-50,50-75,75-100,100-150,150-195",
        # patt_noise_dir="/proj/andreonilab/users/akasha/data/PATTERN_MAPS/",
    )
    survey_obj = BatTools(
        indir="/proj/andreonilab/users/akasha/realtime/data/00086912002",
        outdir="/proj/andreonilab/users/akasha/test/00086912002_surveyresult",
        truncated=True,
        params=input_dict,
    )
    survey_obj.run()
