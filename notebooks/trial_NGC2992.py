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

object_name='NGC2992'
object_location = swiftbat.simbadlocation(object_name)
object_batsource = swiftbat.source(ra=object_location[0], dec=object_location[1], name=object_name)

queryargs = dict(time="2004-12-15 .. 2005-12-16", fields='All', resultmax=0)
table_everything = ba.from_heasarc(**queryargs)

minexposure = 1000     # cm^2 after cos adjust
exposures = np.array([object_batsource.exposure(ra=row['RA'], dec=row['DEC'], roll=row['ROLL_ANGLE'])[0] for row in table_everything])
table_exposed = table_everything[exposures > minexposure]
print(f"Finding everything finds {len(table_everything)} observations, of which {len(table_exposed)} have more than {minexposure:0} cm^2 coded")

#result = ba.download_swiftdata(table_exposed)
obs_ids=[i for i in table_exposed['OBSID'] if result[i]['success']]

noise_map_dir=Path("/local/data/tparsota/PATTERN_MAPS/")
batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids, patt_noise_dir=noise_map_dir, nprocs=30)

batsurvey_obs=ba.parallel.batspectrum_analysis(batsurvey_obs, object_name, use_cstat=True, nprocs=30)

outventory_file=ba.merge_outventory(batsurvey_obs)
time_bins=ba.group_outventory(outventory_file, np.timedelta64(1, "M"), start_datetime=Time("2004-12-15"), end_datetime=Time("2005-12-16"))

mosaic_list, total_mosaic=ba.parallel.batmosaic_analysis(batsurvey_obs, outventory_file, time_bins, nprocs=3)

mosaic_list=ba.parallel.batspectrum_analysis(mosaic_list, object_name, use_cstat=True, nprocs=5)
total_mosaic=ba.parallel.batspectrum_analysis(total_mosaic, object_name, use_cstat=True, nprocs=1)

fig, axes=ba.plot_survey_lc([batsurvey_obs,mosaic_list], id_list= object_name, time_unit="UTC", values=["rate","snr", "flux", "PhoIndex", "exposure"], same_figure=True)
