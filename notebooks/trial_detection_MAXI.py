#problem obs IDs: 00012012026, 00012172020, 00035344062, 00045604023, 00095400024, 03102102001, 03109915005, 03110367008

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
object_name='MAXI J0637-430'
queryargs = dict(time="2019-11-01 .. 2020-01-30", fields='All', resultmax=0)
object_location = swiftbat.simbadlocation(object_name)
object_batsource = swiftbat.source(ra=object_location[0], dec=object_location[1], name=object_name)
table_everything = ba.from_heasarc(name=None, **queryargs)
minexposure = 1000     # cm^2 after cos adjust
exposures = np.array([object_batsource.exposure(ra=row['RA'], dec=row['DEC'], roll=row['ROLL_ANGLE'])[0] for row in table_everything])
table_exposed = table_everything[exposures > minexposure]
print(f"Finding everything finds {len(table_everything)} observations, of which {len(table_exposed)} have more than {minexposure:0} cm^2 coded")

#result = ba.download_swiftdata(table_exposed)
obs_ids=[i for i in table_exposed['OBSID'] if result[i]['success']]
#obs_ids=[i.name for i in sorted(ba.datadir().glob("*")) if i.name.isnumeric()]

#incat=ba.create_custom_catalog(object_name,99.09830, -42.86781 ,251.51841, -20.67087)
incat=Path("./custom_catalog.cat")

input_dict=dict(cleansnr=6,cleanexpr='ALWAYS_CLEAN==T', incatalog=f"{incat}", detthresh=8000, detthresh2=8000)
noise_map_dir=Path("/local/data/bat1raid/tparsota/PATTERN_MAPS/")
batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids, input_dict=input_dict, patt_noise_dir=noise_map_dir, nprocs=10)

batsurvey_obs=ba.parallel.batspectrum_analysis(batsurvey_obs, object_name, use_cstat=True, nprocs=14)


outventory_file=ba.merge_outventory(batsurvey_obs)
time_bins=ba.group_outventory(outventory_file, np.timedelta64(1, "W"))
mosaic_list, total_mosaic=ba.parallel.batmosaic_analysis(batsurvey_obs, outventory_file, time_bins, catalog_file=incat, nprocs=3)

mosaic_list=ba.parallel.batspectrum_analysis(mosaic_list, object_name, use_cstat=True,recalc=True,nprocs=5)
total_mosaic=ba.parallel.batspectrum_analysis(total_mosaic, object_name, use_cstat=True, recalc=True,nprocs=1)

fig, axes=ba.plot_survey_lc([batsurvey_obs,mosaic_list], id_list= object_name, time_unit="UTC", values=["rate","snr", "flux", "PhoIndex", "exposure"], calc_lc=True)



