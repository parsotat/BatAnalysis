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
plt.ion()

#set the path with the BAT survey data
newdir = Path("/Users/tparsota/Documents/CRAB_SURVEY_DATA")
ba.datadir(newdir, mkdir=True)

#query heasarc for all the data within the time period of interest and download it
object_name='Crab_Nebula_Pulsar'
queryargs = dict(time="2004-12-15 .. 2006-10-27", fields='All', resultmax=0)

#use swiftbat to create a bat source object
object_location = swiftbat.simbadlocation(object_name)
object_batsource = swiftbat.source(ra=object_location[0], dec=object_location[1], name=object_name)
table_everything = ba.from_heasarc(name=None, **queryargs)
minexposure = 1000     # cm^2 after cos adjust

#calculate the exposure with partial coding
exposures = np.array([object_batsource.exposure(ra=row['RA'], dec=row['DEC'], roll=row['ROLL_ANGLE'])[0] for row in table_everything])

#select the observations that have greater than the minimum desired exposure
table_exposed = table_everything[exposures > minexposure]
print(f"Finding everything finds {len(table_everything)} observations, of which {len(table_exposed)} have more than {minexposure:0} cm^2 coded")

#download the data
#result = ba.download_swiftdata(table_exposed)

#get a list of the fully downloaded observation IDs
obs_ids=[i for i in table_exposed['OBSID'] if result[i]['success']]

#run batsurvey in parallel with pattern maps
input_dict=dict(cleansnr=6,cleanexpr='ALWAYS_CLEAN==T')
noise_map_dir=Path("/Users/tparsota/Documents/PATTERN_MAPS/")
batsurvey_obs=ba.parallel.batsurvey_analysis(obs_ids, input_dict=input_dict, patt_noise_dir=noise_map_dir, nprocs=20)

#identify our source name based on the name in teh survey catalog
source_name=object_name

#creat the pha files and the appropriate rsp file in parallel.
#use xspec to fit each spectrum with a default powerlaw spectrum
batsurvey_obs=ba.parallel.batspectrum_analysis(batsurvey_obs, source_name, recalc=True,nprocs=14)

#plot the snapshot pointing values of rate, snr, and the fitted flux and photon index
fig, axes=ba.plot_survey_lc(batsurvey_obs, id_list=source_name, time_unit="UTC", values=["rate","snr", "flux", "PhoIndex", "exposure"], calc_lc=True)

#combine all the pointings into a single file to sort into binned fits files
outventory_file=ba.merge_outventory(batsurvey_obs)

#bin into 1 month cadence
time_bins=ba.group_outventory(outventory_file, np.timedelta64(1, "M"), end_datetime=Time("2006-10-27"))

#do the parallel construction of each mosaic for each time bin
mosaic_list, total_mosaic=ba.parallel.batmosaic_analysis(batsurvey_obs, outventory_file, time_bins, nprocs=8)

mosaic_list=ba.parallel.batspectrum_analysis(mosaic_list, source_name, recalc=True,nprocs=11)
total_mosaic=ba.parallel.batspectrum_analysis(total_mosaic, source_name, recalc=True,nprocs=1)

fig, axes=ba.plot_survey_lc(mosaic_list, id_list=source_name, time_unit="UTC", values=["rate","snr", "flux", "PhoIndex", "exposure"], calc_lc=True)


