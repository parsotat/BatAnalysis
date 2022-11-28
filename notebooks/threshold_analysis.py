# %%

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from copy import copy

import numpy as np

import scipy as sp
import astropy as ap
from astropy.io import fits
from astropy import units
from astropy import constants
from pathlib import Path # >=3.4
import datetime


# Copy /Users/palmer/anaconda/envs/dmplab/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc
# to ~/.matplotlib/ (osx) or ~/.config/matplotlib/  (linux) to make changes like default window size
plt.rcParams['hist.bins'] = 'auto' # 1-d histogram only
cmap_gray = copy(plt.get_cmap('gray'))
cmap_gray.set_bad('#000000ff')
cmap_lognorm = mpl.colors.LogNorm(vmin=0.5)


import batanalysis as ba
import swifttools.swift_too as swtoo
import swiftbat

ba.datadir("/tmp/batdata/", mkdir=True)

# %%
# When were we hottest with block 14?
hot14met = 677600500
hot14dt = swiftbat.met2datetime(hot14met)
obs = swtoo.ObsQuery(begin=hot14dt, length=datetime.timedelta(seconds=1))
obs
# %%
data = ba.download_swiftdata(obs, match="*.hk*")
print(data)
data = ba.download_swiftdata(obs, match=["*rate*/*.lc*", "*.dph*"])

data = ba.download_swiftdata(obs)
data
# %%
surveyfiles = []
for obs, values in data.items():
    surveyfiles.extend(list(values['obsoutdir'].glob('**/*.dph*')))
print(surveyfiles)
fits.info(surveyfiles[0])
# %%
surveyfile = surveyfiles[0]
dph = fits.getdata(surveyfile, extension='BAT_DPH')
ebounds = fits.getdata(surveyfile, extension='EBOUNDS')
print(dph['DPH_COUNTS'].shape)
dph.columns
# %%
plt.imshow(dph['DPH_COUNTS'].sum(axis=(0,-1)))
# %%
