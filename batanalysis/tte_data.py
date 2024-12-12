"""
This file holds the BAT TimeTaggedEvents class

Tyler Parsotan Jul 16 2024
"""
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy import table
from astropy.io import fits
from astropy.io.fits import getdata, getheader
from astropy.table import QTable

from .batlib import decompose_det_id


class TimeTaggedEvents(object):
    """
    This class encapsulates the event data that is obtained by the BAT instrument.

    """

    def __init__(
            self,
            time,
            detector_id,
            detx,
            dety,
            quality_flag,
            energy,
            pulse_height_amplitude,
            pulse_invariant,
            mask_weight=None,
    ):
        """
        This initalizes the TimeTaggedEvent class and allows for event data to be accessed easily.

        All attributes must be initalized and kept together here. They should all be astropy Quantity arrays with the
        units appropriately set for each quantity. This should be taken care of by the user.

        :param times: The MET times of each measured photon
        :param detector_id: The detector ID where each photon was measured
        :param detx: The detector X pixel where the photon was measured
        :param dety: The detector Y pixel where the photon was measured
        :param quality_flag: The quality flag for each measured photon
        :param energy: The gain/offset corrected energy of each measured photon
        :param pulse_height_amplitude: The pulse height amplitude of each measured photon
        :param pulse_invariant: The pulse invariant of each measured photon
        :param mask_weight: The mask weighting that may apply to each photon. Can be set to None to ignore mask weighting
        """

        self.time = time
        self.detector_id = detector_id
        self.detx = detx
        self.dety = dety
        self.quality_flag = quality_flag
        self.energy = energy
        self.pha = pulse_height_amplitude
        self.pi = pulse_invariant
        self.mask_weight = mask_weight

        # get the block/DM/sandwich/channel info
        block, dm, side, channel = decompose_det_id(self.detector_id)
        self.detector_block = block
        self.detector_dm = dm
        self.detector_sand = side
        self.detector_chan = channel

    @classmethod
    def from_file(cls, event_file):
        """
        This class method creates a TimeTaggedEvents class from the information in an unzipped event file. The file must
        be unzipped at this point since the processing of event data with heasoft tools require this, so we enforce this
        as well at this time.

        :param event_file: Path to event file that will be parsed
        :return: TimeTaggedEvents object
        """

        event_file = Path(event_file).expanduser().resolve()

        if not event_file.exists():
            raise ValueError(f"The event file passed in to be read {event_file} does not seem to exist.")

        # iteratively read in the data with units
        all_data = {}
        with fits.open(event_file) as file:
            data = file[1].data
            for i in data.columns:
                all_data[i.name] = u.Quantity(data[i.name], i.unit)

        # this column may not exist if no mask weighting has been applied
        if "MASK_WEIGHT" in all_data.keys():
            mask_weight = all_data["MASK_WEIGHT"]
        else:
            mask_weight = None

        return cls(
            all_data["TIME"],
            all_data["DET_ID"],
            all_data["DETX"],
            all_data["DETY"],
            all_data["EVENT_FLAGS"],
            all_data["ENERGY"],
            all_data["PHA"],
            all_data["PI"],
            mask_weight=mask_weight,
        )

    @classmethod
    def concatenate_event(cls, *input_event_files, output_event_file=None):
        """
        This method is meant to concatenate a set of event_files into the output_event_file.

        :param input_event_files: Path objects to the event files that will be contatenated
        :param output_event_file: None or the Path object to the event file that will contain the merged event data
        :return: the output_event_file Path object with the merged event file
        """

        # make sure that we have a set of Paths
        for i in input_event_files:
            if not isinstance(i, Path):
                raise ValueError("All event files that are passed in to be concatenated need to be Path objects.")

        # also check the output
        if output_event_file is not None:
            if not isinstance(output_event_file, Path):
                raise ValueError("The output_event_file that is passed in needs to be a Path object.")
        else:
            output_event_file = input_event_files[0].parent.joinpath("total_events.evt")

        # copy the first one to the new file, include primary, event, and GTI HDUs
        pri_hdr = getheader(input_event_files[0])
        data, hdr = getdata(input_event_files[0], ext=1, header=True)
        fits.append(output_event_file, None, pri_hdr)
        fits.append(output_event_file, data, hdr)
        data, hdr = getdata(input_event_files[0], ext=2, header=True)
        fits.append(output_event_file, data, hdr)

        # iterate over the other input files and append them. Need to collect the times to update TSTART/TSTOP/GTI
        with fits.open(output_event_file, mode="update") as hdul1:
            for i in input_event_files[1:]:
                with fits.open(i) as hdul2:
                    nrows1 = hdul1[1].data.shape[0]
                    nrows2 = hdul2[1].data.shape[0]
                    nrows = nrows1 + nrows2
                    hdu = fits.BinTableHDU.from_columns(hdul1[1].columns, nrows=nrows)
                    for colname in hdul1[1].columns.names:
                        hdu.data[colname][nrows1:] = hdul2[1].data[colname]

                    hdul1[1].data = hdu.data
                    hdul1.flush()

            # sort by time
            idx = np.argsort(hdul1[1].data["TIME"])
            hdul1[1].data = hdul1[1].data[idx]
            hdul1.flush()

            # check for duplicates defined by time and detector ID
            if len(table.unique(QTable(hdul1[1].data), keys=["TIME", "DET_ID"])) != len(hdul1[1].data["TIME"]):
                raise RuntimeError(f"There are duplicate events in the merged event file {output_event_file}.")

            # get the dt for which we have events
            start = hdul1[1].data["TIME"][0]
            stop = hdul1[1].data["TIME"][-1]
            dt = stop - start

            # get the number of events
            nevents = len(hdul1[1].data["TIME"])

            # update the time and other keywords
            for i in hdul1:
                i.header["TSTART"] = start
                i.header["TSTOP"] = stop
                i.header["EXPOSURE"] = dt
                i.header["LIVETIME"] = dt
                i.header["TELAPSE"] = dt
                i.header["ONTIME"] = dt

                if "NEVENTS" in i.header.keys():
                    i.header["NEVENTS"] = nevents

                if "EXTNAME" in i.header.keys() and "GTI" in i.header["EXTNAME"]:
                    i.data["START"] = start
                    i.data["STOP"] = stop

            hdul1.flush()

        return output_event_file
