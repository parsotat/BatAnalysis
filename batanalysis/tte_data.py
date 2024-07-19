"""
This file holds the BAT TimeTaggedEvents class

Tyler Parsotan Jul 16 2024
"""
from pathlib import Path

import astropy.units as u
from astropy.io import fits

from .batlib import decompose_det_id


class TimeTaggedEvents(object):
    """
    This class encapsulates the event data that is obtained by the BAT instrument.

    TODO: add methods to add/concatenate event data, plot event data, etc
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

        return cls(
            all_data["TIME"],
            all_data["DET_ID"],
            all_data["DETX"],
            all_data["DETY"],
            all_data["EVENT_FLAGS"],
            all_data["ENERGY"],
            all_data["PHA"],
            all_data["PI"],
            mask_weight=all_data["MASK_WEIGHT"],
        )
