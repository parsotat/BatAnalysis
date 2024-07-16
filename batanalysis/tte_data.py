"""
This file holds the BAT TimeTaggedEvents class

Tyler Parsotan Jul 16 2024
"""
from .batlib import decompose_det_id


class TimeTaggedEvents(object):
    """
    This class encapsulates the event data that is obtained by the BAT instrument.

    TODO: add methods to add/concatenate event data, plot event data, etc
    """

    def __init__(
            self,
            times,
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
        This initalizes the TimeTaggedEvent class and allows for the data to be accessed easily.

        :param times:
        :param detector_id:
        :param detx:
        :param dety:
        :param quality_flag:
        :param energy:
        :param pulse_height_amplitude:
        :param pulse_invariant:
        :param mask_weight:
        """

        self.time = times
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
