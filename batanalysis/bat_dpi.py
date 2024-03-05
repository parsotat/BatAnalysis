"""
This file holds the BAT Detector plane image class

Tyler Parsotan Feb 21 2024
"""

try:
    import heasoftpy as hsp
except ModuleNotFoundError as err:
    # Error handling
    print(err)
