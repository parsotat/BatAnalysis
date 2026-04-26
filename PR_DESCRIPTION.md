# Fix Mosaic Low-Pcode Detection Handling

## Summary

This PR fixes a set of issues in the healpix/mosaic detection path that could cause weak sources to be missed even when downstream thresholds were configured to allow low partial coding.

## Changes

- align low-partial-coding masking in `BatSkyView._parse_skyimages()` with `mosaic._pcodethresh` instead of a hard-coded 5% floor
- normalize mosaic `pcode_img` by `exposure_img` before applying `pcodethresh` inside `detect_sources()`
- use healpix-aware detection defaults for mosaic views
- convert healpix SNR and pcode arrays to raw numeric arrays before threshold comparison to avoid unit-comparison failures
- allow `BatSkyImage` HPX histograms with `wcs=None` without emitting the detector-plane WCS warning path

## Why

Before this change:

- the mosaic path discarded information below 5% partial coding before the mosaic thresholding stage
- mosaic detection could apply `pcodethresh` to an exposure-weighted quantity rather than to true partial coding
- HPX intermediate images could fall into logic intended for detector-plane WCS validation

Together, these behaviors made it easier to miss weak sources in short or low-partial-coding windows.

## Validation

- `python -m py_compile batanalysis/bat_skyimage.py batanalysis/bat_skyview.py`
- synthetic regression case confirming that a mosaic source with stored `pcode * exposure = 0.004` and normalized `pcode = 0.02` is recovered with `pcodethresh = 0.01`
- end-to-end rerun in the downstream workflow where the best mosaic SNR increased from `5.6297` to `11.4669`

## Included regression tests

- HPX images without WCS no longer trigger the detector-plane warning path
- mosaic detection thresholds are applied to normalized partial coding rather than `pcode * exposure`