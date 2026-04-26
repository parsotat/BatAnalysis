# Mosaic Low-Pcode Detection Fix

This branch captures the `batanalysis` changes that were validated while debugging weak-source losses in the BAT mosaic workflow.

## Files changed

- `batanalysis/bat_skyimage.py`
- `batanalysis/bat_skyview.py`

## Summary of changes

### 1. Allow HPX images without a WCS warning path

`BatSkyImage` now treats histograms that already contain an `HPX` axis as valid even when `wcs=None`.

Why this matters:

- intermediate mosaic healpix images do not need a detector-plane WCS
- warning or validation logic written for tangent-plane images should not be applied to HPX intermediates

### 2. Align low-partial-coding masking with the mosaic floor

`BatSkyView._parse_skyimages()` previously masked SNR and background-stddev images below a hard-coded `0.05` partial-coding threshold.

This branch changes that masking to use `batanalysis.mosaic._pcodethresh` instead.

Why this matters:

- the mosaic code is intended to work down to the configured mosaic floor
- a hard-coded 5% mask discarded information before the mosaic accumulation stage
- this made downstream `pcodethresh` settings less effective for weak sources near the edge of the coded field

### 3. Normalize mosaic pcode before applying detection thresholds

For mosaic skyviews, `pcode_img` is accumulated during image addition as `pcode * exposure`.

This branch adds `_normalized_detection_pcode_img()` and uses it in `detect_sources()` so that the detection threshold is applied to a true partial-coding fraction rather than an exposure-weighted quantity.

Why this matters:

- thresholding on `pcode * exposure` makes the effective pcode cut depend on exposure
- short integrations can be penalized even when the actual partial coding is acceptable
- weak sources in short windows can therefore be missed for the wrong reason

### 4. Use healpix-aware defaults in mosaic detection

`detect_sources()` now recognizes healpix and mosaic views explicitly and defaults their `pcodethresh` to `_pcodethresh` rather than the standard-image default of `0.05`.

In the same path, the SNR and pcode arrays are converted to raw numeric arrays before thresholding to avoid `astropy.units` comparison issues.

## Expected effect

These changes make mosaic detection more consistent with the intended low-partial-coding workflow and reduce the chance that weak sources are lost because of:

- premature masking below 5% partial coding
- exposure-dependent pcode thresholding
- unit-handling mismatches in the healpix detection path

## Validation used

The corresponding installed-package patch was validated with:

- `py_compile` on the modified files
- a synthetic mosaic regression case where a source with normalized pcode `0.02` and stored `pcode * exposure = 0.004` was correctly recovered with `pcodethresh=0.01`
- an end-to-end rerun on trigger `673969345_c0`, where the final maximum mosaic SNR increased from `5.6297` to `11.4669`

## Notes

- This branch contains only the `batanalysis` library-side changes.
- Local pipeline-side tuning done in `bat_glimpse_pipeline.py` is intentionally not included here.