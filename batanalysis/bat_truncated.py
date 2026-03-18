import glob
import warnings

import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.table import Table, Column


def check_for_truncated_images(infile: str, force: bool = True, emax: float = 35.0):
    """Check if the given image file is truncated by looking for the presence of the expected extensions.

    Args:
        infile (str): Image file to check (e.g. "point_20233341529_1.img")
        force (bool): If True, forces the > 35 keV image to be treated as truncated regardless of
            the check. Default is True. That means it only selects the first three bands.
        emax (float): The maximum energy to check for in the header.
            If the E_MIN keyword is greater than this value, the image will be treated as truncated.
            Default is 35.0 keV.
    """
    good_rows = []
    bad_rows = []
    hdul = fits.open(infile)

    for ind, hdu in enumerate(hdul):
        if ("PRIMARY" in hdu.name.upper()) or ("IMAGE" in hdu.name.upper()):
            if force:
                min_en = hdu.header.get("E_MIN", 0.0)
                if min_en < emax:
                    # This is a good image, so include this
                    good_rows.append(ind + 1)
                else:
                    print(
                        f"WARNING: Forcing image {infile} to be treated as truncated at"
                        f" extension {hdu.name} (index {ind}), as the minimum energy is {min_en} keV."
                        "This extension will be skipped in source detection."
                    )
                    bad_rows.append(ind + 1)
            else:
                # Check the data
                img_data = hdu._data
                imsum = np.nansum(img_data)
                if imsum != 0:
                    # That means that this is an good image, so include this
                    good_rows.append(ind + 1)
                else:
                    print(
                        f"WARNING: Image {infile} appears to be truncated at"
                        f" extension {hdu.name} (index {ind}), as the data sum is zero."
                        "This extension will be skipped in source detection."
                    )
                    bad_rows.append(ind + 1)
    hdul.close()
    return good_rows, bad_rows


def patch_truncated_results(files: list, good_rows: list, bad_rows: list):
    """Make the necessary adjustments to the variance map and catalog files to account
    for any truncated images that were detected in the check_for_truncated_images
    step, so that the source detection can proceed without errors.

    Args:
        files (list): List of paths to the input image files that were
            checked for truncation (e.g. ["point_20233341529_1.img", "point_20233341529_2.img"])
        good_rows (list): List of the row numbers corresponding to the good images (e.g. [1, 2, 3])
        bad_rows (list): List of the row numbers corresponding to the truncated images (e.g. [4, 5, 6])
    """

    img = [f for f in files if ".img" in f]
    if len(img) == 0:
        raise ValueError(f"No image file found in the input files: {files}")
    else:
        img = img[0]
    img_hdu = fits.open(img)
    numbins = 0
    for hdu in img_hdu:
        if ("PRIMARY" in hdu.name.upper()) or ("IMAGE" in hdu.name.upper()):
            numbins += 1
    img_hdu.close()
    all_rows = np.arange(1, numbins + 1)
    _, good_row_inds, good_row_inds_old = np.intersect1d(
        all_rows, np.array(good_rows), return_indices=True
    )
    good_row_mask = np.isin(all_rows, np.array(good_rows))
    # bad_row_mask = np.isin(all_rows, np.array(bad_rows))

    # Check to see if there is a variance map and catalog file to update, and if so, update them accordingly
    var = [f for f in files if "var" in f]

    if len(var) > 0:
        var = var[0]
        if Path(var).exists():
            var_exists = True
        else:
            var_exists = False
    else:
        var_exists = False

    im_hdul = fits.open(str(img), mode="update")
    if var_exists:
        var_hdul = fits.open(str(var), mode="update")
    for ind in range(1, numbins + 1):

        if ind in bad_rows:
            im_hdul[ind - 1].header["TRUNCATED"] = (
                True,
                "This extension is truncated and should be ignored in source detection",
            )
            im_hdul[ind - 1].data = np.zeros_like(im_hdul[ind - 1].data)

            if var_exists:
                # There will not be any extensions in the variance map if the image is truncated, but we check just in case
                if ind >= len(var_hdul):
                    var_hdr = im_hdul[ind - 1].header.copy()
                    hduname = im_hdul[ind - 1].name.replace("IMAGE", "VAR")
                    var_hdr["EXTNAME"] = hduname
                    var_hdr["HDUNAME"] = hduname
                    var_data = np.zeros_like(im_hdul[ind - 1].data)
                    var_hdul.append(fits.ImageHDU(data=var_data, header=var_hdr))
        else:
            im_hdul[ind - 1].header["TRUNCATED"] = (
                False,
                "This extension is not truncated and should be included in source detection",
            )
            if var_exists:
                var_hdul[ind - 1].header["TRUNCATED"] = (
                    False,
                    "This extension is not truncated and should be included in source detection",
                )
    im_hdul.flush()
    im_hdul.close()
    if var_exists:
        var_hdul.flush()
        var_hdul.close()

    # Then update catalog
    # Remember to update these names
    cat = [f for f in files if "cat" in f]
    if len(cat) > 0:
        cat = cat[0]
    if Path(cat).exists():
        cat_hdul = fits.open(str(cat), mode="update")
        cat_data = Table(cat_hdul[1].data)

        # Remove the meta, so that it can be regenerated with the new column
        cat_data.meta = None
        for c in cat_data.colnames:
            if cat_data[c].ndim > 1:
                newdata = np.zeros((len(cat_data[c]), numbins))
                newdata[:, good_row_inds] = cat_data[c].data[:, good_row_inds_old]
                cat_data.replace_column(c, Column(data=newdata, name=c))
        try:
            cat_data.add_column(
                Column(
                    data=[(~good_row_mask).astype(bool)] * len(cat_data),
                    name="TRUNCATED",
                    dtype=bool,
                ),
            )
        except:
            cat_data.replace_column(
                "TRUNCATED",
                Column(
                    data=[(~good_row_mask).astype(bool)] * len(cat_data),
                    name="TRUNCATED",
                    dtype=bool,
                ),
            )
        cat_hdul[1] = fits.BinTableHDU(data=cat_data)  # , header=cat_hdul[1].header)
        cat_hdul.flush()
        cat_hdul.close()


def patch_truncated_obsid(obsid_dir: str):
    """BAT when operating in reduced efficiency mode, will only collect data
    in the first 20 energy bins instead of the full 80 bins. This will cause
    some of the images (high energy bands) to be all zero. This function checks
    for this condition and updated the headers accordingly.
    """
    all_truncated_masks = {}
    # This is a truncated OBSID
    pids = sorted(Path(obsid_dir).glob("point*"))
    for pid in pids:
        truncated_mask = []
        img_file = sorted(Path(pid).glob("*_2.img"))
        if len(img_file) > 0:
            img_file = img_file[0]
            img_hdu = fits.open(img_file)

            for i in range(len(img_hdu)):
                if ("PRIMARY" in img_hdu[i].name) or ("BAT_IMAGE" in img_hdu[i].name):
                    is_truncated = img_hdu[i].header.get("TRUNCATED", False)
                    truncated_mask.append(is_truncated)

            img_hdu.close()
            all_truncated_masks[pid.name] = truncated_mask

    # Then look at the stats file and change it
    res_file = Path(obsid_dir).joinpath("stats_point.fits")
    if res_file.exists():
        stats_data = Table.read(res_file, format='fits')
        masks = np.zeros_like(stats_data["CHI2"].data).astype(bool)
        names = [str(i).replace(" ", "") for i in stats_data["IMAGE_ID"]]
        for pid in list(all_truncated_masks.keys()):
            inds = np.where(np.array(names) == pid)[0]
            if len(inds) > 0:
                masks[inds] = all_truncated_masks[pid]

        is_truncated_col = fits.Column(array=masks, name="TRUNCATED_MASK", format=f"{len(masks[0])}L")
        new_cols = fits.ColDefs([is_truncated_col])

        with fits.open(res_file, mode="update") as f:
            if "TRUNCATED_MASK" not in f["STATS_POINT"].columns.names:
                #add the column to the table
                orig_table = f["STATS_POINT"].data
                orig_cols = orig_table.columns
                hdu=fits.BinTableHDU.from_columns(orig_cols + new_cols)
                hdu.name="STATS_POINT"
                new_hdu_keys=[i for i in hdu.header.keys()]

                #copy all the comments over too
                for key, comment in zip(f["STATS_POINT"].header.keys(), f["STATS_POINT"].header.comments):
                    if key in new_hdu_keys:
                        hdu.header.comments[key] = comment

                f["STATS_POINT"] = hdu

            else:
                #overwrite the column data
                f["STATS_POINT"].data['TRUNCATED_MASK'][:] = masks

            f.flush()
    else:
        warnings.warn(f"The file {res_file} doesnt seem to exist for adding information related to the truncation of DPH data.")