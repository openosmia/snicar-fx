"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import xarray as xr


def compute_bin_average(
    ds: xr.DataArray | xr.Dataset, wavelength_bins
) -> xr.DataArray | xr.Dataset:
    """
    Bin a high-resolution DataArray or Dataset along 'wavelength'
    and compute the mean per bin. This is used mainly by the "band"
    SPECTRAL_MODE.
    """

    # group into bins
    grouped = ds.groupby_bins("wavelength", wavelength_bins, right=False)

    def simple_mean(x):
        return x.integrate("wavelength") / (x.wavelength.max() - x.wavelength.min())

    resampled = grouped.apply(simple_mean)

    # rename dimension to 'wavelength'
    # resampled = resampled.rename({"wavelength_bins": "wavelength"})

    # set coordinate to bin centers
    # bin_centers = (grouped.bins.left + grouped.bins.right) / 2
    # resampled = resampled.assign_coords(wavelength=bin_centers)

    return resampled
