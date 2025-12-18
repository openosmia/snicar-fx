"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import xarray as xr
import numpy as np


def compute_bin_average(
    ds: xr.DataArray | xr.Dataset, wavelength_bins
) -> xr.DataArray | xr.Dataset:
    """
    Bin a high-resolution xarray DataArray or Dataset along 'wavelength'
    and compute the mean per bin.
    """
    bin_edges = np.asarray(wavelength_bins)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def _bin_da(da: xr.DataArray) -> xr.DataArray:
        grouped = da.groupby_bins("wavelength", bin_edges, right=False)
        integrated = grouped.map(lambda x: x.integrate("wavelength"))
        mean_per_bin = integrated / xr.DataArray(bin_widths, dims=["wavelength_bins"])
        mean_per_bin = mean_per_bin.assign_coords(
            wavelength=("wavelength_bins", bin_centers)
        )
        return mean_per_bin

    if isinstance(ds, xr.DataArray):
        return _bin_da(ds)
    elif isinstance(ds, xr.Dataset):
        out_vars = {name: _bin_da(da) for name, da in ds.data_vars.items()}
        # take coordinates from one variable
        coords = {k: v for k, v in out_vars[next(iter(out_vars))].coords.items()}
        return xr.Dataset(out_vars, coords=coords, attrs=ds.attrs)
    else:
        raise TypeError("Input must be an xarray DataArray or Dataset")
