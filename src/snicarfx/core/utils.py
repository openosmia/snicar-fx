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


def compute_band_average(
    ds: xr.DataArray | xr.Dataset, band_ranges: np.ndarray, wavelength_dim="nwvl"
):

    band_limits = np.asarray(band_ranges)
    wl_min = band_ranges[:, 0]
    wl_max = band_ranges[:, 1]
    wvl_band = band_ranges[:, 2]

    nband = band_ranges.shape[0]

    def _bandmean_da(da: xr.DataArray) -> xr.DataArray:
        band_means = []
        for lo, hi in zip(wl_min, wl_max):
            da_sel = da.sel({wavelength_dim: slice(lo, hi)})
            band_mean = da_sel.mean(dim=wavelength_dim)
            band_means.append(band_mean)

        out = xr.concat(band_means, dim=wavelength_dim)
        out = out.assign_coords({wavelength_dim: wvl_band})

        # Preserve the original order of dimensions
        # Move the wavelength_dim to its original axis
        original_axes = list(da.dims)
        if wavelength_dim in original_axes:
            axis = original_axes.index(wavelength_dim)
            out = out.transpose(*original_axes)
        return out

    if isinstance(ds, xr.DataArray):
        return _bandmean_da(ds)

    elif isinstance(ds, xr.Dataset):
        out_vars = {}
        for name, da in ds.data_vars.items():
            if wavelength_dim in da.dims:
                out_vars[name] = _bandmean_da(da)
            else:
                out_vars[name] = da

        coords = {k: v for k, v in ds.coords.items() if k != wavelength_dim}
        coords[wavelength_dim] = wvl_band

        return xr.Dataset(out_vars, coords=coords, attrs=ds.attrs)

    else:
        raise TypeError("Input must be an xarray DataArray or Dataset")
