import xarray as xr


def match_matlab_config(column):
    """
    Ensures the refractive index and fresnel coefficients used to generate 
    snicar-fx benchmark data to test snicar-fx against SNICAR-ADv4 correspond 
    to the values used in SNICAR-ADv4.

    Parameters
    ----------
    column : ColumnProperties
        Instance of the ColumnProperties class
        
    Returns
    ----------
    column : ColumnProperties
        Updated instance of the ColumnProperties class

    """

    column.ref_idx_im = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).im_Pic16.values
    column.ref_idx_re = xr.open_dataset(
        "./tests/test_data/rfidx_ice.nc"
    ).re_Pic16.values
    column.fl_r_dif_b = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fb_ice_Pic16.values
    column.fl_r_dif_a = xr.open_dataset(
        "./tests/test_data/fl_reflection_diffuse.nc"
    ).R_dif_fa_ice_Pic16.values

    return column
