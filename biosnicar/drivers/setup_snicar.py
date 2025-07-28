#!/usr/bin/python

import numpy as np
from biosnicar.classes import (
    ColumnProperties,
    SolarIrradiance,
    ModelConfig
)

def setup_snicar(input_file):
    """Calculate column optical properties to feed the RT solver.

    Args:
        None

    Returns:
        column: instance of ColumnProperties class
        irradiance: instance of SolarIrradiance class

    """

        
    (
        column,
        irradiance
    ) = build_classes(input_file)  
    
    calculate_column_ops_clean(column)
    
    add_laps_to_column_ops(column)
        
    return (
        column,
        irradiance
    )


def build_classes(input_file):
    """Instantiates classes according to config in yaml file.

    Args:
        input yaml file

    Returns:
        ice: instance of Ice class
        illumination: instance of Illumination class
    """
    
    model_config = ModelConfig(input_file)
    column = ColumnProperties(model_config)
    irradiance = SolarIrradiance(model_config)

    return column, irradiance


def calculate_column_ops_clean(column):
    """Calculate optical properties of a clean snow/ice column

    Args:
        column: instance of class ColumnProperties

    """
    
    for lyr in range(column.nbr_lyr):
        
        column.layer_mass[lyr] = column.density[lyr] * column.thickness[lyr]
                
        if column.layer_type[lyr] > 0: # ice - only air inclusions for now
            vlm_frac_ice = (column.density[lyr] - column.lwc[lyr] * 1000) / 917
            vlm_frac_air = 1 - column.lwc[lyr] - vlm_frac_ice
            eq_rds = 3 * vlm_frac_air / (column.ssa[lyr] * column.density[lyr]) # Eq from Whicker
            sca_cff_vlm_air_bbl = np.ones(column.nbr_wvl) * 2 * 0.75 / (eq_rds)
            scattering_cff = (
                sca_cff_vlm_air_bbl 
                * vlm_frac_air 
                / column.density[lyr]
                )

            abs_cff = (4 
                       * np.pi 
                       / (column.wavelengths) 
                       / column.density[lyr]
                       * (
                           vlm_frac_ice * column.ref_idx_im
                           + column.lwc[lyr] * column.ref_idx_im_water
                           )
                       )
            
            column.ext_cff[lyr, :] = (
                scattering_cff
                + abs_cff
                )
                
            column.ss_alb[lyr, :] = (
                scattering_cff 
                / column.ext_cff[lyr, :]
                )
            
            # column.asm_prm[lyr, :] = np.ones(column.nbr_wvl) * 0.86
            #Kokhanovsky 2002 
            column.asm_prm[lyr, :] = (0.49274 
                             + 0.44466 
                             / (0.69233 * np.sqrt(np.pi/2)) *
                             np.exp(-2 *((1/column.ref_idx_re - 1.04882) 
                                         / 0.69233)**2)
                             )
            column.asm_prm = np.clip(column.asm_prm, 0, 1)

            column.tau[lyr, :] = column.layer_mass[lyr] * column.ext_cff[lyr, :]
            
        else: #snow
            column.ext_cff[lyr, :] = (column.density[lyr] * column.ssa[lyr] / 2) / column.density[lyr]
            column.tau[lyr, :] = column.layer_mass[lyr] * column.ext_cff[lyr, :]
            W = 0.0611 + 0.17 * (column.ref_idx_re - 1.3)
            k_eq = (column.lwc[lyr] * column.ref_idx_im_water 
                      + (1-column.lwc[lyr]) * column.ref_idx_im
                      )
            c = 24.0 * np.pi * k_eq / (917.0 * column.wavelengths) / column.ssa[lyr]
            
            # change specific single scat albedo and g depending on shape
            if column.grain_shape[lyr] == 0: 
                B0 = 1.25
                g0 = 0.895
                B = B0 + 0.4 * (column.ref_idx_re - 1.3)
                phi = 2.0 / 3 * B / (1 - W)
                column.ss_alb[lyr, :] = 1 - 0.5 * (1 - W) * (1 - np.exp(-c * phi))
                y = 0.728 + 0.752 * (column.ref_idx_re - 1.3)
                ginf = 0.9751 - 0.105 * (column.ref_idx_re - 1.3)
                g00 = g0 - 0.38 * (column.ref_idx_re - 1.3)
                column.asm_prm[lyr,:] = ginf - (ginf - g00) * np.exp(-y * c) 
                
            elif column.grain_shape[lyr] == 1:
                column.asm_prm[lyr, :] = np.ones(column.nbr_wvl) * 0.815
                B = column.ref_idx_re**2
                phi = 2.0 / 3 * B / (1 - W)
                column.ss_alb[lyr, :] = 1 - 0.5 * (1 - W) * (1 - np.exp(-c * phi))
    
    
def add_laps_to_column_ops(column):
    """Calculate optical properties of a clean snow/ice column mixed with light
    absorbing particles.

    Args:
        column: instance of class ColumnProperties

    """
        
    asm_prm_lap = np.zeros([column.nbr_lyr, column.nbr_wvl])
    ss_alb_lap = np.zeros_like(asm_prm_lap)
    tau_lap = np.zeros_like(asm_prm_lap)
    lap_mass = np.zeros([column.nbr_lyr, column.nb_laps])
    
    for lyr in range(column.nbr_lyr):
        # vectorized operations to check here !! 
        lap_mass[lyr, :] = column.layer_mass[lyr] * column.lap_concentrations[lyr, :] 
        
        tau_lap[lyr, :] = np.sum(lap_mass[lyr, :][lyr, None] * column.lap_ext_cff,
                                 axis=0)
        ss_alb_lap[lyr, :] = np.sum(lap_mass[lyr, :][lyr, None] * column.lap_ext_cff 
                                 * column.lap_ss_alb, 
                                 axis=0) 
        asm_prm_lap[lyr, :] = np.sum(lap_mass[lyr, :][lyr, None] * column.lap_ext_cff 
                               * column.lap_ss_alb 
                               * column.lap_asm_prm, 
                               axis=0) 

        # calc column layer mass by removing mass of impurities and re-calc tau
        column.layer_mass[lyr] = column.layer_mass[lyr] - np.sum(lap_mass[lyr, :])
        column.tau[lyr, :] = column.layer_mass[lyr] * column.ext_cff[lyr, :]
    
        # calculate the effective ssa, tau and g
        # first create local variables for bulk only
        tau_clean = column.tau[lyr, :].copy()
        ss_alb_clean = column.ss_alb[lyr, :].copy()
        asm_prm_clean = column.asm_prm[lyr, :].copy()
        
        column.tau[lyr, :] = tau_lap[lyr, :] + tau_clean
        column.ss_alb[lyr, :] = (1 / column.tau[lyr, :]) * (ss_alb_lap[lyr, :] + (ss_alb_clean * tau_clean))
        column.asm_prm[lyr, :] = (1 / (column.tau[lyr, :] * (column.ss_alb[lyr, :]))) * (
            asm_prm_lap[lyr, :] + (asm_prm_clean * ss_alb_clean * tau_clean)
        )
            
        # just in case any unrealistic values arise (none detected so far)
        column.ss_alb[column.ss_alb <= 0] = 0.00000001
        column.ss_alb[column.ss_alb >= 1] = 0.99999999
        column.asm_prm[column.asm_prm <= 0] = 0.00001
        column.asm_prm[column.asm_prm > 0.99] = 0.99

if __name__ == "__main__":
    pass
