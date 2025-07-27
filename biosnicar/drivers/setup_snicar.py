#!/usr/bin/python

import yaml
import numpy as np
from pathlib import Path
import biosnicar
import os
from biosnicar.utils.create_files_at_given_resolution import create_resolution_dependent_files

from biosnicar.classes import (
    Ice,
    Illumination,
    Impurity,
    ModelConfig,
    RTConfig,
)


def setup_snicar(input_file):
    """Builds impurity array and instances of all classes according to config in yaml file.

    Args:
        None

    Returns:
        ice: instance of Ice class
        illumination: instance of Illumination class
        rt_config: instance of RTConfig class
        model_config: instance of ModelConfig class
        plot_config: instance of PlotConfig class
        display_config: instance of DisplayConfig class

    """

    create_resolution_dependent_files(input_file)         
     
    (
        ice,
        illumination,
        rt_config,
        model_config,
    ) = build_classes(input_file)
    
    ice.set_refractive_index()
    
    ice.set_diffuse_fresnel_coefficients()
    
    illumination.calculate_irradiance()
    
    impurities = build_impurities_array(input_file)
    
    for impurity in impurities: 
        impurity.get_impurity_properties()

    return (
        ice,
        illumination,
        rt_config,
        model_config,
        impurities,
    )

def create_resolution_dependent_files(input_file):
    
    with open(input_file, "r") as ymlfile:
        inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    wvl_start = inputs["RTM"]["WVL_START"]
    wvl_end = inputs["RTM"]["WVL_END"]
    res = inputs["RTM"]["RESOLUTION"]
    
    path_op = Path(
        (str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
                + f'/data/OP_data/{wvl_start}_{wvl_end}_{res}/'
        ))
    
    if not path_op.exists():
        # create folders 
        solar_fluxes_subfolder = path_op / "solar_fluxes"
        solar_fluxes_subfolder.mkdir(parents=True)
        laps_subfolder = path_op / "laps"
        laps_subfolder.mkdir(parents=True)
        
        
        

def build_classes(input_file):
    """Instantiates classes according to config in yaml file.

    Args:
        None

    Returns:
        ice: instance of Ice class
        illumination: instance of Illumination class
        rt_config: instance of RTConfig class
        model_config: instance of ModelConfig class
        plot_config: instance of PlotConfig class
        display_config: instance of DisplayConfig class
    """

    ice = Ice(input_file)
    illumination = Illumination(input_file)
    rt_config = RTConfig(input_file)
    model_config = ModelConfig(input_file)

    return ice, illumination, rt_config, model_config


def build_impurities_array(input_file):
    """Creates an array of instances of Impurity.

    creates an array of impurities - each one an instance of Impurity with
    properties defined in yaml file.

    Args:
        None

    Returns:
        impurities: array of instances of Impurity
    """

    with open(input_file, "r") as ymlfile:
        inputs = yaml.load(ymlfile, Loader=yaml.FullLoader)

    file_tag = inputs["RTM"]["NBR_WVL"]
    
    impurities = []

    for i, id in enumerate(inputs["IMPURITIES"]):
        name = inputs["IMPURITIES"][id]["NAME"]
        file = inputs["IMPURITIES"][id]["FILE"]
        coated = inputs["IMPURITIES"][id]["COATED"]
        unit = inputs["IMPURITIES"][id]["UNIT"]
        conc = inputs["IMPURITIES"][id]["CONC"]
        impurities.append(Impurity(file, coated, unit, name, conc, file_tag))

    return impurities

def calculate_column_ops(ice, model_config):
    
    for lyr in range(ice.nbr_lyr):
        if ice.layer_type[lyr] > 0: # ice - only air inclusions for now
            vlm_frac_ice = (ice.rho[lyr] - ice.lwc[lyr] * 1000) / 917
            vlm_frac_air = 1 - ice.lwc[lyr] - vlm_frac_ice
            eq_rds = 3 * vlm_frac_air / (ice.ssa[lyr] * ice.rho[lyr]) # Eq from Whicker
            sca_cff_vlm_air_bbl = np.ones(ice.nbr_wvl) * 2 * 0.75 / (eq_rds)
            scattering_cff = (
                sca_cff_vlm_air_bbl 
                * vlm_frac_air 
                / ice.rho[lyr]
                )

            abs_cff = (4 
                       * np.pi 
                       / (model_config.wavelengths * 1e-6) 
                       / ice.rho[lyr]
                       * (
                           vlm_frac_ice * ice.ref_idx_im
                           + ice.lwc[lyr] * ice.ref_idx_im_water
                           )
                       )
                
            ice.ext[lyr, :] = (
                scattering_cff
                + abs_cff
                )
            ice.ss_alb[lyr, :] = (
                scattering_cff 
                / ice.ext[lyr, :]
                )
            ice.g[lyr, :] = np.ones(ice.nbr_wvl) * 0.86
            
            ice.tau[lyr, :] = ice.rho[lyr] * ice.dz[lyr] * ice.ext[lyr, :]
            
        else: #snow
            ice.ext[lyr, :] = (ice.rho[lyr] * ice.ssa[lyr] / 2) / ice.rho[lyr]
            ice.tau[lyr, :] = ice.rho[lyr] * ice.dz[lyr] * ice.ext[lyr, :]
            W = 0.0611 + 0.17 * (ice.ref_idx_re - 1.3)
            k_eq = (ice.lwc[lyr] * ice.ref_idx_im_water 
                      + (1-ice.lwc[lyr]) * ice.ref_idx_im
                      )
            c = 24.0 * np.pi * k_eq / (917.0 * model_config.wavelengths * 1e-6) / ice.ssa[lyr]
            
            # change specific single scat albedo and g depending on shape
            if ice.grain_shape[lyr] == 0: 
                B0 = 1.25
                g0 = 0.895
                B = B0 + 0.4 * (ice.ref_idx_re - 1.3)
                phi = 2.0 / 3 * B / (1 - W)
                ice.ss_alb[lyr, :] = 1 - 0.5 * (1 - W) * (1 - np.exp(-c * phi))
                y = 0.728 + 0.752 * (ice.ref_idx_re - 1.3)
                ginf = 0.9751 - 0.105 * (ice.ref_idx_re - 1.3)
                g00 = g0 - 0.38 * (ice.ref_idx_re - 1.3)
                ice.g[lyr,:] = ginf - (ginf - g00) * np.exp(-y * c) 
                
            elif ice.grain_shape[lyr] == 1:
                ice.g[lyr,:] = np.ones(ice.nbr_wvl) * 0.815
                B = ice.ref_idx_re**2
                phi = 2.0 / 3 * B / (1 - W)
                ice.ss_alb[lyr, :] = 1 - 0.5 * (1 - W) * (1 - np.exp(-c * phi))


def mix_in_impurities(ice, impurities, model_config):
    """Updates optical properties for the presence of light absorbing particles.

    Takes the optical properties of the clean ice column and adjusts them for
    the presence of light absorbing particles in the ice. Each impurity is an
    instance of the Impurity class whose attributes include the path to the
    specific optical properties for that impurity. Its concentration is generally
    provided in ppb, but concentration of algae can also be given in cells/mL.


    Args:
        ice: instance of Ice class
        impurities: array containing instances of Impurity class
        model_config: instance of ModelConfig class

    Returns:
        tau: updated optical thickness
        ssa: updated single scattering albedo
        g: updated asymmetry parameter
        L_snw: mass of ice ine ach layer

    """

    ssa_aer = np.zeros([len(impurities), model_config.nbr_wvl])
    mac_aer = np.zeros([len(impurities), model_config.nbr_wvl])
    g_aer = np.zeros([len(impurities), model_config.nbr_wvl])
    mss_aer = np.zeros([ice.nbr_lyr, len(impurities)])
    g_sum = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    ssa_sum = np.zeros([ice.nbr_lyr, len(impurities), model_config.nbr_wvl])
    tau = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    ssa = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    g = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    L_aer = np.zeros([ice.nbr_lyr, len(impurities)])
    tau_aer = np.zeros([ice.nbr_lyr, len(impurities), model_config.nbr_wvl])
    tau_sum = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    ssa_sum = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    L_snw = np.zeros(ice.nbr_lyr)
    tau_snw = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])

    for i, impurity in enumerate(impurities):

        g_aer[i, :] = impurity.g
        ssa_aer[i, :] = impurity.ssa

        if impurity.unit == 1:

            mss_aer[0 : ice.nbr_lyr, i] = (
                np.array(impurity.conc) / 917 * 10**6
            ) 

        else:
            mss_aer[0 : ice.nbr_lyr, i] = (
                np.array(impurity.conc) * 1e-9
            ) 

    # for each layer, the layer mass (L) is density * layer thickness
    # for each layer the optical ice.depth is
    # the layer mass * the mass extinction coefficient
    # first for the ice in each layer

    for i in range(ice.nbr_lyr):

        L_snw[i] = ice.rho[i] * ice.dz[i]

        for j, impurity in enumerate(impurities):

            mac_aer[j, :] = impurity.mac

            # kg ice m-2 * cells kg-1 ice = cells m-2
            L_aer[i, j] = L_snw[i] * mss_aer[i, j]
            # cells m-2 * m2 cells-1

            tau_aer[i, j, :] = L_aer[i, j] * mac_aer[j, :]
            tau_sum[i, :] = tau_sum[i, :] + tau_aer[i, j, :]
            ssa_sum[i, :] = ssa_sum[i, :] + (tau_aer[i, j, :] * ssa_aer[j, :])
            g_sum[i, :] = g_sum[i, :] + (tau_aer[i, j, :] * ssa_aer[j, :] * g_aer[j, :])

            # ice mass = snow mass - impurity mass (generally tiny correction)
            # if aer == algae and L_aer is in cells m-2, should be converted
            # to m-2 kg-1 : 1 cell = 1ng = 10**(-12) kg

            if impurity.unit == 1:

                L_snw[i] = L_snw[i] - L_aer[i, j] * 10 ** (-12)

            else:
                L_snw[i] = L_snw[i] - L_aer[i, j]

        tau_snw[i, :] = L_snw[i] * ice.ext[i, :]

        # finally, for each layer calculate the effective ssa, tau and g
        # for the snow+LAP
        ice.tau[i, :] = tau_sum[i, :] + tau_snw[i, :]
        ice.ssa[i, :] = (1 / tau[i, :]) * (ssa_sum[i, :] + (ice.ss_alb[i, :] * tau_snw[i, :]))
        ice.g[i, :] = (1 / (tau[i, :] * (ssa[i, :]))) * (
            g_sum[i, :] + (ice.g[i, :] * ice.ss_alb[i, :] * tau_snw[i, :])
        )

    # just in case any unrealistic values arise (none detected so far)
    ssa[ssa <= 0] = 0.00000001
    ssa[ssa >= 1] = 0.99999999
    g[g <= 0] = 0.00001
    g[g > 0.99] = 0.99



if __name__ == "__main__":
    pass
