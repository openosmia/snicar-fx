#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Lou-Anne Chevrollier, University of Aarhus

This code generates the refractive indices of ice & water as well as the
diffuse Fresnel coefficients at high resolution (1nm), which are then indexed
in the snicar-fx model at a user-defined spectral range.


"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import fixed_quad
import os 
from pathlib import Path
import biosnicar

def create_resolution_dependent_files(input_file):

    wl_start = 200
    wl_end = 5000
    res = 1
    
    path_op = Path(
        (str(os.path.dirname(os.path.dirname(biosnicar.__file__)))
                + '/data/OP_data/'
        ))
    path_to_raw_data = (str(
        os.path.dirname(os.path.dirname(biosnicar.__file__)))
                + '/data/additional_data/'
        )
            

    # generate ref_idx & fresnel files
    generate_refidx_fnl_file(wl_start,
                         wl_end,
                         res,
                         path_to_raw_data,
                         str(path_op) 
                         )
        

def generate_refidx_fnl_file(wl_start,
                         wl_end,
                         res,
                         path_to_raw_data,
                         savepath):
    
    ###########################################################################
    ####################### RIDX CALCULATIONS #################################
    ###########################################################################
    
    wvl_interp_grid = np.arange(wl_start,
                                wl_end, 1) # high res to properly merge raw data
    wvl_new_grid = np.arange(wl_start,
                             wl_end, 
                             res)
    nbr_wl = len(wvl_new_grid)
    path_to_rfidx = path_to_raw_data + 'refractive_indices'

    # Warren & Brandt 2008 refidx 
    data_warren08 = pd.read_csv(f'{path_to_rfidx}/Warren-2008.csv')
    
    # find where k values are inserted
    k_index = np.where(
        [data_warren08[col].str.contains("k", na=False) 
         for col in data_warren08]
        )[1][0]
    
    # get only values for n 
    n_warren08 = data_warren08.iloc[:k_index, 1].astype('float').values 
    
    # get only values for k
    k_warren08 = data_warren08.iloc[k_index+1:, 1].astype('float').values 
    
    wvl_warren08 = (data_warren08.wl[:k_index].astype('float').values * 1000)
    
    # regrid on wl1:wl2:res by interpolation in log space following recommendations  
    # in https://atmos.uw.edu/ice_optical_constants/
    
    warren08_rfidx = pd.DataFrame(
        {'n' : np.interp(np.log(wvl_interp_grid), 
                      np.log(wvl_warren08), 
                      n_warren08
                      ),
         'k' : np.exp(
             np.interp(np.log(wvl_interp_grid), 
                       np.log(wvl_warren08), 
                       np.log(k_warren08)
                       )
             )
         },
        index = wvl_interp_grid
        )
    
    # Warren & Brandt 2008 + Pic16 refidx merge
    
    data_pic16 = pd.read_csv(
        f'{path_to_rfidx}/ice_refractive_picard2016.dat').iloc[:,0]
    
    
    wvl_pic16 = [int(float(str(d).split(' ')[0])) for d in data_pic16]
    
    k_pic16 = [float(str(d).split(' ')[1]) for d in data_pic16]
    
    k_pic16_interp = np.exp(
        np.interp(np.log(np.arange(wvl_pic16[0], wvl_pic16[-1]+1, 1)), 
                  np.log(wvl_pic16), 
                  np.log(k_pic16)
                  )
        )
    
    pic16_rfidx = warren08_rfidx.k.copy()
    
    pic16_rfidx.loc[wvl_pic16[0]:wvl_pic16[-1]] = k_pic16_interp
    
    pic16_rfidx.loc[:319] = pic16_rfidx.loc[320] 
    
    
    # Warren & Brandt 2008 + Coop21 refidx merge 
    
    data_coop21 = pd.read_excel(
        f'{path_to_rfidx}/tc-15-1931-2021-t01.xlsx', 
        header=[0, 1, 2],
        index_col=0)
    
    k_coop21 = (
        data_coop21.iloc[:,-2] * (data_coop21.index * 1e-9) 
        / (4 * np.pi)
        )
    
    k_coop21_interp = np.exp(
        np.interp(np.log(np.arange(k_coop21.index[0], k_coop21.index[-1]+1, 1)), 
                  np.log(k_coop21.index), 
                  np.log(k_coop21.values)
                  )
        )
    
    cooper21_rfidx = warren08_rfidx.k.copy()
    
    cooper21_rfidx.loc[
        data_coop21.index[0]:data_coop21.index[-1]
        ] = k_coop21_interp
    
    cooper21_rfidx.loc[:349] = cooper21_rfidx.loc[350] 
    
    
    # Segelstein 1981 refidx 
    
    data_seg81 = pd.read_csv(f'{path_to_rfidx}/Segelstein.csv')
    
    # find where k values are inserted
    k_index = np.where(
        [data_seg81[col].str.contains("k", na=False) 
         for col in data_seg81]
        )[1][0]
    
    # get only values for n 
    n_seg81 = data_seg81.iloc[:k_index, 1].astype('float').values 
    
    # get only values for k
    k_seg81 = data_seg81.iloc[k_index+1:, 1].astype('float').values 
    
    wvl_seg81 = (data_seg81.wl[:k_index].astype('float').values * 1000)
    
    seg81_rfidx = pd.DataFrame(
        {'n' : np.interp(
            wvl_interp_grid,
            wvl_seg81, 
            n_seg81),
         'k' : np.interp(
             wvl_interp_grid,
             wvl_seg81, 
             k_seg81)
         }, 
        index = wvl_interp_grid
        )
    
    # Rowe 2020 refidx (from 702nm)
    data_row20 = pd.read_csv(f'{path_to_rfidx}/Rowe-273K.csv')
    
    # find where k values are inserted
    k_index = np.where(
        [data_row20[col].str.contains("k", na=False) 
         for col in data_row20]
        )[1][0]
    
    # get only values for n (! index 775 to get values from 702nm only)
    n_row20 = data_row20.iloc[:k_index, 1].astype('float').values 
    
    # get only values for k (! index 775 to get values from 702nm only)
    k_row20 = data_row20.iloc[k_index+1:, 1].astype('float').values 
    
    wvl_row20 = (data_row20.wl[:k_index].astype('float').values * 1000)
    
    # interp between 702 to wl2
    k_row20_interp = np.exp(
        np.interp(np.log(np.arange(702, wl_end, 1)), 
                  np.log(wvl_row20[775:]), 
                  np.log(k_row20[775:])
                  )
        )
    # interp between 827 to wl2
    n_row20_interp = np.exp(
        np.interp(np.log(np.arange(827, wl_end, 1)), 
                  np.log(wvl_row20[2000:]), 
                  np.log(n_row20[2000:])
                  )
        )
    
    row20_rfidx = seg81_rfidx.copy()
    
    row20_rfidx.loc[702:, "k"] = k_row20_interp
    row20_rfidx.loc[827:, "n"] = n_row20_interp
    
    r_idx = xr.Dataset(
        data_vars=dict(
            re_Wrn08=(["wvl"], warren08_rfidx.n.loc[wl_start:wl_end:res].values),
            im_Wrn08=(["wvl"], warren08_rfidx.k.loc[wl_start:wl_end:res].values),
            
            re_Coop21=(["wvl"], warren08_rfidx.n.loc[wl_start:wl_end:res].values),
            im_Coop21=(["wvl"], cooper21_rfidx.loc[wl_start:wl_end:res].values),  
                    
            re_Pic16=(["wvl"], warren08_rfidx.n.loc[wl_start:wl_end:res].values),
            im_Pic16=(["wvl"], pic16_rfidx.loc[wl_start:wl_end:res].values),
            
            # im_Wrn84=(["wvl"], np.zeros(len(wvl_grid))),
            # re_Wrn84=(["wvl"], np.zeros(len(wvl_grid))),
            
            re_Seg81=(["wvl"], seg81_rfidx.n.loc[wl_start:wl_end:res].values),
            im_Seg81=(["wvl"], seg81_rfidx.k.loc[wl_start:wl_end:res].values),
        
            re_Row20=(["wvl"], row20_rfidx.n.loc[wl_start:wl_end:res].values),
            im_Row20=(["wvl"], row20_rfidx.k.loc[wl_start:wl_end:res].values),
            
        ),
        coords=dict(wvl = wvl_new_grid*1e-9),
        attrs=dict(
            description=('Refractive index of ice (Warren and Brandt 2008,'
                         +' Picard 2016 et al. 2016, Cooper et al. 2021) and '
                         +'water (Rowe et al. 2020, Segelstein 1981).'
                         ),
        ),
    )
    
    # save file
    r_idx.to_netcdf(
        savepath
        + '/refractive_indices.nc'
        )
    
    ###########################################################################
    ####################### FRESNEL CALCULATIONS ##############################
    ###########################################################################
    
    def Rf(mu, re1, im1, re2, im2):
        '''
        This function calculate the Fresnel reflectivity from the refractive
        indices of two media and the angle of incidence, accounting for TIR.
        The critical angle is calculated from the complex RI, while the 
        Fresnel coefficients are calculated with the adjusted RI from
        Liou et al. 2002, following the formalism of Whicker et al. 2022.
    
        Parameters
        ----------
        mu : array or list
            nodes/points of the gaussian integration (in cos(theta))
        re1 : float
            real part of the relative refractive index of medium 1 (incident)
        im1 : float
            imaginary part of the relative refractive index of medium 1 
        re2 : float
            real part of the relative refractive index of medium 2 (transm.)
        im2 : float
            imaginary part of the relative refractive index of medium 2
        use_scipy: boolean
            
    
    
        Returns
        -------
        result : array or list
            Fresnel reflectivity coefficient (* mu if scipy used)
    
        '''
    
        # 1 - calculate values of theta where TIR occurs using complex ref index
        theta_c = np.arcsin((re2 - 1j*im2) / (re1 - 1j*im1)) # critical angle 
        mask = (np.arccos(mu) >= theta_c) # mask where TIR occurs
    
        
        if im1 == 0: # in this case air is above, rfix of ice is _2
            temp1 = (
                re2**2
                - im2**2
                + np.sin(np.arccos(mu)) ** 2
            )
            temp2 = (
                re2**2
                - im2**2
                - np.sin(np.arccos(mu)) ** 2
            )
            nr = (np.sqrt(2) / 2) * (
                temp1 + (temp2**2 + 4 * (re2**2) * (im2**2)) ** 0.5
            ) ** 0.5
                
            # angle of transmitted radiation
            mu0n = np.cos(
                    np.arcsin(
                        np.sin(np.arccos(mu)) 
                        / nr
                        )
                    )
                
        else: # in this case air is below, rfix of ice is _1
            temp1 = (
                re1**2
                - im1**2
                + np.sin(np.arccos(mu)) ** 2
            )
            temp2 = (
                re1**2
                - im1**2
                - np.sin(np.arccos(mu)) ** 2
            )
            nr = (np.sqrt(2) / 2) * (
                temp1 + (temp2**2 + 4 * (re1**2) * (im1**2)) ** 0.5
            ) ** 0.5
                
                
            # angle of transmitted radiation
            # first clip the few values above 1 due to mu close to 0
            temp = np.clip(np.sin(np.arccos(mu))* nr, 0, 1)
            mu0n = np.cos(
                    np.arcsin(
                        temp
                        )
                    )
    
        # 3 - calculate reflectivity using Fresnel equations 
        # Eq. 22  Briegleb & Light 2007 or from Liou 2002
        R1 = ((mu - nr * mu0n) / (
            mu + nr * mu0n
        ))  # reflection amplitude factor for perpendicular polarization
        R2 = ((nr * mu - mu0n) / (
            nr * mu + mu0n
        ))  # reflection amplitude factor for parallel polarization
        Rf = (0.5 * (R1**2 + R2**2)) 
        
        # 4 - mask reflectivity where TIR occurs
        Rf[mask] = 1  

        return Rf*mu
        
    ############################## INTEGRATE #####################################
    integral_above_Wrn08 = np.zeros(nbr_wl)
    integral_above_Coop21 = np.zeros(nbr_wl)
    integral_above_Pic16 = np.zeros(nbr_wl)
    
    integral_below_Wrn08 = np.zeros(nbr_wl)
    integral_below_Coop21 = np.zeros(nbr_wl)
    integral_below_Pic16 = np.zeros(nbr_wl)
    
    # denominator of the integral is simply an integration of cos(theta)
    normalization = fixed_quad(lambda x: x, 0, 1, n=10000)[0]
    
    for i in range(nbr_wl):
        integral_above_Wrn08[i] = fixed_quad(Rf, 
                                             0, 1, 
                                             args = (1,
                                                     0,
                                                     r_idx.re_Wrn08[i].values, 
                                                     r_idx.im_Wrn08[i].values),
                                             n=10000)[0]
        integral_below_Wrn08[i] = fixed_quad(Rf, 
                                              0, 1, 
                                              args = (r_idx.re_Wrn08[i].values, 
                                                      r_idx.im_Wrn08[i].values,
                                                      1,
                                                      0),
                                              n=10000)[0]
        integral_above_Pic16[i] = fixed_quad(Rf, 
                                             0, 1, 
                                             args = (1,
                                                     0,
                                                     r_idx.re_Pic16[i].values, 
                                                     r_idx.im_Pic16[i].values),
                                             n=10000)[0]
        integral_below_Pic16[i] = fixed_quad(Rf, 
                                             0, 1, 
                                             args = (r_idx.re_Pic16[i].values, 
                                                     r_idx.im_Pic16[i].values,
                                                     1,
                                                     0),
                                             n=10000)[0]
        integral_above_Coop21[i] = fixed_quad(Rf, 
                                             0, 1, 
                                             args = (1,
                                                     0,
                                                     r_idx.re_Coop21[i].values, 
                                                     r_idx.im_Coop21[i].values),
                                             n=10000)[0]
        integral_below_Coop21[i] = fixed_quad(Rf, 
                                             0, 1, 
                                             args = (r_idx.re_Coop21[i].values, 
                                                     r_idx.im_Coop21[i].values,
                                                     1,
                                                     0),
                                             n=10000)[0]

    # value should be 0.455 for below and 0.063 for above when n=1.31
    # we have 0.454 because the normalization is 0.5 not 0.499
    # find index where to evaluate
    idx_BL2007 = np.argmin(np.abs(r_idx.re_Wrn08.values - 1.31))
    
    print('****Reference value B&L2007 for above: 0.063 \n'
          + f"****Computed value: {(integral_above_Wrn08 / normalization)[idx_BL2007]} \n")
    print('****Reference value B&L2007 for below: 0.455 \n'
          + f"****Computed value: {(integral_below_Wrn08 / normalization)[idx_BL2007]} \n"
    )
    
    fnl_coeffs = xr.Dataset(
        data_vars=dict(
            R_dif_fa_ice_Wrn08=(["wvl"], integral_above_Wrn08 / normalization),
            R_dif_fb_ice_Wrn08=(["wvl"], integral_below_Wrn08 / normalization),
            
            R_dif_fa_ice_Coop21=(["wvl"], integral_above_Coop21 / normalization),
            R_dif_fb_ice_Coop21=(["wvl"], integral_below_Coop21 / normalization),
            
            R_dif_fa_ice_Pic16=(["wvl"], integral_above_Pic16 / normalization),
            R_dif_fb_ice_Pic16=(["wvl"], integral_below_Coop21 / normalization),
        ),
        coords=dict(wvl = wvl_new_grid*1e-9),
        attrs=dict(
            description=('Fresnel reflectivity coefficients for radiation coming from above (fa) and below (fb)'
                         ),
        ),
    )
    
    # save file
    fnl_coeffs.to_netcdf(
        savepath  
        + '/fresnel_diffuse_coefficients.nc'
        )
