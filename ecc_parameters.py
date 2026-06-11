"""
ecc_parameters.py

Translated from ECC_parameters_v6.m
All parameter values match the latest MATLAB version.

Key changes from previous version:
- Added material resource parameters (M_a0, M_max, phi_share, phi_subs)
- Added material depreciation rates (d_m as vector)
- Added recycling parameters (rho, beta, zeta as vectors)
- Updated saturation parameters: c1=0.25, c2=0.4, c3=0.5
- Updated energy production shares: eta_k=0.5, eta_l=0.5
- Added capital depreciation d_k (was d previously)
"""

import numpy as np


def get_parameters():

    params = {}

    params['lfpr'] = 0.75 * 0.61       
    params['H']    = 4.9130             

    extraction_rates = np.array([1800, 140, 0.038, 16])  
    metal_total      = extraction_rates.sum()
    metal_mass       = 0.033 / metal_total * extraction_rates  

    M_a0             = 1000 * np.concatenate([[0.91], metal_mass])
    params['M_a0']   = M_a0

    waste_basket     = 1315 / (45000 + 1800 + 140 + 0.038 + 16) * np.array([45000, 1800, 140, 0.038, 16])
    extra_resources  = 1000 * np.array([191.15, 0.27235, 0.0716, 0.00000137, 0.00034])
    params['M_max']  = M_a0 + waste_basket + extra_resources

    inv_stocks           = 1.0 / M_a0
    params['phi_share']  = inv_stocks / inv_stocks.sum()  
    params['phi_subs']   = 0                               
    params['mu_c'] = 0.985   
    params['mu_f'] = 0.015   

    params['s']   = 0.26
    params['d_k'] = 0.0465   
    params['d_m'] = np.array([0.015, 0.032, 0.040, 0.100, 0.140])  

    params['gamma_k'] = 0.285   
    params['gamma_l'] = 0.625   
    params['gamma_n'] = 0.04    
    params['gamma_h'] = 0.05    

    params['alpha_k'] = 0.335   
    params['alpha_l'] = 0.625   
    params['alpha_n'] = 0.04    

    params['eta_k'] = 0.5       
    params['eta_l'] = 0.5       

    params['PAR']     = 2164.5      
    params['kcalmin'] = 1500 * 365   

    params['c1'] = 0.25   
    params['c2'] = 0.4    
    params['c3'] = 0.5    

    params['rho_max']     = 0.9999
    params['rho_current'] = np.array([0.925, 0.94, 0.93, 0.975, 0.8])

    params['beta_max']     = 0.9999
    params['beta_current'] = np.array([0.315, 0.775, 0.625, 0.575, 0.125])

    params['zeta_max']     = 0.9999
    params['zeta_current'] = np.array([0.0001, 0.1, 0.1, 0.1, 0.001])

    return params
