"""
spatial_g.py

Spatial land productivity functions for the ECC model.
PAR-only version (no water constraints).

Key fixes:
- effective_par_from_g now correctly accounts for EI in denominator
- EI weighted average uses daily PAR as weights (more light = more impact)
- compute_effective_g replaced by compute_effective_g_and_ei
- only one definition of effective_par_from_g (with EI)
"""

import numpy as np
import scipy.io as sio


def load_spatial_data(filepath='hcc_spatial_input.mat'):
    """
    Loads the HCC spatial input data from the .mat file.

    Returns
    -------
    PAR : (180, 360, 365) — photosynthetically active radiation (MJ/m2/day)
    area : (180, 360) — grid cell area (m2)
    farmedPct : (180, 360) — fraction of cell currently farmed
    eimat : (180, 360, 365) — light interception efficiency
    is_land : (180, 360) bool — True for land cells (excludes ocean/ice)
    """
    data      = sio.loadmat(filepath)
    PAR       = data['PAR']
    area      = data['area']
    farmedPct = data['farmedPct']
    eimat     = data['eimat']

    # land mask: cells with zero mean EI have no growing season
    is_land = np.mean(eimat, axis=2) > 0

    return PAR, area, farmedPct, eimat, is_land


def get_kcal_per_ha(PAR, eimat, area, tau):
    """
    Computes annual food productivity per hectare for each grid cell.

    Parameters
    ----------
    PAR : (180, 360, 365) — photosynthetically active radiation (MJ/m2/day)
    eimat : (180, 360, 365) — light interception efficiency
    area : (180, 360) — grid cell area in m2
    tau : float — technology level in [0, 1]

    Returns
    -------
    kcal_per_ha : (180, 360) — food productivity in kcal/ha/yr
    area_ha : (180, 360) — grid cell area in hectares
    """
    conversion   = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
    harvest      = 1.0 - (tau * (0.0 - 0.5) + 0.5)
    potentialNPP = conversion * eimat * PAR       # MJ/m2/day
    annualNPP    = np.sum(potentialNPP, axis=2)   # MJ/m2/yr
    kcal_per_m2  = 239.006 * harvest * annualNPP  # kcal/m2/yr
    kcal_per_ha  = kcal_per_m2 * 10000            # kcal/ha/yr
    area_ha      = area / 10000                   # m2 -> ha
    return kcal_per_ha, area_ha


def compute_effective_g_and_ei(kcal_per_ha, area_ha, h_food_share, is_land, PAR, eimat):
    """
    Computes production-weighted average g AND PAR-weighted average EI
    for the top h_food_share fraction of land cells ranked by productivity.

    EI is weighted by daily PAR because intercepting more light has greater
    impact when there is more light available to intercept.

    Parameters
    ----------
    kcal_per_ha : (180, 360) — food productivity per cell
    area_ha : (180, 360) — cell area in hectares
    h_food_share : float — fraction of total land allocated to food
    is_land : (180, 360) bool — land mask
    PAR : (180, 360, 365) — daily PAR for EI weighting
    eimat : (180, 360, 365) — light interception efficiency

    Returns
    -------
    eff_g : float — production-weighted average g (kcal/ha/yr)
    eff_ei : float — PAR-weighted average EI for selected cells
    """
    # PAR-weighted average EI: weight each day's EI by that day's PAR
    # sum(EI * PAR) / sum(PAR) — gives higher weight to high-light days
    par_sum      = np.sum(PAR, axis=2)                          # total annual PAR per cell
    ei_par_sum   = np.sum(eimat * PAR, axis=2)                  # PAR-weighted EI sum
    ei_par_weighted = np.where(par_sum > 0,
                               ei_par_sum / (par_sum + 1e-10),
                               0.0)                             # (180, 360)

    g_flat    = kcal_per_ha.ravel()
    area_flat = area_ha.ravel()
    land_flat = is_land.ravel()
    ei_flat   = ei_par_weighted.ravel()

    total_land_ha = float(np.nansum(area_flat * land_flat))
    target_ha     = h_food_share * total_land_ha

    order       = np.argsort(-g_flat)
    accumulated = 0.0
    weighted_g  = 0.0
    weighted_ei = 0.0

    for idx in order:
        if not land_flat[idx] or area_flat[idx] <= 0:
            continue
        if accumulated >= target_ha:
            break
        take         = min(area_flat[idx], target_ha - accumulated)
        weighted_g  += g_flat[idx] * take
        weighted_ei += ei_flat[idx] * take
        accumulated += take

    eff_g  = weighted_g  / accumulated if accumulated > 0 else 0.0
    eff_ei = weighted_ei / accumulated if accumulated > 0 else 1.0
    return eff_g, eff_ei


def effective_par_from_g(eff_g, eff_ei, tau):
    """
    Back-calculates effective PAR accounting for EI.

    The food production formula is:
        kcal/ha/yr = 239.006 * 10000 * harvest * conversion * EI * PAR
    Rearranging:
        PAR = eff_g / (239.006 * 10000 * harvest * conversion * eff_ei)

    Parameters
    ----------
    eff_g : float — effective g in kcal/ha/yr
    eff_ei : float — PAR-weighted average EI for selected cells
    tau : float — technology level

    Returns
    -------
    float : effective PAR in MJ/m2/yr
    """
    conversion     = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
    harvest        = 1.0 - (tau * (0.0 - 0.5) + 0.5)
    PAR_KCAL_SCALE = 239.006 * 10000
    tech_scale     = harvest * conversion * eff_ei
    if tech_scale > 0:
        return eff_g / (PAR_KCAL_SCALE * tech_scale)
    else:
        return 2164.5   # fall back to aspatial default
