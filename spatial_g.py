"""
spatial_g.py

Spatial land productivity functions for the ECC model.

Loads the HCC spatial input data and computes spatially-explicit
food productivity (effective g) for use in the ECC optimizer.

Current version: constrained to currently farmed land only.
Future version: will allow expansion to unfarmed land.
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
    # and are ocean, permanent ice, or otherwise non-agricultural
    is_land = np.mean(eimat, axis=2) > 0

    return PAR, area, farmedPct, eimat, is_land


def get_kcal_per_ha(PAR, eimat, area, tau):
    """
    Computes annual food productivity per hectare for each grid cell.

    Parameters
    ----------
    PAR : (180, 360, 365) — photosynthetically active radiation
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


def compute_effective_g(kcal_per_ha, area_ha, farmedPct, h_food_share, is_land):
    """
    Ranks ALL land cells by productivity and selects the top
    h_food_share fraction. At low shares, g is high (best land).
    As share increases, g declines toward the all-land average.
    At h_food_share = 1.0, g equals the all-land average — matching
    the aspatial model's PAR scalar.
    """
    g_flat    = kcal_per_ha.ravel()
    area_flat = area_ha.ravel()
    land_flat = is_land.ravel()

    total_land_ha = float(np.nansum(area_flat * land_flat))
    target_ha     = h_food_share * total_land_ha

    order       = np.argsort(-g_flat)
    accumulated = 0.0
    weighted_g  = 0.0

    for idx in order:
        if not land_flat[idx] or area_flat[idx] <= 0:
            continue
        if accumulated >= target_ha:
            break
        take        = min(area_flat[idx], target_ha - accumulated)
        weighted_g += g_flat[idx] * take
        accumulated += take

    return weighted_g / accumulated if accumulated > 0 else 0.0


def effective_par_from_g(eff_g, tau):
    """
    Back-calculates the effective PAR scalar that would produce
    a given g in the ECC optimizer's food production function:
        g = PAR_kcal * harvest * conversion
        PAR_kcal = 239.006 * 10000 * PAR

    Used to inject spatial g into the existing optimizer without
    changing ecc_optimization.py.

    Parameters
    ----------
    eff_g : float — effective g in kcal/ha/yr
    tau : float — technology level

    Returns
    -------
    float : effective PAR in MJ/m2/yr
    """
    conversion     = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
    harvest        = 1.0 - (tau * (0.0 - 0.5) + 0.5)
    PAR_KCAL_SCALE = 239.006 * 10000
    tech_scale     = harvest * conversion
    if tech_scale > 0:
        return eff_g / (PAR_KCAL_SCALE * tech_scale)
    else:
        return 2164.5  # fall back to aspatial default at tau=0
