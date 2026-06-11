"""
spatial_g.py

Spatial land productivity functions for the ECC model.
Now includes irrigation water allocation from humanlimits.py.
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
    waterflux : (180, 360, 365) — precipitation minus PET (mm/day)
    irrigatedPct : (180, 360) — fraction of farmed land with irrigation
    is_land : (180, 360) bool — True for farmable land cells
    """
    data         = sio.loadmat(filepath)
    PAR          = data['PAR']
    area         = data['area']
    farmedPct    = data['farmedPct']
    eimat        = data['eimat']
    waterflux    = data['waterflux']
    irrigatedPct = data['irrigatedPct']

    ei_mean = np.mean(eimat, axis=2)
    is_land = (ei_mean > 0) & (farmedPct > 0)

    return PAR, area, farmedPct, eimat, waterflux, irrigatedPct, is_land


def allocate_water(waterflux, waterNeed):
    """
    Translated from allocatewater() in humanlimits.py.

    Temporally reallocates stored water surplus to water-deficit days
    within each grid cell, simulating irrigation from stored water.

    Parameters
    ----------
    waterflux : (180, 360, 365) — precipitation minus PET (mm/day)
    waterNeed : (180, 360, 365) — water needed for potential NPP (mm/day)

    Returns
    -------
    waterflux_irr : (180, 360, 365) — adjusted waterflux after irrigation
    """
    noGap    = (waterflux >= 0).astype(float)
    fluxSum  = np.sum(waterflux * noGap, axis=2)       

    growperiod   = (waterNeed > 0).astype(float)
    waterDemand  = noGap * growperiod * waterNeed
    demandSum    = np.sum(waterDemand, axis=2)

    resid = np.clip(fluxSum - demandSum, 0, None)

    latentDemand = (1 - noGap) * growperiod * np.maximum(0, waterNeed - waterflux)

    gapFrac = np.where(latentDemand > 0,
                       (1 - noGap) * growperiod * np.maximum(0, -waterflux) / (latentDemand + 1e-10),
                       0.0)

    idxs          = np.argsort(gapFrac, axis=2)
    latentDemSort = np.take_along_axis(latentDemand, idxs, axis=2)
    cumldem       = np.cumsum(latentDemSort, axis=2)

    lDemMetSort = cumldem <= resid[..., np.newaxis]

    lDemMet = np.zeros_like(lDemMetSort)
    np.put_along_axis(lDemMet, idxs, lDemMetSort, axis=2)

    demandFrac    = np.where(demandSum[..., np.newaxis] > 0,
                             waterDemand / demandSum[..., np.newaxis],
                             0.0)
    waterflux_irr = (fluxSum - resid)[..., np.newaxis] * demandFrac + lDemMet * latentDemand

    return waterflux_irr


def get_kcal_per_ha(PAR, eimat, area, tau,
                    waterflux=None, irrigatedPct=None,
                    use_irrigation=True):
    """
    Computes annual food productivity per hectare for each grid cell.
    Optionally accounts for water availability and irrigation.

    Translated from humanlimits.py lines 117-134.

    Parameters
    ----------
    PAR : (180, 360, 365) — photosynthetically active radiation
    eimat : (180, 360, 365) — light interception efficiency
    area : (180, 360) — grid cell area in m2
    tau : float — technology level in [0, 1]
    waterflux : (180, 360, 365) — precipitation minus PET (mm/day), optional
    irrigatedPct : (180, 360) — fraction of farmed land irrigated, optional
    use_irrigation : bool — whether to apply irrigation algorithm

    Returns
    -------
    kcal_per_ha : (180, 360) — food productivity in kcal/ha/yr
    area_ha : (180, 360) — grid cell area in hectares
    """
    conversion   = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
    harvest      = 1.0 - (tau * (0.0 - 0.5) + 0.5)
    potentialNPP = conversion * eimat * PAR       

    if waterflux is not None:
        waterNeed = 0.037663 * potentialNPP

        if use_irrigation and irrigatedPct is not None:
            waterflux_irr = allocate_water(waterflux, waterNeed)

            water_factor_rain = np.clip(
                waterflux / (waterNeed + 1e-10), 0, 1
            )
            actualNPP_rain = potentialNPP * water_factor_rain

            water_factor_irr = np.clip(
                waterflux_irr / (waterNeed + 1e-10), 0, 1
            )
            actualNPP_irr = potentialNPP * water_factor_irr

            irr = irrigatedPct[..., np.newaxis]  
            actualNPP = (1 - irr) * actualNPP_rain + irr * actualNPP_irr

        else:
            water_factor = np.clip(waterflux / (waterNeed + 1e-10), 0, 1)
            actualNPP    = potentialNPP * water_factor

    else:
        actualNPP = potentialNPP

    annualNPP   = np.sum(actualNPP, axis=2)       
    kcal_per_m2 = 239.006 * harvest * annualNPP   
    kcal_per_ha = kcal_per_m2 * 10000             
    area_ha     = area / 10000                    

    return kcal_per_ha, area_ha


def compute_effective_g_and_ei(kcal_per_ha, area_ha, farmedPct, 
                                h_food_share, is_land, eimat):
    """
    Returns both effective g AND production-weighted average EI
    for the selected food cells.
    """
    ei_annual = np.mean(eimat, axis=2)  
    
    g_flat    = kcal_per_ha.ravel()
    area_flat = area_ha.ravel()
    land_flat = is_land.ravel()
    ei_flat   = ei_annual.ravel()

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
        kcal = 239.006 * harvest * conversion * EI * PAR * 10000
    So:
        effective_PAR = eff_g / (239.006 * 10000 * harvest * conversion * eff_ei)
    """
    conversion     = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
    harvest        = 1.0 - (tau * (0.0 - 0.5) + 0.5)
    PAR_KCAL_SCALE = 239.006 * 10000
    tech_scale     = harvest * conversion * eff_ei  
    if tech_scale > 0:
        return eff_g / (PAR_KCAL_SCALE * tech_scale)
    else:
        return 2164.5


def effective_par_from_g(eff_g, tau):
    """
    Back-calculates the effective PAR scalar that produces a given g
    in the ECC optimizer's food production function.

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
        return 2164.5 
