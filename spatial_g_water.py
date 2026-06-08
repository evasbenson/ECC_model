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

    # land mask: only cells that are currently farmed with nonzero EI
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
    # days with positive water balance (surplus days)
    noGap    = (waterflux >= 0).astype(float)
    fluxSum  = np.sum(waterflux * noGap, axis=2)       # total surplus per cell

    # water demand on growing days
    # growperiod is implicitly all days where waterNeed > 0
    growperiod   = (waterNeed > 0).astype(float)
    waterDemand  = noGap * growperiod * waterNeed
    demandSum    = np.sum(waterDemand, axis=2)

    # residual surplus after meeting demand on surplus days
    resid = np.clip(fluxSum - demandSum, 0, None)

    # latent demand: deficit on non-surplus growing days
    latentDemand = (1 - noGap) * growperiod * np.maximum(0, waterNeed - waterflux)

    # relative shortfall per day (used for sorting priority)
    gapFrac = np.where(latentDemand > 0,
                       (1 - noGap) * growperiod * np.maximum(0, -waterflux) / (latentDemand + 1e-10),
                       0.0)

    # sort days by relative shortfall (ascending) — fill easiest deficits first
    idxs          = np.argsort(gapFrac, axis=2)
    latentDemSort = np.take_along_axis(latentDemand, idxs, axis=2)
    cumldem       = np.cumsum(latentDemSort, axis=2)

    # mark days where reallocated water is sufficient to meet latent demand
    lDemMetSort = cumldem <= resid[..., np.newaxis]

    # reverse sort to restore original time ordering
    lDemMet = np.zeros_like(lDemMetSort)
    np.put_along_axis(lDemMet, idxs, lDemMetSort, axis=2)

    # reallocated waterflux
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
    potentialNPP = conversion * eimat * PAR       # MJ/m2/day

    if waterflux is not None:
        # water need per day (kg/m2/day) — from humanlimits.py line 117
        # 0.037663 kg water per MJ of potential NPP
        waterNeed = 0.037663 * potentialNPP

        if use_irrigation and irrigatedPct is not None:
            # run irrigation water allocation algorithm
            waterflux_irr = allocate_water(waterflux, waterNeed)

            # non-irrigated actual NPP: limited by raw waterflux
            water_factor_rain = np.clip(
                waterflux / (waterNeed + 1e-10), 0, 1
            )
            actualNPP_rain = potentialNPP * water_factor_rain

            # irrigated actual NPP: limited by reallocated waterflux
            water_factor_irr = np.clip(
                waterflux_irr / (waterNeed + 1e-10), 0, 1
            )
            actualNPP_irr = potentialNPP * water_factor_irr

            # blend rainfed and irrigated based on irrigatedPct
            irr = irrigatedPct[..., np.newaxis]   # broadcast over time axis
            actualNPP = (1 - irr) * actualNPP_rain + irr * actualNPP_irr

        else:
            # rainfed only
            water_factor = np.clip(waterflux / (waterNeed + 1e-10), 0, 1)
            actualNPP    = potentialNPP * water_factor

    else:
        # no water data — use potential NPP (original aspatial behavior)
        actualNPP = potentialNPP

    annualNPP   = np.sum(actualNPP, axis=2)       # MJ/m2/yr
    kcal_per_m2 = 239.006 * harvest * annualNPP   # kcal/m2/yr
    kcal_per_ha = kcal_per_m2 * 10000             # kcal/ha/yr
    area_ha     = area / 10000                    # m2 -> ha

    return kcal_per_ha, area_ha


def compute_effective_g(kcal_per_ha, area_ha, farmedPct, h_food_share, is_land):
    """
    Computes the production-weighted average food productivity (g)
    for the land allocated to food production.

    Ranks ALL land cells by productivity (best first). As h_food_share
    increases, progressively less productive cells are included,
    capturing diminishing returns to land expansion.

    Parameters
    ----------
    kcal_per_ha : (180, 360) — food productivity per cell
    area_ha : (180, 360) — cell area in hectares
    farmedPct : (180, 360) — currently farmed fraction per cell
    h_food_share : float — fraction of total land for food
    is_land : (180, 360) bool — land mask

    Returns
    -------
    float : effective g in kcal/ha/yr
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
        return 2164.5  # fall back to aspatial default at tau=0
