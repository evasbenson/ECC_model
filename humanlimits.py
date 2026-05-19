"""
humanlimits.py

Human carrying capacity model.

Estimates the maximum global human population supportable given:
  - Geospatial climate inputs (PAR, water flux)
  - Agricultural configuration (farmed fraction, irrigation, growing season)
  - Technology parameters (crop efficiency, solar PV, fusion energy, etc.)
  - Biological parameters (longevity, activity level, BMI)

Returns a ModelOutput dataclass with:
  - CarryingCapacity : float — estimated maximum sustainable population
  - GlobalLimit      : int   — index of the binding resource constraint (0=calories, 1-9=nutrients, 10=water)
  - LocalLimits      : ndarray — per-cell fraction of forgone NPP due to water scarcity
  - Harvest          : ndarray — harvestable kcal per grid cell
"""

import numpy as np
from dataclasses import dataclass
from typing import Union

ArrayLike = Union[np.ndarray, float, int]


@dataclass
class ModelOutput:
    CarryingCapacity: float
    GlobalLimit: int
    LocalLimits: np.ndarray
    Harvest: np.ndarray


def humanlimits(
    PAR: ArrayLike,
    waterflux: ArrayLike,
    farmed: ArrayLike,
    irrigated: ArrayLike,
    growperiod: ArrayLike,
    eiArray: ArrayLike,
    area: ArrayLike,
    synthetic: bool,
    lightmove: bool,
    fusion: bool,
    contech: float,
    longevity: int,
    activity: float,
    bmi: float,
) -> ModelOutput:
    """
    Parameters
    ----------
    PAR         : 3-D array — photosynthetically active radiation (MJ/m²/day) per grid cell per period
    waterflux   : 3-D array — precipitation minus PET (mm or kg/m²) per grid cell per period
    farmed      : 2-D array — fraction of each grid cell under cultivation (or scalar 1)
    irrigated   : 2-D array — fraction of cultivated land that is irrigated (or scalar 1)
    growperiod  : 3-D binary array — 1 if within growing season, else 0 (or scalar 1)
    eiArray     : 3-D array — fraction of PAR intercepted by crop leaves per grid cell per period
    area        : 2-D array — area of each grid cell (m²)
    synthetic   : bool — True if synthetic plants capture the 700-1075 nm spectrum
    lightmove   : bool — True if excess PAR can be stored/transferred spatially
    fusion      : bool — True if nuclear fusion is a viable energy source
    contech     : float in [0,1] — technology level (0=today, 1=theoretical maximum)
    longevity   : int — deterministic human lifetime (years)
    activity    : float — multiplier on BMR for daily activity level
    bmi         : float — body mass index (normal: 18.5–25)
    """

    # ------------------------------------------------------------------
    # Resource limits (kg) — maximum availability of biologically
    # essential elements in Earth's cycles
    # ------------------------------------------------------------------
    resource_limits = np.array([
        1e20,                        # 0  C  — total carbon cycle stocks
        8.97e22,                     # 1  Ca
        2.48e23,                     # 2  Cl
        1.256e23,                    # 3  K
        7.714e23,                    # 4  Mg
        1.35e14 + 3.971e18,          # 5  N  — terrestrial + atmospheric
        1.55e23,                     # 6  Na
        5.92e21,                     # 7  P
        1.73e23,                     # 8  S
        10665e15,                    # 9  H₂O — liquid freshwater (aquifers, lakes, etc.)
    ])

    # ------------------------------------------------------------------
    # Technology parameters
    # ------------------------------------------------------------------
    tech_params = _set_params(contech, resource_limits)
    conversion  = tech_params[0]   # PAR → biomass conversion efficiency
    harvest     = 1.0 - tech_params[1]  # harvestable fraction of NPP
    spve        = tech_params[2]   # solar PV efficiency
    resource_limits[7] = tech_params[3] * resource_limits[7]  # accessible P
    LMA         = tech_params[4]   # leaf mass per unit area (kg/m²)
    storage     = tech_params[5]   # battery/energy storage efficiency
    eefp        = tech_params[6]   # production energy per unit food energy

    # Solar energy required for production per unit food energy produced
    # (zero if fusion provides energy)
    eip = (1.0 - float(fusion)) / spve * eefp

    # Effective light interception efficiency (boosted during growing season)
    ei = contech * (1.0 - eiArray) * growperiod + eiArray

    # ------------------------------------------------------------------
    # Net Primary Production (NPP)
    # ------------------------------------------------------------------
    # Available PAR after accounting for energy diverted to agriculture
    # Denominator reduces PAR by fraction needed for production
    availPAR = (
        (1.0 + float(synthetic))
        / (1.0 + 0.487 * (1.0 + float(synthetic)) * conversion * eip * harvest * ei)
        * PAR
    )

    # Potential NPP (MJ/m²/day), limited by light interception
    potentialNPP = conversion * ei * availPAR

    # Water requirement (kg/m²/day) — derived from photosynthesis stoichiometry
    # 6 mol H₂O per mol glucose; 3.7663e-2 kg per MJ
    waterNeed = 0.037663 * potentialNPP

    # Track water surplus for potential reallocation via irrigation
    waterDiff   = waterflux - growperiod * waterNeed
    excessWater = farmed * (1.0 - irrigated) * area * waterDiff * growperiod  # broadcast over time axis

    # Allocate stored/irrigated water to most productive days
    waterflux_irr, leftover = _allocate_water(
        waterflux, growperiod, waterNeed, farmed, irrigated, area
    )

    # Actual NPP with and without irrigation
    actualNPP    = growperiod * potentialNPP * np.clip(waterflux     / waterNeed, 0, 1)
    actualNPPirr = growperiod * potentialNPP * np.clip(waterflux_irr / waterNeed, 0, 1)

    # Optional: spatially redistribute excess PAR to water-limited locations
    if lightmove:
        addedNPP, addedNPPirr = _lightmove_fcn(
            potentialNPP, actualNPP, actualNPPirr,
            farmed, irrigated, area, excessWater, leftover, storage
        )
    else:
        addedNPP    = 0.0
        addedNPPirr = 0.0

    # ------------------------------------------------------------------
    # Convert NPP to harvestable kcal
    # ------------------------------------------------------------------
    storedEnergyNonIrr = np.sum(actualNPP,    axis=2) * farmed * (1.0 - irrigated) * area + np.sum(addedNPP,    axis=2) if lightmove else np.sum(actualNPP, axis=2) * farmed * (1.0 - irrigated) * area
    storedEnergyIrr    = np.sum(actualNPPirr, axis=2) * farmed * irrigated          * area + addedNPPirr if lightmove else np.sum(actualNPPirr, axis=2) * farmed * irrigated * area
    storedEnergy       = storedEnergyNonIrr + storedEnergyIrr

    kcalNPP       = 239.006 * storedEnergy          # 1 MJ = 239.006 kcal
    kcalHarvested = harvest * kcalNPP               # edible fraction
    kcalTotal     = np.sum(kcalHarvested)

    # ------------------------------------------------------------------
    # Per-capita human and plant nutrient needs
    # ------------------------------------------------------------------
    human_needs = _get_human_needs(longevity, bmi, activity)
    plant_needs = _get_plant_needs(LMA, farmed, area)

    # Resources available to humans = total stocks minus what plants embody
    # Index 0 = calories; indices 1–10 = elemental stocks
    resources_for_humans = np.concatenate([[kcalTotal], resource_limits]) - np.concatenate([[0], plant_needs])

    # ------------------------------------------------------------------
    # Liebig's Law of the Minimum → carrying capacity & binding constraint
    # ------------------------------------------------------------------
    ratios      = resources_for_humans / human_needs
    global_limit = int(np.argmin(ratios))
    CC           = ratios[global_limit]

    # ------------------------------------------------------------------
    # Local limits: fraction of forgone NPP attributable to water scarcity
    # ------------------------------------------------------------------
    local_limits = _find_local(
        waterflux, waterflux_irr, waterNeed, growperiod,
        farmed, irrigated, area, excessWater, leftover,
        addedNPP, addedNPPirr, lightmove
    )

    return ModelOutput(
        CarryingCapacity=CC,
        GlobalLimit=global_limit,
        LocalLimits=local_limits,
        Harvest=kcalHarvested,
    )


# ===========================================================================
# Helper functions (translated from MATLAB nested functions)
# ===========================================================================

def _set_params(contech: float, resource_limits: np.ndarray) -> np.ndarray:
    """
    Map the 0–1 technology index to concrete model parameter values by
    interpolating linearly between current and theoretical-best values.
    """
    best    = np.array([0.123, 0.0,   0.868, 1.0,        0.018, 1.0, 0.000156])
    current = np.array([
        best[0] / 2.5,                         # conversion efficiency
        0.5,                                   # residual biomass fraction
        0.445,                                 # solar PV efficiency
        2.18e13 / resource_limits[7],          # accessible P fraction
        0.072,                                 # LMA (kg/m²)
        0.9,                                   # battery storage efficiency
        0.112,                                 # production energy per food energy
    ])
    return contech * (best - current) + current


def _get_human_needs(longevity: int, bmi: float, activity: float) -> np.ndarray:
    """
    Per-capita annual caloric need and elemental stock requirements.

    Uses the Preece–Baines (1978) growth model for height-at-age, and the
    Mifflin–St Jeor equation for basal metabolic rate.
    """
    age = np.arange(1, longevity + 1, dtype=float)

    # Height (cm) — Preece–Baines Model 1, sex-stratified
    height_f = 163.3  - 2 * (163.3  - 151.7)   / (np.exp(0.9261  * (age - 12.07))     + np.exp(0.1216  * (age - 12.07)))
    height_m = 176.97 - 2 * (176.97 - 163.28)  / (np.exp(0.9034  * (age - 13.938))    + np.exp(0.1007  * (age - 13.938)))

    # Weight (kg)
    weight_f = bmi * (height_f / 100.0) ** 2
    weight_m = bmi * (height_m / 100.0) ** 2

    # BMR (kcal/year) — Mifflin–St Jeor equation
    bmr_f = 365 * (9.99 * weight_f + 6.25 * height_f - 4.92 * age - 161)
    bmr_m = 365 * (9.99 * weight_m + 6.25 * height_m - 4.92 * age + 5)
    bmr   = 0.5 * bmr_f + 0.5 * bmr_m

    mean_weight = np.mean(0.5 * weight_f + 0.5 * weight_m)
    req_kcal    = activity * np.mean(bmr)

    # Elemental stock requirements as fractions of body mass
    req_stock = np.array([
        0.18   * mean_weight,   # C
        0.015  * mean_weight,   # Ca
        0.0015 * mean_weight,   # Cl
        0.004  * mean_weight,   # K
        0.001  * mean_weight,   # Mg
        0.03   * mean_weight,   # N
        0.0015 * mean_weight,   # Na
        0.01   * mean_weight,   # P
        0.0025 * mean_weight,   # S
        0.9 * 0.72 * mean_weight,  # H₂O (10% body fat assumed; 72% water in lean mass)
    ])

    return np.concatenate([[req_kcal], req_stock])


def _get_plant_needs(LMA: float, farmed: ArrayLike, area: ArrayLike) -> np.ndarray:
    """
    Total elemental stocks embodied in the standing crop of leaves
    (unavailable to humans). Based on Redfield ratios and Knecht & Göransson (2004).
    """
    plant_principle = np.sum(LMA * farmed * area)   # kg dry leaf mass

    C  = 0.45   * plant_principle
    N  = 0.1761849057 * C
    Ca = 0.083  * N
    K  = 0.683  * N
    Mg = 0.087  * N
    P  = 0.024350441 * C
    S  = 0.05   * N
    H2O = 3.0   * plant_principle   # 75% moisture content

    # Order must match resource_limits: C, Ca, Cl, K, Mg, N, Na, P, S, H2O
    plant_stock = np.array([C, Ca, 0.0, K, Mg, N, 0.0, P, S, H2O])
    return plant_stock


def _allocate_water(
    waterflux, growperiod, waterNeed, farmed, irrigated, area
):
    """
    Temporally reallocates stored/irrigated water within each grid cell,
    prioritising days with the smallest relative water deficit.

    Returns
    -------
    waterflux_irr : adjusted waterflux after irrigation reallocation
    leftover      : water surplus not productively reallocated (absolute kg)
    """
    noGap     = (waterflux >= 0).astype(float)
    fluxSum   = np.sum(waterflux * noGap, axis=2)          # total positive flux per cell

    waterDemand = noGap * growperiod * waterNeed
    demandSum   = np.sum(waterDemand, axis=2)

    resid = fluxSum - demandSum
    resid = np.clip(resid, 0, None)

    latentDemand = (~noGap.astype(bool)).astype(float) * growperiod * (waterNeed - waterflux)
    # Relative shortfall for each day
    gapFrac = np.where(latentDemand > 0,
                       (~noGap.astype(bool)).astype(float) * growperiod * (-waterflux) / latentDemand,
                       0.0)

    # Sort days by relative shortfall (ascending) per cell
    idxs = np.argsort(gapFrac, axis=2)

    # Gather latent demand in sorted order
    latentDemSort = np.take_along_axis(latentDemand, idxs, axis=2)
    cumldem       = np.cumsum(latentDemSort, axis=2)

    # Mark days where reallocated water is sufficient
    lDemMetSort = cumldem <= resid[..., np.newaxis]

    # Reverse the sort to restore original time ordering
    lDemMet = np.zeros_like(lDemMetSort)
    np.put_along_axis(lDemMet, idxs, lDemMetSort, axis=2)

    # Reallocated waterflux
    demandFrac = np.where(demandSum[..., np.newaxis] > 0,
                          waterDemand / demandSum[..., np.newaxis],
                          0.0)
    waterflux_irr = (fluxSum - resid)[..., np.newaxis] * demandFrac + lDemMet * latentDemand

    leftover = resid - np.sum(lDemMet * latentDemand, axis=2)
    leftover = leftover * area * farmed * irrigated   # absolute units, irrigated areas only

    return waterflux_irr, leftover


def _lightmove_fcn(
    potentialNPP, actualNPP, actualNPPirr,
    farmed, irrigated, area, excessWater, leftover, storage
):
    """
    Spatially redistributes excess stored PAR to locations where water
    is available but light is the limiting factor, thereby increasing yield.
    """
    # NPP gain achievable if water-limited areas received more light
    hypotheticalNPP    = area * farmed * (1.0 - irrigated) * (potentialNPP - actualNPP)
    hypotheticalNPPirr = area * farmed * irrigated          * (potentialNPP - actualNPPirr)

    hypotheticalSum    = np.sum(hypotheticalNPP)
    hypotheticalSumIrr = np.sum(hypotheticalNPPirr)

    excessSum    = np.sum(excessWater)
    excessSumIrr = np.sum(leftover)

    # Spatial allocation fractions (proportional to excess water availability)
    excessFrac    = np.where(excessSum    > 0, excessWater / excessSum,    0.0)
    excessFracIrr = np.where(excessSumIrr > 0, leftover    / excessSumIrr, 0.0)

    # Ratio of available water to water required to use the redirected light
    denom    = 0.037663 * hypotheticalSum    * excessFrac
    denomIrr = 0.037663 * hypotheticalSumIrr * excessFracIrr
    excessRatio    = np.where(denom    > 0, excessWater / denom,    0.0)
    excessRatioIrr = np.where(denomIrr > 0, leftover    / denomIrr, 0.0)

    addedNPP    = storage * hypotheticalSum    * excessFrac    * np.minimum(1.0, excessRatio)
    addedNPPirr = storage * hypotheticalSumIrr * excessFracIrr * np.minimum(1.0, excessRatioIrr)

    return addedNPP, addedNPPirr


def _find_local(
    waterflux, waterflux_irr, waterNeed, growperiod,
    farmed, irrigated, area, excessWater, leftover,
    addedNPP, addedNPPirr, lightmove
):
    """
    For each grid cell, compute the fraction of forgone NPP attributable
    to water scarcity (vs. light scarcity). Returns a 2-D spatial array.
    """
    excessWaterIrr    = farmed * irrigated * area * (waterflux_irr - waterNeed) * growperiod

    insufficientWater    = np.clip(-excessWater,    0, None)
    insufficientWaterIrr = np.clip(-excessWaterIrr, 0, None)
    excessWater_pos      = np.clip( excessWater,    0, None)

    if lightmove:
        addedNPP_sum    = np.sum(addedNPP,    axis=2)
        addedNPPirr_val = addedNPPirr
    else:
        addedNPP_sum    = 0.0
        addedNPPirr_val = 0.0

    forgoneNPPdue2PAR = (
        (1.0 - irrigated) * (1.0 / 0.037663 * np.sum(excessWater_pos, axis=2) - addedNPP_sum)
        + irrigated        * (1.0 / 0.037663 * leftover - addedNPPirr_val)
    )
    forgoneNPPdue2water = (
        (1.0 - irrigated) * (1.0 / 0.037663 * np.sum(insufficientWater,    axis=2))
        + irrigated        * (1.0 / 0.037663 * np.sum(insufficientWaterIrr, axis=2))
    )

    forgoneNPP = forgoneNPPdue2PAR + forgoneNPPdue2water
    local_limits = np.where(forgoneNPP > 0, forgoneNPPdue2water / forgoneNPP, 0.0)

    return local_limits
